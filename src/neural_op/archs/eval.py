import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# ── Helpers compartilhados ────────────────────────────────────────────────────

def _imshow(ax, data, title, cmap='viridis', vmin=None, vmax=None):
    im = ax.imshow(data, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _scatter(ax, r, c, values, title, cmap='viridis', vmin=None, vmax=None, s=6):
    """
    Scatter de nós na posição real (r_base, c_base) — usado para dados de malha
    real do FEMM (mode='femm_mesh') em vez de rasterizar em H×W: a malha tem
    densidade muito variável (refina perto de interfaces), e rasterizar dilui o
    erro em regiões densas e cria buracos em regiões esparsas. NaN em `values`
    usa cmap.set_bad (mesma convenção do cmap_nan já usado em _imshow).
    Ordena por valor crescente para que os valores mais altos (ex: erro maior)
    sejam desenhados por cima em regiões de sobreposição.

    `s` pode ser escalar (tamanho fixo) ou array do mesmo tamanho de `values`
    (tamanho por ponto — usado nos painéis de erro para que erro maior também
    seja um ponto maior, não só mais quente na cor; sem isso, a concentração de
    erro nas quinas de alta densidade de malha fica visualmente afogada entre
    milhares de pontos do mesmo tamanho).
    """
    order = np.argsort(np.where(np.isfinite(values), values, -np.inf))
    s_arr = s[order] if hasattr(s, '__len__') else s
    sc = ax.scatter(c[order], r[order], c=values[order], cmap=cmap,
                     vmin=vmin, vmax=vmax, s=s_arr, edgecolors='none')
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


def _err_sizes(en, vmax, s_min=3, s_max=50):
    """Tamanho do marcador crescente com o erro (área ~ erro/vmax), para os
    painéis de erro em _plot_fno_gnn_mesh. `en` pode ter NaN (nós irrelevantes,
    fora da máscara) — tratados como erro 0 (tamanho mínimo)."""
    v = np.where(np.isfinite(en), en, 0.0)
    norm = np.clip(v / max(vmax, 1e-9), 0.0, 1.0)
    return s_min + (s_max - s_min) * norm


def _masked_metrics(err, mask, B_ref):
    e = err[mask] / B_ref * 100
    return np.mean(e), np.median(e), np.percentile(e, 95)


def _err_norm(err, mask, B_ref):
    return np.where(mask, err / B_ref * 100, np.nan)


def _err_display(err, mask, B_ref, mode):
    """
    Painel de erro alternável: 'percent' (normalizado por B_ref, %) ou
    'absolute' (erro bruto |pred-alvo|, mesma unidade do campo — Tesla).
    Retorna (array com NaN fora da máscara, label do painel).
    """
    if mode == 'absolute':
        return np.where(mask, err, np.nan), 'Erro absoluto [T]'
    return _err_norm(err, mask, B_ref), 'Erro norm. [%]'


def _err_vmax(disps, eval_cfg):
    """
    vmax do colormap de erro. error_cap só se aplica no modo 'percent' (é definido
    em % — não tem tradução direta para o modo 'absolute', que sempre auto-escala).
    """
    if eval_cfg.error_plot_mode == 'percent' and eval_cfg.error_cap_enabled:
        return eval_cfg.error_cap
    return max(float(np.nanmax(d)) if np.isfinite(d).any() else 1.0 for d in disps)


# ── Helpers FNO_GNN ───────────────────────────────────────────────────────────

# [REMOVIDO] _qtree_render — projetava nós via scatter_add para H×W (média por célula base),
# perdendo a estrutura real das folhas refinadas. Substituído por _qtree_render_as_grid.
# def _qtree_render(values, node_x, H, W): ...

def _qtree_render(values, node_x, H, W):
    """Projeta nós para H×W via scatter_add. Usado apenas para métricas escalares."""
    C    = values.shape[1]
    rows = (node_x[:, 3] * H).long().clamp(0, H - 1)
    cols = (node_x[:, 4] * W).long().clamp(0, W - 1)
    flat = rows * W + cols
    grid  = torch.zeros(H * W, C)
    count = torch.zeros(H * W, 1)
    grid.scatter_add_(0, flat.unsqueeze(1).expand_as(values), values)
    count.scatter_add_(0, flat.unsqueeze(1), torch.ones(len(flat), 1))
    count.clamp_(min=1)
    return (grid / count).view(H, W, C).permute(2, 0, 1)


# [REMOVIDO] _node_density_map — rasterizava contagem de nós em pixels H×W para o
# painel "Densidade de nós (malha)". Substituído por scatter direto dos nós
# (ver _plot_fno_gnn_mesh) — a própria densidade de pontos no scatter já mostra
# onde a malha refina mais, sem precisar binar em pixels.
# def _node_density_map(node_x, H, W):
#     rows  = (node_x[:, 3] * H).long().clamp(0, H - 1)
#     cols  = (node_x[:, 4] * W).long().clamp(0, W - 1)
#     flat  = rows * W + cols
#     count = torch.zeros(H * W)
#     count.scatter_add_(0, flat, torch.ones(len(flat)))
#     return count.view(H, W).numpy()


def _qtree_render_as_grid(values, node_x, H, W):
    """
    Renderiza nós quadtree na extensão espacial real de cada folha.

    Simula o DFS a partir dos depths derivados de cell_area (node_x[:,2]),
    preenchendo uma grade H·2^D_max × W·2^D_max onde cada folha ocupa seu
    retângulo exato. A ordem DFS (TL/TR/BL/BR, row-major sobre células base)
    é idêntica à de _qtree_dfs em motor_model.py.

    values  : [S, C]  — valores a renderizar (predição ou GT nos nós)
    node_x  : [S, N]  — col 2=cell_area=(1/2^d)², col 3=r_base, col 4=c_base
    H, W    : dimensões da grade base

    Retorna : [C, H*s, W*s]  onde s = 2^D_max
    """
    S = len(values)
    C = values.shape[1]

    if S == 0:
        return torch.zeros(C, H, W)

    cell_area = node_x[:, 2].numpy().clip(1e-12)
    # cell_area = 4^(-d)  →  d = -log2(cell_area) / 2
    depths = np.round(-np.log2(cell_area) / 2).astype(np.int64)
    D_max  = int(depths.max())
    scale  = 1 << D_max          # 2^D_max

    HR, WR = H * scale, W * scale
    val_np = values.numpy()
    grid   = np.zeros((HR, WR, C), dtype=np.float32)
    cursor = [0]

    def _fill(r0, c0, side, level):
        if cursor[0] >= S:
            return
        d = int(depths[cursor[0]])
        if d == level:
            grid[r0:r0 + side, c0:c0 + side] = val_np[cursor[0]]
            cursor[0] += 1
        else:
            half = side >> 1
            _fill(r0,        c0,        half, level + 1)  # TL
            _fill(r0,        c0 + half, half, level + 1)  # TR
            _fill(r0 + half, c0,        half, level + 1)  # BL
            _fill(r0 + half, c0 + half, half, level + 1)  # BR

    for ir in range(H):
        for ia in range(W):
            _fill(ir * scale, ia * scale, scale, 0)

    return torch.from_numpy(grid).permute(2, 0, 1)  # [C, HR, WR]


# [REMOVIDO] _add_qtree_overlay — sobrepunha cruzes (+) nas células refinadas;
# substituído por renderização real da quadtree em _qtree_render_as_grid.
# def _add_qtree_overlay(ax, refined_map): ...


# ── Funções de avaliação por arquitetura ──────────────────────────────────────

def fno_eval_fn(model, d, eval_cfg):
    """
    Eval FNO2d: carrega amostra, infere e plota grid 3×4.
    d        : dict carregado de quad_chunk_*.pt
    eval_cfg : EvalCfg
    """
    i   = eval_cfg.sample_idx
    thr = eval_cfg.irrelevance_threshold

    x = d['x_hw'][i]   # [C_in, H, W]
    y = d['y_hw'][i]   # [C_out, H, W]

    with torch.no_grad():
        pred = model(x.unsqueeze(0)).squeeze(0)   # [C_out, H, W]

    C_in  = x.shape[0]
    C_out = y.shape[0]
    motor = (C_in == 2 and C_out == 2)

    in_labels  = ['Mu_avg', 'M']  if motor else [f'in_{k}'  for k in range(C_in)]
    out_labels = ['Bx',     'By'] if motor else [f'out_{k}' for k in range(C_out)]

    # [REMOVIDO] labels hardcoded para motor — substituídos por detecção automática de canais
    # mu      = x[0].numpy()
    # mag_m   = x[1].numpy()
    # bx_true = y[0].numpy();    by_true = y[1].numpy()
    # bx_pred = pred[0].numpy(); by_pred = pred[1].numpy()

    if C_out >= 2:
        mag_true = (y[0]**2    + y[1]**2).sqrt().numpy()
        mag_pred = (pred[0]**2 + pred[1]**2).sqrt().numpy()
    else:
        mag_true = y[0].abs().numpy()
        mag_pred = pred[0].abs().numpy()
    mag_err = np.abs(mag_pred - mag_true)

    mask  = mag_true >= thr
    B_ref = np.sqrt(np.mean(mag_true[mask]**2)) if mask.any() else 1.0

    en, en_label = _err_display(mag_err, mask, B_ref, eval_cfg.error_plot_mode)
    m, med, p95 = _masked_metrics(mag_err, mask, B_ref)
    cmap_nan = plt.cm.hot.copy(); cmap_nan.set_bad(color='#444444')

    # ── Métrica opcional: projeta a predição (grade base) nas folhas da qtree ────
    # e compara contra node_y (GT exato por folha, nunca promediado). Revela o
    # erro que a saída em grade uniforme escapa em células mistas/interfaces.
    # Cada folha conta como 1 amostra (sem peso por área).
    # 'node_A' in d -> chunk é malha real do FEMM (mode='femm_mesh'): node_x[:,2] ali
    # é node_dual_area (mm² real), não cell_area (potência de 4) como no qtree —
    # _qtree_render_as_grid/depths abaixo pressupõem qtree e dariam lixo nesse caso.
    qtree_ok = (eval_cfg.qtree_metric_enabled and ('node_x' in d) and ('node_y' in d)
                and ('node_A' not in d))
    en_qt_render = None
    if qtree_ok:
        H, W  = int(d['dim'][0]), int(d['dim'][1])
        L     = d['L']
        n_off = torch.cat([torch.zeros(1, dtype=torch.long), L.cumsum(0)])
        ns, ne = int(n_off[i]), int(n_off[i + 1])

        node_x_i = d['node_x'][ns:ne]   # [S_i, 5] mu_r, M, cell_area, r_base, c_base
        node_y_i = d['node_y'][ns:ne]   # [S_i, 2] Bx, By — valor exato por folha

        rows = (node_x_i[:, 3] * H).long().clamp(0, H - 1)
        cols = (node_x_i[:, 4] * W).long().clamp(0, W - 1)

        # nearest-neighbor: a folha herda a predição da célula base que a contém
        pred_qt = pred[:, rows, cols].T   # [S_i, C_out]

        if pred_qt.shape[1] >= 2:
            mag_true_qt = (node_y_i[:, 0]**2 + node_y_i[:, 1]**2).sqrt().numpy()
            mag_pred_qt = (pred_qt[:, 0]**2  + pred_qt[:, 1]**2).sqrt().numpy()
        else:
            mag_true_qt = node_y_i[:, 0].abs().numpy()
            mag_pred_qt = pred_qt[:, 0].abs().numpy()
        err_qt = np.abs(mag_pred_qt - mag_true_qt)

        mask_qt  = mag_true_qt >= thr
        B_ref_qt = np.sqrt(np.mean(mag_true_qt[mask_qt]**2)) if mask_qt.any() else 1.0
        qt_m, qt_med, qt_p95 = _masked_metrics(err_qt, mask_qt, B_ref_qt)

        print(f"FNO (grade H×W)      — média={m:.1f}%  mediana={med:.1f}%  p95={p95:.1f}%")
        print(f"FNO (qtree refinada) — média={qt_m:.1f}%  mediana={qt_med:.1f}%  p95={qt_p95:.1f}%  "
              f"({int(mask_qt.sum())} folhas relevantes / {len(mask_qt)})")

        en_qt, en_qt_label = _err_display(err_qt, mask_qt, B_ref_qt, eval_cfg.error_plot_mode)
        en_qt = en_qt.astype(np.float32)
        en_qt_render = _qtree_render_as_grid(
            torch.from_numpy(en_qt).unsqueeze(1), node_x_i, H, W
        )[0].numpy()

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    fig.suptitle(
        f"FNO2d — amostra {i}  |  ref={B_ref:.4f}  "
        f"região relevante: {mask.mean()*100:.1f}%",
        fontsize=12,
    )

    # Linha 0 — entradas
    _imshow(axes[0, 0], x[0].numpy(), f'Entrada: {in_labels[0]}')
    if C_in >= 2:
        _imshow(axes[0, 1], x[1].numpy(), f'Entrada: {in_labels[1]}')
    else:
        axes[0, 1].axis('off')
    axes[0, 2].imshow(mask.astype(float), origin='lower', aspect='auto',
                      cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Máscara |y|>={thr}  (branco=relevante)', fontsize=9)
    if en_qt_render is not None:
        vmax_qt = _err_vmax([en_qt_render], eval_cfg)
        _imshow(axes[0, 3], en_qt_render,
                f'{en_qt_label} qtree refinada\nmédia={qt_m:.1f}%  p95={qt_p95:.1f}%',
                cmap=cmap_nan, vmin=0, vmax=vmax_qt)
    else:
        axes[0, 3].axis('off')

    # Linha 1 — alvo
    _imshow(axes[1, 0], y[0].numpy(), f'Alvo: {out_labels[0]}')
    if C_out >= 2:
        _imshow(axes[1, 1], y[1].numpy(), f'Alvo: {out_labels[1]}')
    else:
        axes[1, 1].axis('off')
    _imshow(axes[1, 2], mag_true, 'Alvo: |y|')
    axes[1, 3].axis('off')

    # Linha 2 — predição
    _imshow(axes[2, 0], pred[0].numpy(), f'Pred: {out_labels[0]}')
    if C_out >= 2:
        _imshow(axes[2, 1], pred[1].numpy(), f'Pred: {out_labels[1]}')
    else:
        axes[2, 1].axis('off')
    err_vmax = _err_vmax([en], eval_cfg)
    _imshow(axes[2, 2], mag_err, 'Erro absoluto |y| [T]')
    ref_note = f'  (÷ ref={B_ref:.3f})' if eval_cfg.error_plot_mode == 'percent' else ''
    _imshow(axes[2, 3], en,
            f'{en_label} global{ref_note}\n'
            f'média={m:.1f}%  mediana={med:.1f}%  p95={p95:.1f}%',
            cmap=cmap_nan, vmin=0, vmax=err_vmax)
    plt.tight_layout()
    plt.show()


def _plot_fno_gnn_mesh(model, y_hw_fno, y_nodes, node_x, node_y, Li, thr, eval_cfg, i):
    """
    Eval FNO_GNN/GNN_PostBase para malha real do FEMM (mode='femm_mesh') — plota
    só os nós reais (scatter em r_base/c_base), sem rasterizar em H×W. A malha
    tem densidade muito variável (refina perto de interfaces), e rasterizar
    dilui o erro em pixels densos e cria buracos em pixels sem nó nenhum.

    Erro sempre nó-a-nó: node_y é o GT exato por nó; a saída do FNO (grade
    regular) é interpolada para a posição exata de cada nó via
    _interpolate_fno_to_nodes (mesma interpolação usada internamente pelo
    modelo) antes de comparar — FNO, GNN e GT ficam no mesmo conjunto de pontos.

    node_x: col 0=mu_r, col 1=M, col 2=node_dual_area, col 3=r_base, col 4=c_base

    Suporta alvo vetorial (node_y [S,2] = Bx,By, ex: FEMM_MESH_PARSER) ou
    escalar (node_y [S,1] = A, ex: FEMM_MESH_A_PARSER) — detectado via
    node_y.shape[1]. No caso escalar os painéis de By/|B| ficam em branco
    (axis('off')), só a coluna 0 (valor) e a de erro são preenchidas.
    """
    from src.neural_op.archs.fno_gnn import _interpolate_fno_to_nodes

    r = node_x[:, 3].numpy()
    c = node_x[:, 4].numpy()

    mu_n = node_x[:, 0].numpy()
    m_n  = node_x[:, 1].numpy()

    vec = node_y.shape[1] >= 2   # False -> alvo escalar (A), True -> vetorial (Bx,By)
    c0_label = 'Bx' if vec else 'A'

    fno_at_nodes = _interpolate_fno_to_nodes(y_hw_fno, node_x, Li)

    c0_true, c0_fno, c0_gnn = node_y[:, 0].numpy(), fno_at_nodes[:, 0].numpy(), y_nodes[:, 0].numpy()
    if vec:
        c1_true, c1_fno, c1_gnn = node_y[:, 1].numpy(), fno_at_nodes[:, 1].numpy(), y_nodes[:, 1].numpy()
        mag_true = np.sqrt(c0_true**2 + c1_true**2)
        mag_fno  = np.sqrt(c0_fno**2  + c1_fno**2)
        mag_gnn  = np.sqrt(c0_gnn**2  + c1_gnn**2)
    else:
        c1_true = c1_fno = c1_gnn = None
        mag_true, mag_fno, mag_gnn = np.abs(c0_true), np.abs(c0_fno), np.abs(c0_gnn)

    err_fno  = np.abs(mag_fno - mag_true)
    err_gnn  = np.abs(mag_gnn - mag_true)

    mask  = mag_true >= thr
    B_ref = np.sqrt(np.mean(mag_true[mask]**2)) if mask.any() else 1.0

    fno_m, fno_med, fno_p95 = _masked_metrics(err_fno, mask, B_ref)
    gnn_m, gnn_med, gnn_p95 = _masked_metrics(err_gnn, mask, B_ref)
    print(f"B_ref = {B_ref:.4f}  |  região relevante: {mask.mean()*100:.1f}%  "
          f"({int(mask.sum())} nós relevantes / {len(mask)})")
    print(f"FNO  — média={fno_m:.1f}%  mediana={fno_med:.1f}%  p95={fno_p95:.1f}%")
    print(f"GNN  — média={gnn_m:.1f}%  mediana={gnn_med:.1f}%  p95={gnn_p95:.1f}%")

    en_fno, en_label = _err_display(err_fno, mask, B_ref, eval_cfg.error_plot_mode)
    en_gnn, _        = _err_display(err_gnn, mask, B_ref, eval_cfg.error_plot_mode)
    cmap_nan = plt.cm.hot.copy(); cmap_nan.set_bad(color='#444444')
    err_vmax = _err_vmax([en_fno, en_gnn], eval_cfg)

    def _lim(*a): return min(x.min() for x in a), max(x.max() for x in a)
    c0_lim  = _lim(c0_true, c0_fno, c0_gnn)
    c1_lim  = _lim(c1_true, c1_fno, c1_gnn) if vec else None
    mag_lim = _lim(mag_true, mag_fno, mag_gnn)

    fig, axes = plt.subplots(4, 4, figsize=(18, 13))
    fig.suptitle(
        f"{type(model).__name__} — amostra {i} (malha, {len(mag_true)} nós)  |  "
        f"B_ref={B_ref:.4f}  região relevante: {mask.mean()*100:.1f}%",
        fontsize=12,
    )

    # Linha 0 — entradas (nós) + máscara + densidade de nós
    _scatter(axes[0, 0], r, c, mu_n, 'Entrada: Mu_r (nós)')
    _scatter(axes[0, 1], r, c, m_n,  'Entrada: M (nós)')
    _scatter(axes[0, 2], r, c, mask.astype(float), f'Máscara |{c0_label}|>={thr}',
             cmap='gray', vmin=0, vmax=1)
    axes[0, 3].scatter(c, r, s=5, color='steelblue', alpha=0.4, edgecolors='none')
    axes[0, 3].set_title(f'Densidade de nós (malha) — {len(r)} nós', fontsize=9)
    axes[0, 3].set_xlim(0, 1); axes[0, 3].set_ylim(0, 1)

    # Linhas 1-3 — GT / FNO@nós / GNN, coluna 0 sempre preenchida (Bx ou A);
    # colunas 1 (By) e 2 (|B|) só existem no caso vetorial — ver docstring.
    for row, (label, v0, v1, vmag) in enumerate([
        ('GT',      c0_true, c1_true, mag_true),
        ('FNO@nós', c0_fno,  c1_fno,  mag_fno),
        ('GNN',     c0_gnn,  c1_gnn,  mag_gnn),
    ], start=1):
        _scatter(axes[row, 0], r, c, v0, f'{label}: {c0_label} (nós)', vmin=c0_lim[0], vmax=c0_lim[1])
        if vec:
            _scatter(axes[row, 1], r, c, v1,    f'{label}: By (nós)',  vmin=c1_lim[0],  vmax=c1_lim[1])
            _scatter(axes[row, 2], r, c, vmag,  f'{label}: |B| (nós)', vmin=mag_lim[0], vmax=mag_lim[1])
        else:
            axes[row, 1].axis('off')
            axes[row, 2].axis('off')

    axes[1, 3].axis('off')

    size_fno = _err_sizes(en_fno, err_vmax)
    _scatter(axes[2, 3], r, c, en_fno,
             f'FNO {en_label} (nó-a-nó)\nmédia={fno_m:.1f}%  p95={fno_p95:.1f}%',
             cmap=cmap_nan, vmin=0, vmax=err_vmax, s=size_fno)

    size_gnn = _err_sizes(en_gnn, err_vmax)
    _scatter(axes[3, 3], r, c, en_gnn,
             f'GNN {en_label} (nó-a-nó)\nmédia={gnn_m:.1f}%  p95={gnn_p95:.1f}%',
             cmap=cmap_nan, vmin=0, vmax=err_vmax, s=size_gnn)

    plt.tight_layout()
    plt.show()


def fno_gnn_eval_fn(model, d, eval_cfg):
    """
    Eval FNO_GNN/GNN_PostBase: carrega amostra + grafo, infere e plota.

    Suporta dois layouts de grafo (mesmo arch, dataset decide via 'node_A' in d):
    - qtree (mode='qtree'): node_x[:,2]=cell_area, folhas alinhadas em retângulos
      exatos da grade base → renderizado via _qtree_render_as_grid (DFS). Grid 4×4:
      linha 0 = Mu_r/M (qtree) + máscara + depth map; linha 1 = GT (qtree);
      linha 2 = FNO (H×W) + erro; linha 3 = GNN (qtree) + erro (H×W vs GT grid).
    - malha real do FEMM (mode='femm_mesh', chunk tem chave 'node_A'): despachado
      para _plot_fno_gnn_mesh — a malha não é uma subdivisão em retângulos e tem
      densidade muito variável, então todo o plot é por scatter dos nós reais,
      sem rasterizar em H×W, com erro sempre nó-a-nó contra node_y.
    """
    i   = eval_cfg.sample_idx
    thr = eval_cfg.irrelevance_threshold
    is_mesh = 'node_A' in d

    L, E_L = d['L'], d['E_L']
    H, W   = int(d['dim'][0]), int(d['dim'][1])

    n_off = torch.cat([torch.zeros(1, dtype=torch.long), L.cumsum(0)])
    e_off = torch.cat([torch.zeros(1, dtype=torch.long), E_L.cumsum(0)])
    ns, ne = int(n_off[i]), int(n_off[i + 1])
    es, ee = int(e_off[i]), int(e_off[i + 1])

    x_hw       = d['x_hw'][i:i + 1]
    y_hw       = d['y_hw'][i:i + 1]
    y_hw_grid  = d['y_hw_grid'][i:i + 1] if 'y_hw_grid' in d else y_hw  # métricas H×W

    node_x     = d['node_x'][ns:ne]
    node_y     = d['node_y'][ns:ne]
    edge_index = d['edge_index'][:, es:ee] - ns
    edge_attr  = d['edge_attr'][es:ee]
    Li         = L[i:i + 1]

    with torch.no_grad():
        y_hw_fno, y_nodes = model(x_hw, node_x, edge_index, edge_attr, Li)

    if is_mesh:
        _plot_fno_gnn_mesh(model, y_hw_fno, y_nodes, node_x, node_y, Li, thr, eval_cfg, i)
        return

    # ── Renders dos nós (input, GT, predição GNN) — qtree ────────────────────
    # node_x: col 0=mu_r, col 1=M, col 2=cell_area, col 3=r_base, col 4=c_base
    node_x_qt  = _qtree_render_as_grid(node_x[:, :2], node_x, H, W)   # [2, H*s, W*s]
    node_y_qt  = _qtree_render_as_grid(node_y[:, :2], node_x, H, W)   # [2, H*s, W*s]
    y_nodes_qt = _qtree_render_as_grid(y_nodes,       node_x, H, W)   # [2, H*s, W*s]

    # depth map: profundidade de cada folha → mostra onde a qtree refina
    cell_area  = node_x[:, 2].clamp(min=1e-12)
    depths_val = torch.round(-torch.log2(cell_area) / 2).unsqueeze(1)  # [S, 1]
    depth_map      = _qtree_render_as_grid(depths_val, node_x, H, W)[0].numpy()
    depth_map_title = 'Depth map (qtree)'

    # ── Métricas: scatter_add H×W contra y_hw_grid ───────────────────────────
    y_nodes_hw = _qtree_render(y_nodes, node_x, H, W)   # [2, H, W]

    mu_qt  = node_x_qt[0].numpy();     m_qt   = node_x_qt[1].numpy()
    bx_tq  = node_y_qt[0].numpy();     by_tq  = node_y_qt[1].numpy()
    bx_gnn = y_nodes_qt[0].numpy();    by_gnn = y_nodes_qt[1].numpy()
    bx_fno = y_hw_fno[0, 0].numpy();   by_fno = y_hw_fno[0, 1].numpy()

    bx_true   = y_hw_grid[0, 0].numpy(); by_true   = y_hw_grid[0, 1].numpy()
    bx_gnn_hw = y_nodes_hw[0].numpy();   by_gnn_hw = y_nodes_hw[1].numpy()

    mag_tq     = np.sqrt(bx_tq**2     + by_tq**2)
    mag_fno    = np.sqrt(bx_fno**2    + by_fno**2)
    mag_gnn    = np.sqrt(bx_gnn**2    + by_gnn**2)
    mag_true   = np.sqrt(bx_true**2   + by_true**2)
    mag_gnn_hw = np.sqrt(bx_gnn_hw**2 + by_gnn_hw**2)
    err_fno    = np.abs(mag_fno    - mag_true)
    err_gnn    = np.abs(mag_gnn_hw - mag_true)

    mask  = mag_true >= thr
    B_ref = np.sqrt(np.mean(mag_true[mask]**2)) if mask.any() else 1.0

    fno_m, fno_med, fno_p95 = _masked_metrics(err_fno, mask, B_ref)
    gnn_m, gnn_med, gnn_p95 = _masked_metrics(err_gnn, mask, B_ref)
    print(f"B_ref = {B_ref:.4f}  |  região relevante: {mask.mean()*100:.1f}%")
    print(f"FNO  — média={fno_m:.1f}%  mediana={fno_med:.1f}%  p95={fno_p95:.1f}%")
    print(f"GNN  — média={gnn_m:.1f}%  mediana={gnn_med:.1f}%  p95={gnn_p95:.1f}%")

    en_fno, en_fno_label = _err_display(err_fno, mask, B_ref, eval_cfg.error_plot_mode)
    en_gnn, en_gnn_label = _err_display(err_gnn, mask, B_ref, eval_cfg.error_plot_mode)
    cmap_nan = plt.cm.hot.copy(); cmap_nan.set_bad(color='#444444')

    def _lim(*a): return min(x.min() for x in a), max(x.max() for x in a)
    bx_lim  = _lim(bx_tq, bx_fno, bx_gnn)
    by_lim  = _lim(by_tq, by_fno, by_gnn)
    mag_lim = _lim(mag_tq, mag_fno, mag_gnn)
    err_vmax = _err_vmax([en_fno, en_gnn], eval_cfg)

    tag = '(qtree)'

    fig, axes = plt.subplots(4, 4, figsize=(18, 13))
    fig.suptitle(
        f"{type(model).__name__} — amostra {i}  |  B_ref={B_ref:.4f}  "
        f"região relevante: {mask.mean()*100:.1f}%",
        fontsize=12,
    )

    # Linha 0 — entradas (nós) + depth map / densidade de nós
    _imshow(axes[0, 0], mu_qt, f'Entrada: Mu_r {tag}')
    _imshow(axes[0, 1], m_qt,  f'Entrada: M {tag}')
    axes[0, 2].imshow(mask.astype(float), origin='lower', aspect='auto',
                      cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Máscara |B|≥{thr}', fontsize=9)
    _imshow(axes[0, 3], depth_map, depth_map_title, cmap='Blues')

    # Linha 1 — GT nos nós
    _imshow(axes[1, 0], bx_tq,  f'GT: Bx {tag}',  vmin=bx_lim[0],  vmax=bx_lim[1])
    _imshow(axes[1, 1], by_tq,  f'GT: By {tag}',  vmin=by_lim[0],  vmax=by_lim[1])
    _imshow(axes[1, 2], mag_tq, f'GT: |B| {tag}', vmin=mag_lim[0], vmax=mag_lim[1])
    axes[1, 3].axis('off')

    # Linha 2 — FNO (grade H×W)
    _imshow(axes[2, 0], bx_fno,  'FNO pred: Bx',  vmin=bx_lim[0],  vmax=bx_lim[1])
    _imshow(axes[2, 1], by_fno,  'FNO pred: By',  vmin=by_lim[0],  vmax=by_lim[1])
    _imshow(axes[2, 2], mag_fno, 'FNO pred: |B|', vmin=mag_lim[0], vmax=mag_lim[1])
    _imshow(axes[2, 3], en_fno,
            f'FNO {en_fno_label}\nmédia={fno_m:.1f}%  p95={fno_p95:.1f}%',
            cmap=cmap_nan, vmin=0, vmax=err_vmax)

    # Linha 3 — GNN (nós reais); erro em H×W
    _imshow(axes[3, 0], bx_gnn,  f'GNN pred: Bx {tag}',  vmin=bx_lim[0],  vmax=bx_lim[1])
    _imshow(axes[3, 1], by_gnn,  f'GNN pred: By {tag}',  vmin=by_lim[0],  vmax=by_lim[1])
    _imshow(axes[3, 2], mag_gnn, f'GNN pred: |B| {tag}', vmin=mag_lim[0], vmax=mag_lim[1])
    _imshow(axes[3, 3], en_gnn,
            f'GNN {en_gnn_label} (H×W vs GT grid)\nmédia={gnn_m:.1f}%  p95={gnn_p95:.1f}%',
            cmap=cmap_nan, vmin=0, vmax=err_vmax)

    plt.tight_layout()
    plt.show()


def single_mat_fno_eval_fn(model, d, eval_cfg):
    """
    Eval FNO2d_SingleMat: infere e plota grid 4×4.
    Lê material_id do modelo via atributo, ou usa eval_cfg se disponível.

    Linha 0: Mu_r, M, máscara do material alvo, mapa de materiais (4 cores)
    Linha 1: GT Bx, GT By, GT |B| (domínio), GT |B| (só material alvo)
    Linha 2: Pred Bx, Pred By, Pred |B| (domínio), Pred |B| (só material alvo)
    Linha 3: Erro abs |B| (domínio), Erro norm [%] (só material alvo), —, —
    """
    from src.neural_op.archs.fno_mat import _make_material_masks

    _labels = ['ferro', 'ar', 'ima', 'cobre']

    i   = eval_cfg.sample_idx
    thr = eval_cfg.irrelevance_threshold

    x = d['x_hw'][i]   # [2, H, W]
    y = d['y_hw'][i]   # [2, H, W]

    with torch.no_grad():
        pred  = model(x.unsqueeze(0)).squeeze(0)                      # [2, H, W]
        masks = _make_material_masks(x[0].unsqueeze(0)).squeeze(0)    # [4, H, W]

    # material_id armazenado como atributo na instância do modelo (injetado pelo __init__.py)
    material_id = getattr(model, '_single_mat_id', 0)
    mat_name    = _labels[material_id]
    mask_m      = masks[material_id].numpy()   # [H, W] bool

    bx_true = y[0].numpy();         by_true = y[1].numpy()
    bx_pred = pred[0].numpy();      by_pred = pred[1].numpy()
    mag_true = np.sqrt(bx_true ** 2 + by_true ** 2)
    mag_pred = np.sqrt(bx_pred ** 2 + by_pred ** 2)
    mag_err  = np.abs(mag_pred - mag_true)

    # máscara de relevância restrita ao material alvo
    mask_rel = mask_m & (mag_true >= thr)
    B_ref    = np.sqrt(np.mean(mag_true[mask_rel] ** 2)) if mask_rel.any() else 1.0

    # mapa categórico de materiais [H, W]
    mat_map = torch.zeros(masks.shape[1], masks.shape[2], dtype=torch.long)
    for m in range(4):
        mat_map[masks[m]] = m
    mat_map = mat_map.numpy()

    en, en_label = _err_display(mag_err, mask_rel, B_ref, eval_cfg.error_plot_mode)
    m_, med, p95 = _masked_metrics(mag_err, mask_rel, B_ref)
    cmap_nan    = plt.cm.hot.copy(); cmap_nan.set_bad(color='#444444')

    mag_true_mat = np.where(mask_m, mag_true, np.nan)
    mag_pred_mat = np.where(mask_m, mag_pred, np.nan)

    fig, axes = plt.subplots(4, 4, figsize=(18, 13))
    fig.suptitle(
        f"FNO2d_SingleMat [{mat_name}] — amostra {i}  |  ref={B_ref:.4f}  "
        f"região {mat_name}: {mask_m.mean()*100:.1f}%",
        fontsize=12,
    )

    # Linha 0 — entradas + máscara do material + mapa geral
    _imshow(axes[0, 0], x[0].numpy(), 'Entrada: Mu_r')
    _imshow(axes[0, 1], x[1].numpy(), 'Entrada: M')
    axes[0, 2].imshow(mask_m.astype(float), origin='lower', aspect='auto',
                      cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Máscara: {mat_name}  (branco=alvo)', fontsize=9)
    im   = axes[0, 3].imshow(mat_map, origin='lower', aspect='auto',
                              cmap=_MAT_CMAP, norm=_MAT_NORM)
    cbar = plt.colorbar(im, ax=axes[0, 3], ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(_labels)
    axes[0, 3].set_title('Materiais', fontsize=9)

    # Linha 1 — alvo
    _imshow(axes[1, 0], bx_true,  'GT: Bx')
    _imshow(axes[1, 1], by_true,  'GT: By')
    _imshow(axes[1, 2], mag_true, 'GT: |B| (domínio)')
    _imshow(axes[1, 3], mag_true_mat, f'GT: |B| ({mat_name})', cmap='viridis')

    # Linha 2 — predição (espelha GT)
    _imshow(axes[2, 0], bx_pred,      'Pred: Bx')
    _imshow(axes[2, 1], by_pred,      'Pred: By')
    _imshow(axes[2, 2], mag_pred,     'Pred: |B| (domínio)')
    _imshow(axes[2, 3], mag_pred_mat, f'Pred: |B| ({mat_name})', cmap='viridis')

    # Linha 3 — erros
    _imshow(axes[3, 0], mag_err, 'Erro abs |B| (domínio) [T]')
    err_vmax = _err_vmax([en], eval_cfg)
    ref_note = f'  ref={B_ref:.3f}' if eval_cfg.error_plot_mode == 'percent' else ''
    _imshow(axes[3, 1], en,
            f'{en_label}{ref_note}\n'
            f'média={m_:.1f}%  mediana={med:.1f}%  p95={p95:.1f}%',
            cmap=cmap_nan, vmin=0, vmax=err_vmax)
    axes[3, 2].axis('off')
    axes[3, 3].axis('off')

    plt.tight_layout()
    plt.show()


_MAT_LABELS = ['ferro', 'ar', 'ima', 'cobre']
_MAT_CMAP   = ListedColormap(['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c'])
_MAT_NORM   = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)


def masked_fno_eval_fn(model, d, eval_cfg):
    """
    Eval MaskedFNO2d: inferência + montagem dos 8 canais + plot 3×4.
    Linha 0: Mu_r, M, mapa de materiais (4 cores), vazio
    Linha 1: GT Bx, GT By, GT |B|, vazio
    Linha 2: Pred Bx (assembled), Pred By, erro absoluto |B|, erro normalizado [%]
    """
    from src.neural_op.archs.fno_mat import _make_material_masks

    i   = eval_cfg.sample_idx
    thr = eval_cfg.irrelevance_threshold

    x = d['x_hw'][i]   # [2, H, W]
    y = d['y_hw'][i]   # [2, H, W]

    with torch.no_grad():
        pred8     = model(x.unsqueeze(0))                          # [1, 8, H, W]
        masks     = _make_material_masks(x[0].unsqueeze(0))          # [1, 4, H, W]
        assembled = model.assemble(pred8, masks).squeeze(0)        # [2, H, W]

    pred8 = pred8.squeeze(0)   # [8, H, W]
    masks = masks.squeeze(0)   # [4, H, W]

    bx_true = y[0].numpy();          by_true = y[1].numpy()
    bx_pred = assembled[0].numpy();  by_pred = assembled[1].numpy()
    mag_true = np.sqrt(bx_true ** 2 + by_true ** 2)
    mag_pred = np.sqrt(bx_pred ** 2 + by_pred ** 2)
    mag_err  = np.abs(mag_pred - mag_true)

    mask_rel = mag_true >= thr
    B_ref    = np.sqrt(np.mean(mag_true[mask_rel] ** 2)) if mask_rel.any() else 1.0

    # mapa categórico de materiais [H, W]
    mat_map = torch.zeros(masks.shape[1], masks.shape[2], dtype=torch.long)
    for m in range(4):
        mat_map[masks[m]] = m
    mat_map = mat_map.numpy()

    en, en_label = _err_display(mag_err, mask_rel, B_ref, eval_cfg.error_plot_mode)
    m_, med, p95 = _masked_metrics(mag_err, mask_rel, B_ref)
    cmap_nan    = plt.cm.hot.copy(); cmap_nan.set_bad(color='#444444')

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    fig.suptitle(
        f"MaskedFNO2d — amostra {i}  |  ref={B_ref:.4f}  "
        f"região relevante: {mask_rel.mean() * 100:.1f}%",
        fontsize=12,
    )

    # Linha 0 — entradas + mapa de materiais
    _imshow(axes[0, 0], x[0].numpy(), 'Entrada: Mu_r')
    _imshow(axes[0, 1], x[1].numpy(), 'Entrada: M')
    im   = axes[0, 2].imshow(mat_map, origin='lower', aspect='auto',
                              cmap=_MAT_CMAP, norm=_MAT_NORM)
    cbar = plt.colorbar(im, ax=axes[0, 2], ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(_MAT_LABELS)
    axes[0, 2].set_title('Materiais', fontsize=9)
    axes[0, 3].axis('off')

    # Linha 1 — alvo
    _imshow(axes[1, 0], bx_true,  'GT: Bx')
    _imshow(axes[1, 1], by_true,  'GT: By')
    _imshow(axes[1, 2], mag_true, 'GT: |B|')
    axes[1, 3].axis('off')

    # Linha 2 — predição montada + erro
    _imshow(axes[2, 0], bx_pred, 'Pred: Bx (assembled)')
    _imshow(axes[2, 1], by_pred, 'Pred: By (assembled)')
    err_vmax = _err_vmax([en], eval_cfg)
    ref_note = f'  ref={B_ref:.3f}' if eval_cfg.error_plot_mode == 'percent' else ''
    _imshow(axes[2, 2], mag_err, 'Erro absoluto |B| [T]')
    _imshow(axes[2, 3], en,
            f'{en_label}{ref_note}\n'
            f'média={m_:.1f}%  mediana={med:.1f}%  p95={p95:.1f}%',
            cmap=cmap_nan, vmin=0, vmax=err_vmax)

    plt.tight_layout()
    plt.show()


def masked_fno_gnn_eval_fn(model, d, eval_cfg):
    """
    Eval MaskedFNO_GNN: inferência + assemble (grade e nós) + plot 4×4.
    Linha 0: Mu_r (qtree), M (qtree), mat map (qtree), depth map (qtree)
    Linha 1: GT Bx (qtree), GT By (qtree), GT |B| (qtree), —
    Linha 2: FNO assembled (H×W): Bx, By, |B|, erro
    Linha 3: GNN assembled (qtree): Bx, By, |B|, erro H×W
    """
    i   = eval_cfg.sample_idx
    thr = eval_cfg.irrelevance_threshold

    L, E_L = d['L'], d['E_L']
    H, W   = int(d['dim'][0]), int(d['dim'][1])

    n_off = torch.cat([torch.zeros(1, dtype=torch.long), L.cumsum(0)])
    e_off = torch.cat([torch.zeros(1, dtype=torch.long), E_L.cumsum(0)])
    ns, ne = int(n_off[i]), int(n_off[i + 1])
    es, ee = int(e_off[i]), int(e_off[i + 1])

    x_hw       = d['x_hw'][i:i + 1]          # [1, 2, H, W]
    y_hw       = d['y_hw'][i:i + 1]          # [1, 2, H, W]
    y_hw_grid  = d['y_hw_grid'][i:i + 1] if 'y_hw_grid' in d else y_hw  # métricas H×W

    node_x     = d['node_x'][ns:ne]          # [S_i, 5]
    node_y     = d['node_y'][ns:ne]          # [S_i, 2]
    edge_index = d['edge_index'][:, es:ee] - ns
    edge_attr  = d['edge_attr'][es:ee]
    Li         = L[i:i + 1]

    with torch.no_grad():
        y_hw_8, masks, y_nodes_2 = model(x_hw, node_x, edge_index, edge_attr, Li)
        # y_hw_8:    [1, 8, H, W]  — saída FNO por material
        # masks:     [1, 4, H, W]  — calculado internamente no forward
        # y_nodes_2: [S_i, 2]      — já assemblado (GNN opera pós-assemble)

    fno_assembled  = model.assemble_grid(y_hw_8, masks).squeeze(0)   # [2, H, W]
    node_assembled = y_nodes_2                                         # [S_i, 2]

    # ── Renders em qtree real ────────────────────────────────────────────────
    node_x_qt  = _qtree_render_as_grid(node_x[:, :2], node_x, H, W)   # [2, H*s, W*s]
    node_y_qt  = _qtree_render_as_grid(node_y[:, :2], node_x, H, W)   # [2, H*s, W*s]
    y_nodes_qt = _qtree_render_as_grid(node_assembled, node_x, H, W)  # [2, H*s, W*s]

    # mapa de materiais em qtree derivado de mu_r (node_x[:,0])
    # thresholds: ferro mu>10=0, ima 1.01<mu<10=2, cobre mu<0.9995=3, ar=1
    mu_r_nodes = node_x[:, 0]
    mat_ids = torch.ones(len(mu_r_nodes), dtype=torch.float32)   # ar=1 default
    mat_ids[mu_r_nodes > 10]                             = 0.0   # ferro
    mat_ids[(mu_r_nodes > 1.01) & (mu_r_nodes <= 10)]   = 2.0   # ima
    mat_ids[mu_r_nodes < 0.9995]                         = 3.0   # cobre
    mat_map_qt = _qtree_render_as_grid(mat_ids.unsqueeze(1), node_x, H, W)[0].numpy()

    # depth map
    cell_area  = node_x[:, 2].clamp(min=1e-12)
    depths_val = torch.round(-torch.log2(cell_area) / 2).unsqueeze(1)
    depth_map  = _qtree_render_as_grid(depths_val, node_x, H, W)[0].numpy()

    # [REMOVIDO] y_true_grid — renderizava node_y via scatter_add; nunca foi exibida no plot
    # y_true_grid, _ = _qtree_render(node_y[:, :2], node_x, H, W)

    # ── Métricas: scatter_add H×W contra y_hw_grid ───────────────────────────
    y_nodes_hw = _qtree_render(node_assembled, node_x, H, W)   # [2, H, W]

    mu_qt  = node_x_qt[0].numpy();     m_qt   = node_x_qt[1].numpy()
    bx_tq  = node_y_qt[0].numpy();     by_tq  = node_y_qt[1].numpy()
    bx_gnn = y_nodes_qt[0].numpy();    by_gnn = y_nodes_qt[1].numpy()
    bx_fno = fno_assembled[0].numpy(); by_fno = fno_assembled[1].numpy()

    bx_true   = y_hw_grid[0, 0].numpy(); by_true   = y_hw_grid[0, 1].numpy()
    bx_gnn_hw = y_nodes_hw[0].numpy();   by_gnn_hw = y_nodes_hw[1].numpy()

    mag_tq     = np.sqrt(bx_tq**2     + by_tq**2)
    mag_fno    = np.sqrt(bx_fno**2    + by_fno**2)
    mag_gnn    = np.sqrt(bx_gnn**2    + by_gnn**2)
    mag_true   = np.sqrt(bx_true**2   + by_true**2)
    mag_gnn_hw = np.sqrt(bx_gnn_hw**2 + by_gnn_hw**2)
    err_fno    = np.abs(mag_fno    - mag_true)
    err_gnn    = np.abs(mag_gnn_hw - mag_true)

    mask_rel = mag_true >= thr
    B_ref    = np.sqrt(np.mean(mag_true[mask_rel]**2)) if mask_rel.any() else 1.0

    fno_m, fno_med, fno_p95 = _masked_metrics(err_fno, mask_rel, B_ref)
    gnn_m, gnn_med, gnn_p95 = _masked_metrics(err_gnn, mask_rel, B_ref)
    print(f"B_ref = {B_ref:.4f}  |  região relevante: {mask_rel.mean()*100:.1f}%")
    print(f"FNO  — média={fno_m:.1f}%  mediana={fno_med:.1f}%  p95={fno_p95:.1f}%")
    print(f"GNN  — média={gnn_m:.1f}%  mediana={gnn_med:.1f}%  p95={gnn_p95:.1f}%")

    en_fno, en_fno_label = _err_display(err_fno, mask_rel, B_ref, eval_cfg.error_plot_mode)
    en_gnn, en_gnn_label = _err_display(err_gnn, mask_rel, B_ref, eval_cfg.error_plot_mode)
    cmap_nan = plt.cm.hot.copy(); cmap_nan.set_bad(color='#444444')

    def _lim(*a): return min(x.min() for x in a), max(x.max() for x in a)
    bx_lim  = _lim(bx_tq, bx_fno, bx_gnn)
    by_lim  = _lim(by_tq, by_fno, by_gnn)
    mag_lim = _lim(mag_tq, mag_fno, mag_gnn)
    err_vmax = _err_vmax([en_fno, en_gnn], eval_cfg)

    fig, axes = plt.subplots(4, 4, figsize=(18, 13))
    fig.suptitle(
        f"MaskedFNO_GNN — amostra {i}  |  B_ref={B_ref:.4f}  "
        f"região relevante: {mask_rel.mean()*100:.1f}%",
        fontsize=12,
    )

    # Linha 0 — entradas + mapa de materiais + depth map (todos em qtree)
    _imshow(axes[0, 0], mu_qt, 'Entrada: Mu_r (qtree)')
    _imshow(axes[0, 1], m_qt,  'Entrada: M (qtree)')
    im   = axes[0, 2].imshow(mat_map_qt, origin='lower', aspect='auto',
                              cmap=_MAT_CMAP, norm=_MAT_NORM)
    cbar = plt.colorbar(im, ax=axes[0, 2], ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(_MAT_LABELS)
    axes[0, 2].set_title('Materiais (qtree)', fontsize=9)
    _imshow(axes[0, 3], depth_map, 'Depth map (qtree)', cmap='Blues')

    # Linha 1 — GT em qtree
    _imshow(axes[1, 0], bx_tq,  'GT: Bx (qtree)',  vmin=bx_lim[0],  vmax=bx_lim[1])
    _imshow(axes[1, 1], by_tq,  'GT: By (qtree)',  vmin=by_lim[0],  vmax=by_lim[1])
    _imshow(axes[1, 2], mag_tq, 'GT: |B| (qtree)', vmin=mag_lim[0], vmax=mag_lim[1])
    axes[1, 3].axis('off')

    # Linha 2 — FNO assembled (H×W)
    _imshow(axes[2, 0], bx_fno,  'FNO assembled: Bx',  vmin=bx_lim[0],  vmax=bx_lim[1])
    _imshow(axes[2, 1], by_fno,  'FNO assembled: By',  vmin=by_lim[0],  vmax=by_lim[1])
    _imshow(axes[2, 2], mag_fno, 'FNO assembled: |B|', vmin=mag_lim[0], vmax=mag_lim[1])
    _imshow(axes[2, 3], en_fno,
            f'FNO {en_fno_label}\nmédia={fno_m:.1f}%  p95={fno_p95:.1f}%',
            cmap=cmap_nan, vmin=0, vmax=err_vmax)

    # Linha 3 — GNN assembled (qtree real); erro em H×W
    _imshow(axes[3, 0], bx_gnn,  'GNN assembled: Bx (qtree)',  vmin=bx_lim[0],  vmax=bx_lim[1])
    _imshow(axes[3, 1], by_gnn,  'GNN assembled: By (qtree)',  vmin=by_lim[0],  vmax=by_lim[1])
    _imshow(axes[3, 2], mag_gnn, 'GNN assembled: |B| (qtree)', vmin=mag_lim[0], vmax=mag_lim[1])
    _imshow(axes[3, 3], en_gnn,
            f'GNN {en_gnn_label} (H×W vs GT grid)\nmédia={gnn_m:.1f}%  p95={gnn_p95:.1f}%',
            cmap=cmap_nan, vmin=0, vmax=err_vmax)

    plt.tight_layout()
    plt.show()


# [REMOVIDO] phi_deeponet_eval_fn — removida junto com PhiDeepONet (2026-05-27).
# def phi_deeponet_eval_fn(model, d, eval_cfg):
#     ...
# Código original preservado em src/neural_op/archs/phi_deeponet.py (comentado).
