import torch


def _magnitude(t):
    """[B,C,H,W] (grade) ou [S,C] (nós) — magnitude vetorial se C>=2, abs se C==1."""
    if t.shape[1] >= 2:
        return (t[:, 0] ** 2 + t[:, 1] ** 2).sqrt()
    return t[:, 0].abs()


def topk_tail_term(pred, y, k_frac):
    """
    Termo de cauda reutilizável: erro quadrático relativo médio dos k_frac
    elementos (pixels/nós) com maior erro de magnitude, normalizado pelo RMS
    global de |y| (mesma referência relativa — B_ref — usada em
    src/bench/metrics.py) para ficar em escala comparável à MSE.

    pred, y : [B, C, H, W] (grade) ou [S, C] (nós), mesmo shape
    k_frac  : fração (0,1] dos elementos com maior erro a penalizar
    """
    mag_true = _magnitude(y)
    mag_pred = _magnitude(pred)
    b_ref_sq = mag_true.pow(2).mean().clamp(min=1e-8)
    err_rel  = ((mag_pred - mag_true) ** 2 / b_ref_sq).reshape(-1)
    k = max(1, int(k_frac * err_rel.numel()))
    return torch.topk(err_rel, k).values.mean()


def mse_loss(out, y, tail_alpha=0.0, tail_k_frac=0.05):
    base = torch.mean((out - y) ** 2)
    if tail_alpha == 0.0:
        return base
    tail = topk_tail_term(out, y, tail_k_frac)
    return (1.0 - tail_alpha) * base + tail_alpha * tail


def mae_loss(out, y):
    return torch.mean(torch.abs(out - y))


def relative_l2_loss(out, y):
    # Normaliza por amostra (dim 0) — funciona para grade [B,C,H,W] e nós [S,C]
    diff = (out - y).reshape(out.shape[0], -1)
    norm = y.reshape(y.shape[0], -1)
    return (diff.norm(dim=1) / (norm.norm(dim=1) + 1e-8)).mean()


def masked_fno_loss(pred, y, masks):
    """
    Loss mascarada por material — usada por MaskedFNO2d.

    Assinatura estendida: (pred, y, masks) — step_fn é responsável por passar masks.

    pred  : [B, 8, H, W]  — 4 pares (Bx_m, By_m), um por material (ferro=0, ar=1, ima=2, cobre=3)
    y     : [B, 2, H, W]  — alvo (Bx, By)
    masks : [B, 4, H, W]  — bool, partição do domínio sem sobreposição

    Peso de cada material proporcional à sua contagem de pixels no batch (normalizado).
    Bx e By contribuem igualmente (0.5 cada).
    Loss por material: média sobre os pixels do material (evita viés por volume).
    """
    masks_f      = masks.float()
    pixel_counts = masks_f.sum(dim=(0, 2, 3))       # [4] — pixels por material no batch
    total_pixels = pixel_counts.sum().clamp(min=1.0)
    weights      = pixel_counts / total_pixels       # [4], soma 1

    total = pred.new_zeros(())
    for m in range(4):
        mask_m = masks_f[:, m]                                    # [B, H, W]
        n_m    = pixel_counts[m].clamp(min=1.0)
        err_bx = (pred[:, 2 * m    ] - y[:, 0]) ** 2             # [B, H, W]
        err_by = (pred[:, 2 * m + 1] - y[:, 1]) ** 2
        loss_m = ((0.5 * err_bx + 0.5 * err_by) * mask_m).sum() / n_m
        total  = total + weights[m] * loss_m

    return total


def single_material_fno_loss(pred, y, mask_m):
    """
    Loss MSE restrita ao material alvo.

    pred   : [B, 2, H, W]
    y      : [B, 2, H, W]
    mask_m : [B, H, W] bool — pixels do material alvo
    """
    mask_f = mask_m.float().unsqueeze(1)   # [B, 1, H, W]
    n      = mask_f.sum().clamp(min=1.0)
    return ((pred - y) ** 2 * mask_f).sum() / n


# [REMOVIDO] masked_gnn_node_loss — GNN agora produz 2 canais diretamente;
# parcela de nós é MSE simples em masked_fno_gnn_loss.


def masked_fno_gnn_loss(y_hw_8, y_hw, masks, y_nodes_2, node_y, lambda_loss):
    """
    Loss combinada de MaskedFNO_GNN — grade (mascarada) + nós (MSE simples).

    Parcela de grade : masked_fno_loss(y_hw_8, y_hw, masks) — ponderada por material
    Parcela de nós   : mse_loss(y_nodes_2, node_y)          — saída 2-canal da GNN
    Combinação       : lambda_loss * grade + (1 - lambda_loss) * nós

    y_hw_8     : [B, 8, H, W]  — pred FNO 8-canal na grade
    y_hw       : [B, 2, H, W]  — alvo na grade
    masks      : [B, 4, H, W]  — bool, derivado de x_hw[:,0] (Mu_r)
    y_nodes_2  : [S_tot, 2]    — pred GNN 2-canal nos nós (após assemble + delta)
    node_y     : [S_tot, 2]    — alvo nos nós
    lambda_loss: float         — peso da parcela de grade (cfg.lambda_loss)
    """
    loss_grid  = masked_fno_loss(y_hw_8, y_hw, masks)
    loss_nodes = mse_loss(y_nodes_2, node_y)
    return lambda_loss * loss_grid + (1.0 - lambda_loss) * loss_nodes


def _p1_grad_coeffs(v_coords):
    """
    b,c (shape function P1) + área*2 (assinada) dos 3 vértices de cada
    elemento -- mesma fórmula fechada de src/data_gen/parsers/ans_parsing.py::
    _element_b_from_A, em torch/batelado. v_coords: [M,3,2] (coordenadas dos
    3 vértices, ordem v0,v1,v2). Retorna b,c [M,3], area [M].
    """
    x0, y0 = v_coords[:, 0, 0], v_coords[:, 0, 1]
    x1, y1 = v_coords[:, 1, 0], v_coords[:, 1, 1]
    x2, y2 = v_coords[:, 2, 0], v_coords[:, 2, 1]
    area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    b = torch.stack([y1 - y2, y2 - y0, y0 - y1], dim=1)
    c = torch.stack([x2 - x1, x0 - x2, x1 - x0], dim=1)
    return b, c, area2.abs() / 2.0


def node_x_to_xy_mm(node_x, r_in_mm, r_ext_mm, ang_1_deg, ang_2_deg):
    """
    Converte node_x=[r_base,c_base] (coordenadas polares normalizadas, ver
    src/data_gen/parsers/femm_mesh_v2.py) pra (x,y) reais em mm -- exato, não
    aproximado, porque a janela de amostragem (r_in/r_ext/ang_1/ang_2) é
    CONSTANTE pra todo o dataset (ver DivBLossCfg, src/configs/loss.py, pra
    a confirmação/justificativa completa). r = r_in + r_base·(r_ext-r_in),
    θ = ang_1 + c_base·(ang_2-ang_1), x=r·cosθ, y=r·senθ.
    """
    r_base, c_base = node_x[:, 0], node_x[:, 1]
    r = r_in_mm + r_base * (r_ext_mm - r_in_mm)
    ang_1_rad = torch.deg2rad(node_x.new_tensor(ang_1_deg))
    ang_2_rad = torch.deg2rad(node_x.new_tensor(ang_2_deg))
    theta = ang_1_rad + c_base * (ang_2_rad - ang_1_rad)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)


def graph_weak_divergence(node_field, node_x, cross_edge_index,
                           r_in_mm, r_ext_mm, ang_1_deg, ang_2_deg):
    """
    Divergente nodal discreto (fraco/Galerkin, massa lumped) de um campo
    vetorial 2D definido nos vértices de um grafo de malha (mode=
    'femm_mesh_v2') -- versão torch/batelada/diferenciável de
    tests/proto_div_b_check.py::weak_nodal_divergence. Derivação completa
    (forma fraca, integração por partes, phi_i linear por elemento) na
    docstring daquele arquivo.

    Roda sobre um BATCH inteiro (vários grafos desconexos concatenados,
    mesma convenção de índices globais de src/neural_op/dataloaders/
    grid_loader.py::femm_mesh_v2_collate) -- scatter/index_add não mistura
    contribuições entre amostras porque os índices de nó/elemento de
    amostras diferentes nunca se sobrepõem.

    Geometria em (x,y) mm reais (node_x_to_xy_mm), não em r_base/c_base cru
    -- ver DivBLossCfg (src/configs/loss.py) pro porquê isso é exato (janela
    de amostragem fixa) e não uma aproximação.

    node_field       : [S,2]  campo (Bx,By ou equivalente) por nó
    node_x           : [S,2]  r_base,c_base por nó
    cross_edge_index : [2,C]  linha0=índice do elemento, linha1=índice do
                       vértice -- 3 entradas consecutivas por elemento
                       (v0,v1,v2), mesma convenção de
                       src/data_gen/parsers/femm_mesh_v2.py
    r_in_mm/r_ext_mm/ang_1_deg/ang_2_deg : janela de amostragem (constantes
                       do dataset, ver DivBLossCfg)
    Retorna div [S] em T/m -- unidade real, sem distorção de métrica (nós
    sem nenhum elemento incidente ficam 0/0 -> 0, via clamp no denominador).
    """
    n_nodes = node_field.shape[0]
    # metros, não mm -- mesma pegadinha (erro de ~1000x se esquecido) já
    # documentada em ans_parsing.py::_element_b_from_A/CLAUDE.md "Extração
    # de dados direto do arquivo .ans".
    xy_m = node_x_to_xy_mm(node_x, r_in_mm, r_ext_mm, ang_1_deg, ang_2_deg) * 1e-3
    vtx = cross_edge_index[1].view(-1, 3)                      # [M,3]
    coords = xy_m[vtx]                                         # [M,3,2]
    b, c, area = _p1_grad_coeffs(coords)                       # [M,3],[M,3],[M]

    field = node_field[vtx]                                    # [M,3,2]
    Bx_e = field[..., 0].mean(dim=1)                            # [M] média simples dos 3 vértices
    By_e = field[..., 1].mean(dim=1)

    contrib = (Bx_e[:, None] * b + By_e[:, None] * c) / 2.0    # [M,3]

    num       = node_field.new_zeros(n_nodes)
    dual_area = node_field.new_zeros(n_nodes)
    for corner in range(3):
        num.index_add_(0, vtx[:, corner], contrib[:, corner])
        dual_area.index_add_(0, vtx[:, corner], area / 3.0)

    return -num / dual_area.clamp(min=1e-12)


def graph_div_b_loss(y_nodes, node_y, node_x, cross_edge_index, base_loss='mae', lambda_div=0.1,
                      r_in_mm=28.5, r_ext_mm=46.5, ang_1_deg=0.0, ang_2_deg=120.0):
    """
    Loss de ajuste (`base_loss`, chave simples em LOSS_REGISTRY) + penalidade
    pelo divergente nodal (graph_weak_divergence, em (x,y) mm reais) da
    PREDIÇÃO -- ver DivBLossCfg (src/configs/loss.py) pro racional completo e
    a única aproximação restante (normalização z-score de y_nodes/node_y
    durante o treino, não T/m calibrado).

    Assinatura estendida -- requer step_fn compatível
    (src/neural_op/archs/femm_mesh_v2_gnn.py::make_fno_bipartite_gnn_step),
    que só aplica esta loss na parcela de NÓS; a parcela de grade usa
    `base_loss` puro (ver docstring de DivBLossCfg).
    """
    fit = LOSS_REGISTRY[base_loss](y_nodes, node_y)
    div = graph_weak_divergence(y_nodes, node_x, cross_edge_index,
                                 r_in_mm, r_ext_mm, ang_1_deg, ang_2_deg)
    return fit + lambda_div * (div ** 2).mean()


class BaseLoss:
    """
    Wrapper extensível em volta de uma função de loss pura (fn) -- __call__
    só repassa pra fn (mesma assinatura de sempre, nenhuma mudança em nenhum
    step_fn de arch) e acrescenta log_epoch como PROPRIEDADE da loss: quem
    decide o que aparece no console a cada época é a própria loss, não o
    loop de treino (fit(), src/neural_op/training_utils.py).

    Por ora toda entrada de LOSS_REGISTRY usa o log_epoch padrão abaixo
    (mesmo texto que já era hardcoded em fit()). Uma loss que precise
    destacar termos próprios (ex: erro de ajuste/"data" vs penalidade
    física -- caso de graph_div_b_loss, que já calcula `fit` e `div`
    separadamente) sobrescreve log_epoch numa subclasse; nada mais no
    projeto precisa mudar pra isso (scripts/train.py já extrai
    `loss_obj.log_epoch` do objeto, antes de qualquer functools.partial).
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def log_epoch(self, ep, train_loss, test_loss, train_time_s, eval_time_s, samples_per_s):
        print(f"epoch {ep:>4d}  train {train_loss:.4e}  test {test_loss:.4e}"
              f"  [{train_time_s:.1f}s + {eval_time_s:.1f}s eval]  {samples_per_s:.0f} samp/s")


class DivBLoss(BaseLoss):
    """
    BaseLoss especializado para graph_div_b_loss -- calcula fit (erro de
    ajuste) e div (penalidade de divergente) separadamente a cada __call__
    (mesma fórmula de graph_div_b_loss, que fica intacta pra quem mais usar)
    e acumula as somas em tensores no device, separadas por fase
    treino/teste -- fase detectada via torch.is_grad_enabled() (True dentro
    de train_epoch, False dentro do torch.no_grad() de eval_epoch), sem
    precisar tocar train_epoch/eval_epoch/fit() (src/neural_op/
    training_utils.py) nem a assinatura de nenhum step_fn.

    log_epoch imprime a média de fit/div de cada fase, além do loss
    combinado de sempre, e zera os acumuladores a cada chamada (uma por
    época). Acumula via .detach() na GPU -- só sincroniza (.item()) uma vez
    por época, no print, mesmo padrão que o loss combinado já usa em
    train_epoch/eval_epoch.
    """

    def __init__(self, fn):
        super().__init__(fn)
        self._sums = None   # inicializado no primeiro __call__ (não sabe o device antes)

    def _zero_sums(self, device):
        self._sums = {
            'train': [torch.zeros((), device=device), torch.zeros((), device=device), 0],
            'test':  [torch.zeros((), device=device), torch.zeros((), device=device), 0],
        }

    def __call__(self, y_nodes, node_y, node_x, cross_edge_index, base_loss='mae', lambda_div=0.1,
                 r_in_mm=28.5, r_ext_mm=46.5, ang_1_deg=0.0, ang_2_deg=120.0):
        fit = LOSS_REGISTRY[base_loss](y_nodes, node_y)
        div = graph_weak_divergence(y_nodes, node_x, cross_edge_index,
                                     r_in_mm, r_ext_mm, ang_1_deg, ang_2_deg)
        div_term = (div ** 2).mean()

        if self._sums is None:
            self._zero_sums(fit.device)
        phase = 'train' if torch.is_grad_enabled() else 'test'
        s = self._sums[phase]
        with torch.no_grad():
            s[0] += fit.detach()
            s[1] += div_term.detach()
        s[2] += 1

        return fit + lambda_div * div_term

    def log_epoch(self, ep, train_loss, test_loss, train_time_s, eval_time_s, samples_per_s):
        if self._sums is None:
            self._zero_sums('cpu')
        tr_fit, tr_div, tr_n = self._sums['train']
        te_fit, te_div, te_n = self._sums['test']
        tr_fit = (tr_fit / tr_n).item() if tr_n else 0.0
        tr_div = (tr_div / tr_n).item() if tr_n else 0.0
        te_fit = (te_fit / te_n).item() if te_n else 0.0
        te_div = (te_div / te_n).item() if te_n else 0.0
        print(f"epoch {ep:>4d}  train {train_loss:.4e} (data {tr_fit:.4e}  div {tr_div:.4e})"
              f"  test {test_loss:.4e} (data {te_fit:.4e}  div {te_div:.4e})"
              f"  [{train_time_s:.1f}s + {eval_time_s:.1f}s eval]  {samples_per_s:.0f} samp/s")
        self._zero_sums(self._sums['train'][0].device)


# [REMOVIDO 2026-08-18] LOSS_REGISTRY com funções soltas — cada entrada
# agora é um BaseLoss(fn) (ver classe acima), pra dar à loss uma propriedade
# de print de época (log_epoch) sem mudar nenhuma chamada existente
# (LOSS_REGISTRY[nome](out, y, ...) continua idêntico — instância é
# chamável igual função).
# LOSS_REGISTRY: dict = {
#     'mse':                     mse_loss,
#     'mae':                     mae_loss,
#     'relative_l2':             relative_l2_loss,
#     'masked_fno_loss':         masked_fno_loss,
#     'single_material_fno_loss': single_material_fno_loss,
#     'masked_fno_gnn_loss':     masked_fno_gnn_loss,
#     'graph_div_b_loss':        graph_div_b_loss,
# }

LOSS_REGISTRY: dict = {
    'mse':                      BaseLoss(mse_loss),
    'mae':                      BaseLoss(mae_loss),
    'relative_l2':              BaseLoss(relative_l2_loss),
    # assinatura estendida (pred, y, masks) — requer step_fn compatível (MaskedFNO2d)
    'masked_fno_loss':          BaseLoss(masked_fno_loss),
    # assinatura estendida (pred, y, mask_m) — requer step_fn compatível (FNO2d_SingleMat)
    'single_material_fno_loss': BaseLoss(single_material_fno_loss),
    # assinatura estendida (y_hw_8, y_hw, masks, y_nodes_2, node_y, lambda_loss)
    'masked_fno_gnn_loss':      BaseLoss(masked_fno_gnn_loss),
    # assinatura estendida (y_nodes, node_y, node_x, cross_edge_index, base_loss, lambda_div)
    # — exclusiva de grafos (FNO_BipartiteGNN); ver graph_div_b_loss acima e
    # DivBLossCfg (src/configs/loss.py). DivBLoss (não BaseLoss simples) —
    # log_epoch próprio, imprime fit/div separados (ver classe acima).
    'graph_div_b_loss':         DivBLoss(graph_div_b_loss),
}
