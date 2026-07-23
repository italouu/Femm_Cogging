"""
femm_mesh.py
------------
Geração de dados via a malha real do FEMM (triangulação do solver) como
grafo, em vez do quadtree Shapely (ver src/data_gen/data_utils.py). Usa
BLDC_FEMM_Model_Sym120_Annular (setor 120° + truncamento radial).

Promovido de tests/proto_femm_mesh_gendata.py + tests/proto_export_fem_mesh.py
(validado 2026-07-22, promovido 2026-07-23), com duas mudanças em relação
aos protótipos:

  - material_id/mu_r/M passam a vir 100% do FEMM (sem Shapely) — ver
    _block_magnet_polarity/_node_magnet_polarity. O protótipo original usava
    point-query Shapely (_magnet_sign_at_points) pra resolver a polaridade
    do ímã, porque mu sozinho (via [BlockProps]) não distingue N35p de
    N35n (mesma mu_r=1.05). Descoberta desta sessão: cada polo gera uma
    entrada PRÓPRIA em [BlockProps] (mesmo nome "N35", duplicada — o FEMM
    cria uma entrada nova a cada mi_getmaterial(), mesmo pro mesmo nome), na
    MESMA ordem em que os polos são desenhados — confirmado empiricamente
    contando os elementos de cada polo por `label`. Combinado com a
    paridade já conhecida do código (direction = 0 se pole%2==0 senão 180,
    em BLDC_FEMM_Model_Sym120.draw_motor), dá pra recuperar a polaridade só
    da ordem dos blocos, sem nenhuma consulta de geometria.

  - edge_attr NÃO tem mais shared_length (sem análogo real numa malha
    triangular — decisão 2026-07-23, ver histórico) e ganha arestas de
    "wrap" no contorno periódico theta=0/120 (_wrap_edge_pairs). Confirmado
    empiricamente que o FEMM garante a MESMA discretização radial nos dois
    cortes (mesmos r, diff ~1e-14) e resolve A como o mesmo grau de
    liberdade nos dois lados (diff |A| entre pares casados por r ~1e-11,
    ruído de ponto flutuante) — o pareamento por raio ordenado é exato.

x_hw_grid/y_hw_grid/a_hw_grid deixaram de usar Shapely e point-query live
(mo_getb/mo_geta por pixel) — 2026-07-23, a pedido do usuário: agora derivam
100% da malha real já extraída do .ans (trifinder + interpolação
baricêntrica, ver _assign_mesh_to_grid abaixo). Único custo de COM que resta
é o loop de B por nó (necessário pro grafo — B não dá pra reconstruir de A
com precisão).
"""

from pathlib import Path

import numpy as np
import femm
import matplotlib.tri as mtri
# [REMOVIDO] from shapely.geometry import Point -- usado só pra grade Mu_r/M via
# point-query Shapely, agora derivada da malha (ver _assign_mesh_to_grid, 2026-07-23)

from src.data_gen.motor_model import (
    BLDC_Process, BLDC_FEMM_Model_Sym120_Annular,
    # [REMOVIDO] BLDC_Shapely_Model -- só usado pra grade Mu_r/M via point-query
    # Shapely; grade agora deriva 100% da malha (ver _assign_mesh_to_grid)
)

_N_MATERIALS = len(set(BLDC_Process.MATERIAL_ID.values()))  # 4: ferro,ar,ima,cobre
_MAGNET_ID = BLDC_Process.MATERIAL_ID['N35p']                # == MATERIAL_ID['N35n']

_MU_BY_ID = np.zeros(_N_MATERIALS, dtype=np.float32)
for _name, _mid in BLDC_Process.MATERIAL_ID.items():
    _MU_BY_ID[_mid] = BLDC_Process.PERMEABILITY[_name]


# ---------------------------------------------------------------------------
# leitura do .ans (só arquivo, sem chamada COM)
# ---------------------------------------------------------------------------

def _parse_solution(ans_path):
    with open(ans_path) as f:
        lines = f.readlines()

    def section(name):
        return next(i for i, l in enumerate(lines) if l.strip().startswith(name))

    idx = section('[Solution]')
    n_nodes = int(lines[idx + 1].strip())
    nodes = np.loadtxt(lines[idx + 2: idx + 2 + n_nodes])[:, :3]  # x, y, A
    idx2 = idx + 2 + n_nodes
    n_elems = int(lines[idx2].strip())
    elems = np.loadtxt(lines[idx2 + 1: idx2 + 1 + n_elems])[:, :4].astype(np.int64)  # n1,n2,n3,label
    return lines, nodes, elems


def _parse_block_materials(lines):
    """Por índice de BlockProps (0-based, ordem do arquivo): material_id e
    mu_r CONSTANTES (convenção antiga PERMEABILITY), classificados por
    limiar de mu.

    Limiar de cobre (0.999995) é MAIS APERTADO que o de _material_ids_from_mu
    (mu<0.9995) usado no resto do projeto — aquele opera sobre o mu_r já
    canonicalizado (só 4 valores possíveis: 5000/1.05/0.999/1.0), enquanto
    aqui é o Mu_x BRUTO do bloco no FEMM, e o material "copper" da
    biblioteca do FEMM tem Mu_x=0.999991 (~1.0, bem mais perto de vácuo do
    que o 0.999 aproximado do projeto) — confirmado lendo o .fem gerado.
    Com o limiar antigo (0.9995) todo nó de cobre caía silenciosamente em
    vácuo/ar; achado e corrigido nesta sessão (2026-07-23)."""
    idx = next(i for i, l in enumerate(lines) if l.strip().startswith('[BlockProps]'))
    nblocks = int(lines[idx].split('=')[1].strip())

    material_id = np.empty(nblocks, dtype=np.int64)
    mu_const = np.empty(nblocks, dtype=np.float32)

    b = 0
    i = idx + 1
    mu_x = has_bh = None
    while b < nblocks:
        line = lines[i]
        if '<Mu_x>' in line:
            mu_x = float(line.split('=')[1].strip())
        elif '<BHPoints>' in line:
            has_bh = int(line.split('=')[1].strip()) > 0
        elif '<EndBlock>' in line:
            if has_bh or mu_x > 10:
                canon = 'iron_1008'
            elif 1.01 < mu_x < 10:
                canon = 'N35p'
            elif mu_x < 0.999995:
                canon = 'copper'
            else:
                canon = 'vacuum'
            material_id[b] = BLDC_Process.MATERIAL_ID[canon]
            mu_const[b] = BLDC_Process.PERMEABILITY[canon]
            b += 1
            mu_x = has_bh = None
        i += 1

    return material_id, mu_const


def _block_magnet_polarity(block_material_id, n_poles_sector):
    """Polaridade (MAGNETIZATION, +-1) de cada bloco de ímã, derivada só da
    ORDEM DE CRIAÇÃO dos blocos no FEMM — sem Shapely.

    Depende de uma invariante estrutural do desenho
    (BLDC_FEMM_Model_Sym120.draw_motor): os n_poles_sector polos são
    criados em um laço sequencial, cada _create_sec chamando
    mi_getmaterial(material_mag) uma vez — o que gera uma entrada NOVA em
    [BlockProps] por polo, na MESMA ordem em que os polos são desenhados
    (índice crescente = polo 0, 1, 2, ...). Combinado com a paridade
    conhecida do código (direction = 0 se pole%2==0 senão 180), dá pra
    recuperar a polaridade de cada bloco só pela posição dele na lista
    ordenada de blocos classificados como ímã.

    Se esse laço de criação mudar (ordem, número de polos, chamadas
    intercaladas de outro material 'ímã'), essa correspondência quebra —
    por isso o assert de contagem abaixo falha alto em vez de silenciosamente
    dar polaridade errada.
    """
    magnet_block_idx = np.where(block_material_id == _MAGNET_ID)[0]
    if len(magnet_block_idx) != n_poles_sector:
        raise ValueError(
            f"esperava {n_poles_sector} blocos de ímã (1 por polo do setor), "
            f"achei {len(magnet_block_idx)} — invariante de ordem de criação "
            f"quebrada (ver docstring de _block_magnet_polarity), não dá pra "
            f"recuperar polaridade só do FEMM.")

    block_M = np.zeros(len(block_material_id), dtype=np.float32)
    for pole_i, block_idx in enumerate(magnet_block_idx):
        name = 'N35p' if pole_i % 2 == 0 else 'N35n'
        block_M[block_idx] = BLDC_Process.MAGNETIZATION[name]
    return block_M


def _build_edges(elems):
    tri = elems[:, :3]
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


# ---------------------------------------------------------------------------
# geometria da malha (numpy puro, sem FEMM)
# ---------------------------------------------------------------------------

def _element_areas(nodes: np.ndarray, elems: np.ndarray) -> np.ndarray:
    x1, y1 = nodes[elems[:, 0], 0], nodes[elems[:, 0], 1]
    x2, y2 = nodes[elems[:, 1], 0], nodes[elems[:, 1], 1]
    x3, y3 = nodes[elems[:, 2], 0], nodes[elems[:, 2], 1]
    return 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def _node_material_stats(nodes: np.ndarray, elems: np.ndarray, area: np.ndarray,
                          elem_material_id: np.ndarray):
    """Vota material dominante por nó (area-weighted, mesmo princípio do
    frac_dom do quadtree) e computa a área lumped do nó (soma de 1/3 da
    área dos triângulos incidentes — massa lumped padrão de FEM linear).

    Retorna: node_material_id [n_nodes], frac_dom [n_nodes], node_dual_area [n_nodes] (mm^2)
    """
    n_nodes = nodes.shape[0]

    area_per_mat = np.zeros((n_nodes, _N_MATERIALS), dtype=np.float64)
    for corner in range(3):
        np.add.at(area_per_mat, (elems[:, corner], elem_material_id), area)
    node_material_id = area_per_mat.argmax(axis=1)
    total = area_per_mat.sum(axis=1)
    frac_dom = np.divide(area_per_mat.max(axis=1), total,
                          out=np.ones(n_nodes, dtype=np.float64), where=total > 0)

    node_dual_area = np.zeros(n_nodes, dtype=np.float64)
    for corner in range(3):
        np.add.at(node_dual_area, elems[:, corner], area / 3.0)

    return node_material_id, frac_dom.astype(np.float32), node_dual_area.astype(np.float32)


def _node_magnet_polarity(nodes: np.ndarray, elems: np.ndarray, area: np.ndarray,
                           elem_material_id: np.ndarray, elem_M: np.ndarray) -> np.ndarray:
    """M por nó, restrito a nós que tocam ao menos um elemento classificado
    como ímã: votação por área (mesmo princípio de frac_dom) SÓ entre os
    elementos incidentes já classificados como ímã, escolhendo o sinal
    (+-MAGNETIZATION) de maior área somada. Nós que não tocam nenhum ímã
    ficam com M=0 (ferro/ar/cobre já têm MAGNETIZATION=0 em toda a área).

    100% dados do FEMM (elem_M vem de _block_magnet_polarity) — sem
    Shapely. A ambiguidade só existiria num nó exatamente na borda entre
    dois ímãs de polaridade diferente — como cada polo é fisicamente
    separado por ar, isso não acontece na prática; o voto cobre o caso
    geral mesmo assim.
    """
    n_nodes = nodes.shape[0]
    is_magnet = elem_material_id == _MAGNET_ID

    area_pos = np.zeros(n_nodes, dtype=np.float64)
    area_neg = np.zeros(n_nodes, dtype=np.float64)
    for corner in range(3):
        idx = elems[:, corner]
        m_pos = is_magnet & (elem_M > 0)
        np.add.at(area_pos, idx[m_pos], area[m_pos])
        m_neg = is_magnet & (elem_M < 0)
        np.add.at(area_neg, idx[m_neg], area[m_neg])

    touches_magnet = (area_pos + area_neg) > 0
    sign = np.where(area_pos >= area_neg, 1.0, -1.0)
    node_M = np.where(touches_magnet, sign * BLDC_Process.MAGNETIZATION['N35p'], 0.0)
    return node_M.astype(np.float32)


def _wrap_edge_pairs(nodes: np.ndarray, ang_1_deg: float, ang_2_deg: float, tol_deg: float = 1e-3):
    """Pares (i,j) de nós identificados pelo contorno periódico
    theta=ang_1 <-> theta=ang_2 do FEMM.

    Confirmado empiricamente (motor real, seed=42): o FEMM garante a MESMA
    discretização radial nos dois cortes (mesmos r, diff ~1e-14) e resolve
    A como o mesmo grau de liberdade nos dois lados (diff |A| entre pares
    casados por r ~1e-11, ruído de ponto flutuante) — o pareamento por raio
    ordenado é exato, sem heurística de tolerância espacial.
    """
    x, y = nodes[:, 0], nodes[:, 1]
    th_deg = np.degrees(np.arctan2(y, x)) % 360

    idx_1 = np.where(np.abs(th_deg - (ang_1_deg % 360)) < tol_deg)[0]
    idx_2 = np.where(np.abs(th_deg - (ang_2_deg % 360)) < tol_deg)[0]

    if len(idx_1) != len(idx_2):
        raise ValueError(
            f"contorno periódico com contagem de nós diferente: "
            f"theta={ang_1_deg} tem {len(idx_1)} nós, theta={ang_2_deg} tem {len(idx_2)}")

    r = np.hypot(x, y)
    idx_1 = idx_1[np.argsort(r[idx_1])]
    idx_2 = idx_2[np.argsort(r[idx_2])]
    return idx_1, idx_2


def _build_bidirectional_edge_attrs(nodes: np.ndarray, edges_undirected: np.ndarray,
                                     wrap_idx_1: np.ndarray, wrap_idx_2: np.ndarray,
                                     r_base: np.ndarray, c_base: np.ndarray,
                                     node_mu: np.ndarray, H: int, W: int):
    """edge_index [2,E] + edge_attr [E,4] bidirecional, colunas
    [delta_r, delta_c, center_dist, delta_mu] — mesma convenção de produção
    (delta_mu = origem-destino, i-j), SEM shared_length (removido — sem
    análogo direto numa malha triangular, decisão 2026-07-23).

    Inclui as arestas de "wrap" do contorno periódico (wrap_idx_1<->
    wrap_idx_2, ver _wrap_edge_pairs) com delta_r=delta_c=center_dist=0 —
    não são um salto espacial real, são identificação do mesmo grau de
    liberdade nos dois lados do corte periódico.
    """
    i = np.concatenate([edges_undirected[:, 0], wrap_idx_1])
    j = np.concatenate([edges_undirected[:, 1], wrap_idx_2])
    n_wrap = len(wrap_idx_1)

    dx = nodes[j, 0] - nodes[i, 0]
    dy = nodes[j, 1] - nodes[i, 1]
    center_dist = np.hypot(dx, dy).astype(np.float32)
    delta_r = ((r_base[j] - r_base[i]) * H).astype(np.float32)
    delta_c = ((c_base[j] - c_base[i]) * W).astype(np.float32)
    if n_wrap:
        center_dist[-n_wrap:] = 0.0
        delta_r[-n_wrap:] = 0.0
        delta_c[-n_wrap:] = 0.0
    delta_mu = (node_mu[i] - node_mu[j]).astype(np.float32)

    attr_fwd = np.stack([delta_r, delta_c, center_dist, delta_mu], axis=1)
    attr_bwd = np.stack([-delta_r, -delta_c, center_dist, -delta_mu], axis=1)

    edge_index = np.stack([np.concatenate([i, j]), np.concatenate([j, i])], axis=0)
    edge_attr = np.concatenate([attr_fwd, attr_bwd], axis=0)
    return edge_index.astype(np.int64), edge_attr.astype(np.float32)


# ---------------------------------------------------------------------------
# grade H×W a partir da malha (trifinder + interpolação baricêntrica)
# ---------------------------------------------------------------------------

def _assign_mesh_to_grid(nodes: np.ndarray, elems: np.ndarray,
                          elem_mu: np.ndarray, elem_M: np.ndarray,
                          node_mu: np.ndarray, node_M: np.ndarray,
                          node_bx: np.ndarray, node_by: np.ndarray, node_A: np.ndarray,
                          r_in: float, r_ext: float, ang_1_rad: float, ang_2_rad: float,
                          n_r: int, n_a: int):
    """Projeta os campos da malha real na grade H×W regular -- SEM Shapely,
    SEM chamada COM adicional ao FEMM por pixel (só os dados já extraídos do
    .ans + node_bx/node_by, que vêm do loop mo_getb por-nó de qualquer forma,
    necessário pro grafo). Promovido de
    tests/proto_femm_mesh_plot.py::_assign_mesh_to_grid (validado nessa sessão
    contra o point-query antigo; ver docstring lá para a justificativa
    completa dos dois regimes de atribuição usados abaixo). 2026-07-23.

    Regimes de atribuição (mesma lógica do protótipo):
      - Mu_r/M: PIECEWISE-CONSTANT por elemento -- localiza o triângulo que
        contém o centro do pixel via trifinder e copia o valor CONSTANTE
        desse elemento. Preserva fronteiras nítidas (sem blending) -- mesmo
        espírito do point-query antigo, só que "point-query" agora é
        "localizar o triângulo real" em vez de reavaliar geometria Shapely.
      - Bx/By/A: interpolação BARICÊNTRICA (LinearTriInterpolator) a partir
        dos 3 valores nodais do elemento -- é literalmente a mesma avaliação
        que o FEM linear faz internamente num ponto arbitrário dentro de um
        elemento (funções de forma), então é o análogo correto de "consultar
        o FEMM naquele ponto exato".

    Pixels fora da triangulação (trifinder retorna -1 -- esperado raro, só
    arredondamento de ponto flutuante na borda do setor, já que a malha foi
    gerada na MESMA janela r_in/r_ext/ang_1/ang_2 da grade) caem no valor do
    NÓ MAIS PRÓXIMO (fallback via cKDTree, calculado só se necessário).
    """
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=elems[:, :3])
    trifinder = tri.get_trifinder()

    dr = (r_ext - r_in) / n_r
    da = (ang_2_rad - ang_1_rad) / n_a
    r_vals = r_in + (np.arange(n_r) + 0.5) * dr
    a_vals = ang_1_rad + (np.arange(n_a) + 0.5) * da
    Rg, Ag = np.meshgrid(r_vals, a_vals, indexing='ij')
    Xg, Yg = Rg * np.cos(Ag), Rg * np.sin(Ag)

    elem_idx = trifinder(Xg.ravel(), Yg.ravel())
    valid = elem_idx >= 0
    n_invalid = int((~valid).sum())
    if n_invalid:
        from scipy.spatial import cKDTree
        nearest = cKDTree(nodes[:, :2]).query(
            np.stack([Xg.ravel()[~valid], Yg.ravel()[~valid]], axis=1))[1]

    def _assign_const(elem_field, node_fallback):
        out = np.empty(n_r * n_a, dtype=np.float32)
        out[valid] = elem_field[elem_idx[valid]]
        if n_invalid:
            out[~valid] = node_fallback[nearest]
        return out.reshape(n_r, n_a)

    def _assign_interp(node_field):
        interp = mtri.LinearTriInterpolator(tri, node_field)
        out = np.ma.filled(interp(Xg, Yg).astype(np.float32), 0.0).reshape(-1)
        if n_invalid:
            out[~valid] = node_field[nearest].astype(np.float32)
        return out.reshape(n_r, n_a)

    Mu_hw = _assign_const(elem_mu, node_mu)
    M_hw  = _assign_const(elem_M, node_M)
    Bx_hw = _assign_interp(node_bx)
    By_hw = _assign_interp(node_by)
    A_hw  = _assign_interp(node_A)
    return Mu_hw, M_hw, Bx_hw, By_hw, A_hw


# ---------------------------------------------------------------------------
# pipeline completo por amostra
# ---------------------------------------------------------------------------

def generate_mesh_sample(motor_params: dict, tmp_dir: Path, out_path: Path,
                          n_r: int, n_a: int, ang_1: float = 0, ang_2: float = 120) -> dict:
    """Gera uma amostra completa (grafo da malha real + grade H×W) e grava
    em out_path como .npz (escrita atômica: .tmp.npz -> rename, igual
    process_and_save_sample do pipeline qtree — out_path deve terminar em
    .npz).

    Layout de saída (staging — ver scripts/generate_data_femm_mesh.py):
        node_x       [S,9]  float32  material_id,mu_r,M,node_dual_area,r_base,c_base,frac_dom,0,0
        node_y       [S,2]  float32  Bx,By
        node_A       [S]    float32  A (potencial vetor, valor nodal exato do .ans)
        edge_index   [2,E]  int64    bidirecional (malha + wrap periódico)
        edge_attr    [E,4]  float32  delta_r,delta_c,center_dist,delta_mu
        x_hw_grid    [2,H,W] float32 Mu_r,M (derivado da malha -- trifinder, valor
                                             constante do elemento; ver _assign_mesh_to_grid)
        y_hw_grid    [2,H,W] float32 Bx,By  (derivado da malha -- interp. baricêntrica
                                             a partir dos nós; ver _assign_mesh_to_grid)
        a_hw_grid    [H,W]   float32 A      (idem Bx/By, interp. baricêntrica)
        L            [1]     int64   número de nós desta amostra
        dim_H, dim_W [ ]      int64   n_r, n_a (dim reconstruído como tupla no agrupamento)
        r_in_mm, r_ext_mm, ang_1_deg, ang_2_deg — metadados de geometria (descartados no
            agrupamento final, ver scripts/build_data_chunks_femm_mesh.py)
    """
    tmp_dir = Path(tmp_dir)
    out_path = Path(out_path)
    fem_file = str(tmp_dir / "model.fem")
    ans_file = str(tmp_dir / "model.ans")

    ang_1_rad, ang_2_rad = np.deg2rad(ang_1), np.deg2rad(ang_2)

    model = BLDC_FEMM_Model_Sym120_Annular(motor_params=motor_params, phase=0)
    r_in = model.inner_diameter / 2
    r_ext = model.outer_diameter / 2

    femm.openfemm('bHide')
    femm.main_resize(1000, 1000)
    femm.newdocument(0)
    femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, 0, 200)
    model.draw_motor()
    femm.mi_zoomnatural()
    femm.mi_saveas(fem_file)
    femm.mi_createmesh()
    femm.mi_analyze()
    femm.mi_loadsolution()

    # --- parse do .ans (só arquivo, sem COM) ---
    lines, nodes, elems = _parse_solution(ans_file)
    block_material_id, block_mu = _parse_block_materials(lines)
    block_M = _block_magnet_polarity(block_material_id, model.N_POLES_SECTOR)

    elem_material_id = block_material_id[elems[:, 3]]
    elem_mu = block_mu[elems[:, 3]]
    elem_M = block_M[elems[:, 3]]
    edges_undirected = _build_edges(elems)

    r_node = np.hypot(nodes[:, 0], nodes[:, 1])
    th_node = np.arctan2(nodes[:, 1], nodes[:, 0])
    r_base = (r_node - r_in) / (r_ext - r_in)
    c_base = (th_node - ang_1_rad) / (ang_2_rad - ang_1_rad)

    # --- estatísticas por nó (votação de material, frac_dom, área lumped, polaridade) ---
    area = _element_areas(nodes, elems)
    node_material_id, frac_dom, node_dual_area = _node_material_stats(
        nodes, elems, area, elem_material_id)
    node_mu = _MU_BY_ID[node_material_id]
    node_M = _node_magnet_polarity(nodes, elems, area, elem_material_id, elem_M)

    # --- arestas bidirecionais (malha + wrap periódico) ---
    wrap_idx_1, wrap_idx_2 = _wrap_edge_pairs(nodes, ang_1, ang_2)
    edge_index, edge_attr = _build_bidirectional_edge_attrs(
        nodes, edges_undirected, wrap_idx_1, wrap_idx_2, r_base, c_base, node_mu, n_r, n_a)

    # --- B por nó (grafo) -- único custo residual de COM por-nó (B não dá
    # pra reconstruir de A com precisão -- ver CLAUDE.md/histórico) ---
    n_nodes = nodes.shape[0]
    bx = np.empty(n_nodes, dtype=np.float32)
    by = np.empty(n_nodes, dtype=np.float32)
    for k in range(n_nodes):
        b = femm.mo_getb(nodes[k, 0], nodes[k, 1])
        bx[k], by[k] = b[0], b[1]

    femm.closefemm()

    # [REMOVIDO] grade H×W via point-query live (mo_getb/mo_geta por pixel,
    # ~n_r*n_a chamadas COM) + Mu_r/M via Shapely (BLDC_Shapely_Model +
    # Point.covers por pixel) -- substituído por _assign_mesh_to_grid, que
    # deriva TUDO só da malha já extraída do .ans (sem Shapely, sem COM
    # adicional além do loop de B por nó acima). Decisão do usuário
    # 2026-07-23: usar apenas dados obtidos através da malha do FEMM.
    #
    # bx_hw, by_hw, a_hw = [], [], []
    # for x, y, r, th in model._iter_polar_points(r_in, r_ext, ang_1_rad, ang_2_rad, n_r, n_a):
    #     b = femm.mo_getb(x, y)
    #     bx_hw.append(b[0]); by_hw.append(b[1])
    #     a_hw.append(femm.mo_geta(x, y))
    # Bx_hw = np.asarray(bx_hw, dtype=np.float32).reshape(n_r, n_a)
    # By_hw = np.asarray(by_hw, dtype=np.float32).reshape(n_r, n_a)
    # A_hw = np.asarray(a_hw, dtype=np.float32).reshape(n_r, n_a)
    #
    # model_shapely = BLDC_Shapely_Model(motor_params=motor_params, phase=0)
    # model_shapely.draw_motor()
    # permeability, magnetization = BLDC_Process.PERMEABILITY, BLDC_Process.MAGNETIZATION
    # mu_hw_list, m_hw_list = [], []
    # for x, y, r, th in model_shapely._iter_polar_points(r_in, r_ext, ang_1_rad, ang_2_rad, n_r, n_a):
    #     p = Point(x, y)
    #     mu_r, m, found = permeability['vacuum'], magnetization['vacuum'], False
    #     for material, geoms in model_shapely.geometries.items():
    #         for geom in geoms:
    #             if geom.covers(p):
    #                 mu_r, m, found = permeability[material], magnetization[material], True
    #                 break
    #         if found:
    #             break
    #     mu_hw_list.append(mu_r)
    #     m_hw_list.append(m)
    # Mu_hw = np.asarray(mu_hw_list, dtype=np.float32).reshape(n_r, n_a)
    # M_hw = np.asarray(m_hw_list, dtype=np.float32).reshape(n_r, n_a)

    Mu_hw, M_hw, Bx_hw, By_hw, A_hw = _assign_mesh_to_grid(
        nodes, elems, elem_mu, elem_M, node_mu, node_M, bx, by, nodes[:, 2],
        r_in, r_ext, ang_1_rad, ang_2_rad, n_r, n_a)

    # --- monta node_x [S,9] no layout de produção ---
    node_x = np.stack([
        node_material_id.astype(np.float32),   # 0 material_id
        node_mu.astype(np.float32),             # 1 mu_r
        node_M.astype(np.float32),              # 2 M
        node_dual_area.astype(np.float32),      # 3 "cell_area" -> node_dual_area (mm^2)
        r_base.astype(np.float32),              # 4 r_base
        c_base.astype(np.float32),              # 5 c_base
        frac_dom.astype(np.float32),            # 6 frac_dom
        np.zeros(n_nodes, dtype=np.float32),    # 7 normal_x (placeholder)
        np.zeros(n_nodes, dtype=np.float32),    # 8 normal_y (placeholder)
    ], axis=1)
    node_y = np.stack([bx, by], axis=1)  # [S,2] Bx,By

    # dim salvo como dois escalares (dim_H/dim_W) -- .npz não tem tupla nativa,
    # mesma convenção de sample_processor.py/build_data_chunks.py. motor_params
    # não entra aqui (dict não é nativo de .npz) -- proveniência já fica em
    # valid_designs.csv via BLDC_Process.export_params, igual pipeline qtree.
    arrays = {
        'node_x':     node_x,
        'node_y':     node_y,
        'node_A':     nodes[:, 2].astype(np.float32),
        'edge_index': edge_index,
        'edge_attr':  edge_attr,
        'x_hw_grid':  np.stack([Mu_hw, M_hw], axis=0),
        'y_hw_grid':  np.stack([Bx_hw, By_hw], axis=0),
        'a_hw_grid':  A_hw,
        'L':          np.array([n_nodes], dtype=np.int64),
        'dim_H':      np.array(n_r, dtype=np.int64),
        'dim_W':      np.array(n_a, dtype=np.int64),
        'r_in_mm':    np.array(r_in, dtype=np.float32),
        'r_ext_mm':   np.array(r_ext, dtype=np.float32),
        'ang_1_deg':  np.array(ang_1, dtype=np.float32),
        'ang_2_deg':  np.array(ang_2, dtype=np.float32),
    }

    # escrita atômica: salva em .tmp.npz e renomeia -- mesma técnica de
    # process_and_save_sample (evita arquivo corrompido em caso de falha)
    stem = out_path.stem
    tmp_path = out_path.parent / f"{stem}.tmp"       # np.savez adiciona .npz → .tmp.npz
    tmp_npz  = out_path.parent / f"{stem}.tmp.npz"
    np.savez(tmp_path, **arrays)
    tmp_npz.replace(out_path)

    return {
        'n_nodes': n_nodes,
        'n_edges_undirected': edges_undirected.shape[0],
        'n_edges_wrap': len(wrap_idx_1),
        'n_edges_directed': edge_index.shape[1],
    }
