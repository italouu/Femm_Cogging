"""
ans_parsing.py
---------------
Funções puras de parsing/geometria sobre o `.ans` do FEMM (nós, elementos,
materiais, arestas, grade H×W) -- SEM `import femm` (COM), sem sessão FEMM
aberta, só numpy/matplotlib.tri sobre dados já em memória/disco.

Extraído de src/data_gen/femm_mesh.py em 2026-08-13: essas funções eram
reaproveitadas por src/data_gen/parsers/femm_mesh_v2.py (parse_ans_gzip_sample,
pipeline mode='femm_mesh_v2' -- roda sem FEMM aberto, só sobre o .ans.gz já
salvo), mas femm_mesh.py tem `import femm` no topo (necessário só pras
funções que abrem o FEMM de verdade -- generate_mesh_sample,
save_ans_gzip_sample, _query_grid_pointvalues). Importar qualquer coisa de
femm_mesh.py arrastava esse import junto, mesmo pro caminho 100% numpy --
dependência desnecessária (módulo `femm`/pywin32) pra um código que o
próprio projeto documenta como "deve rodar sem FEMM, possivelmente em
Linux depois" (ver CLAUDE.md, "Raw -- malha real do FEMM v2").

femm_mesh.py reimporta essas funções de volta (`from
src.data_gen.parsers.ans_parsing import ...`) pra continuar funcionando sem
duplicação -- generate_mesh_sample (mode='femm_mesh', v1) ainda usa todas
elas internamente, só que agora como orquestrador (desenha+malha+mi_analyze
via COM, depois chama essas funções puras sobre o resultado).
"""
import numpy as np
import matplotlib.tri as mtri

from src.data_gen.motor_model import BLDC_Process

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

def _grid_polar_xy(r_in: float, r_ext: float, ang_1_rad: float, ang_2_rad: float,
                    n_r: int, n_a: int):
    """Coordenadas (x,y) do centro de cada pixel da grade H×W (flattened,
    ordem row-major r depois c) -- fatorado pra ser reaproveitado tanto por
    _assign_mesh_to_grid (Mu_r/M) quanto por femm_mesh.py::_query_grid_pointvalues
    (A/Bx/By) e src/data_gen/parsers/femm_mesh_v2.py::parse_ans_gzip_sample."""
    dr = (r_ext - r_in) / n_r
    da = (ang_2_rad - ang_1_rad) / n_a
    r_vals = r_in + (np.arange(n_r) + 0.5) * dr
    a_vals = ang_1_rad + (np.arange(n_a) + 0.5) * da
    Rg, Ag = np.meshgrid(r_vals, a_vals, indexing='ij')
    return (Rg * np.cos(Ag)).ravel(), (Rg * np.sin(Ag)).ravel()


def _assign_mesh_to_grid(nodes: np.ndarray, elems: np.ndarray,
                          elem_mu: np.ndarray, elem_M: np.ndarray,
                          node_mu: np.ndarray, node_M: np.ndarray,
                          r_in: float, r_ext: float, ang_1_rad: float, ang_2_rad: float,
                          n_r: int, n_a: int):
    """Projeta Mu_r/M da malha real na grade H×W regular -- SEM Shapely, SEM
    chamada COM ao FEMM (só os dados já extraídos do .ans). Promovido de
    tests/proto_femm_mesh_plot.py::_assign_mesh_to_grid. 2026-07-23.

    Mu_r/M: PIECEWISE-CONSTANT por elemento -- localiza o triângulo que
    contém o centro do pixel via trifinder e copia o valor CONSTANTE desse
    elemento. Preserva fronteiras nítidas (sem blending).

    # [REMOVIDO 2026-07-23, mesmo dia] regime 2 (Bx/By/A via interpolação
    # baricêntrica/LinearTriInterpolator) -- revertido pra point-query ao
    # vivo (mo_getpointvalues) a pedido do usuário, ver
    # femm_mesh.py::_query_grid_pointvalues e nota no topo daquele arquivo.
    # Mu_r/M continuam aqui (via malha) porque point-query NÃO tem como
    # recuperar essas duas colunas corretamente (cobre vira ar por Mu1,
    # magnetização não é consultável — Jm sempre 0 mesmo em ímãs, testado
    # empiricamente).

    Pixels fora da triangulação (trifinder retorna -1 -- esperado raro, só
    arredondamento de ponto flutuante na borda do setor, já que a malha foi
    gerada na MESMA janela r_in/r_ext/ang_1/ang_2 da grade) caem no valor do
    NÓ MAIS PRÓXIMO (fallback via cKDTree, calculado só se necessário).
    """
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=elems[:, :3])
    trifinder = tri.get_trifinder()

    Xg, Yg = _grid_polar_xy(r_in, r_ext, ang_1_rad, ang_2_rad, n_r, n_a)

    elem_idx = trifinder(Xg, Yg)
    valid = elem_idx >= 0
    n_invalid = int((~valid).sum())
    if n_invalid:
        from scipy.spatial import cKDTree
        nearest = cKDTree(nodes[:, :2]).query(
            np.stack([Xg[~valid], Yg[~valid]], axis=1))[1]

    def _assign_const(elem_field, node_fallback):
        out = np.empty(n_r * n_a, dtype=np.float32)
        out[valid] = elem_field[elem_idx[valid]]
        if n_invalid:
            out[~valid] = node_fallback[nearest]
        return out.reshape(n_r, n_a)

    Mu_hw = _assign_const(elem_mu, node_mu)
    M_hw  = _assign_const(elem_M, node_M)
    return Mu_hw, M_hw
