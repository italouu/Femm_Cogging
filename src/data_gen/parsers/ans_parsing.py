"""
ans_parsing.py
---------------
Funções puras de parsing/geometria sobre o `.ans` do FEMM (nós, elementos,
materiais, correntes, arestas, grade H×W) -- SEM `import femm` (COM), sem
sessão FEMM aberta, só numpy sobre dados já em memória/disco. Usadas pelos
parsers smooth (femm_mesh_v2/femm_mesh_smooth/femm_mesh_elem*) e pelo
pós-processamento (src/postprocess/), que rodam sem FEMM (Linux).
"""
import gzip
import io
from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.data_gen.motor_constants import MATERIAL_ID, PERMEABILITY, MAGNETIZATION
from src.data_gen.parsers.ans_b_smoothing import AnsMesh, load_ans_lines

_N_MATERIALS = len(set(MATERIAL_ID.values()))  # 4: ferro,ar,ima,cobre
_MAGNET_ID = MATERIAL_ID['N35p']                # == MATERIAL_ID['N35n']

_MU_BY_ID = np.zeros(_N_MATERIALS, dtype=np.float32)
for _name, _mid in MATERIAL_ID.items():
    _MU_BY_ID[_mid] = PERMEABILITY[_name]


# ---------------------------------------------------------------------------
# leitura do .ans (só arquivo, sem chamada COM)
# ---------------------------------------------------------------------------

def _parse_solution(ans_path):
    with open(ans_path) as f:
        lines = f.readlines()
    return _parse_solution_lines(lines)


def _parse_solution_lines(lines):
    """_parse_solution sobre as linhas já lidas (readlines, com '\\n')."""
    def section(name):
        return next(i for i, l in enumerate(lines) if l.strip().startswith(name))

    idx = section('[Solution]')
    n_nodes = int(lines[idx + 1].strip())
    nodes = np.loadtxt(lines[idx + 2: idx + 2 + n_nodes])[:, :3]  # x, y, A
    idx2 = idx + 2 + n_nodes
    n_elems = int(lines[idx2].strip())
    elems = np.loadtxt(lines[idx2 + 1: idx2 + 1 + n_elems])[:, :4].astype(np.int64)  # n1,n2,n3,label
    return lines, nodes, elems


class AnsData(NamedTuple):
    """`.ans` de uma amostra lido uma vez (read_ans): saída de _parse_solution + AnsMesh do
    load_ans. Passado aos parsers (`ans=`) para não descomprimir/ler o arquivo de novo."""
    lines: list
    nodes: np.ndarray       # [S,3] x, y, A
    elems: np.ndarray       # [M,4] n1, n2, n3, label
    mesh: AnsMesh


def read_ans(ans_gz_path: Path) -> AnsData:
    """Descomprime o `.ans.gz` uma vez, em memória (2026-10-22) -- leitura única do `.ans`
    em todo o código (parsers, pós-processamento). Linhas em utf-8 explícito (antes: a
    codificação padrão do sistema, cp1252 no Windows; os 6000 .ans do raw BLDC_138x276 são
    ASCII puro -> linhas idênticas, conferido); o AnsMesh (load_ans_lines) com
    utf-8/errors='replace', como o load_ans."""
    with gzip.open(ans_gz_path, 'rb') as f:
        raw = f.read()
    lines = io.TextIOWrapper(io.BytesIO(raw), encoding='utf-8').readlines()
    _, nodes, elems = _parse_solution_lines(lines)
    text = io.TextIOWrapper(io.BytesIO(raw), encoding='utf-8', errors='replace').read()
    return AnsData(lines, nodes, elems, load_ans_lines(text.splitlines()))


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
            material_id[b] = MATERIAL_ID[canon]
            mu_const[b] = PERMEABILITY[canon]
            b += 1
            mu_x = has_bh = None
        i += 1

    return material_id, mu_const


def _parse_label_materials(lines):
    """Por RÓTULO DE BLOCO ([BlockLabels], 0-based) — o índice que o campo
    `label` de cada elemento do .ans de fato referencia: material_id, mu_r
    constante e M (polaridade do ímã, MAGNETIZATION +-1). 2026-10-05.

    Substitui o par _parse_block_materials + polaridade por ordem de bloco indexado
    direto pelo label do elemento, que só funcionava por coincidência (cada
    _create_sec chama mi_getmaterial logo antes de criar o rótulo, então
    [BlockProps] e [BlockLabels] saem com a mesma ordem/tamanho) e exigia
    exatamente N_POLES_SECTOR blocos de ímã — quebra com a fase do rotor
    aplicada (polo partido no corte = 15 blocos). Aqui:
      - material: rótulo -> sua referência em [BlockProps] (3ª coluna, 1-based)
        -> _parse_block_materials;
      - polaridade: magdir do rótulo (6ª coluna, graus) comparado ao ângulo da
        posição do rótulo — magnetização radial para fora (N35p, magdir ~ θ)
        ou para dentro (N35n, magdir ~ θ+180°). Independe de ordem/contagem.
    Validado idêntico ao método antigo nos .ans de fase 0 (ver CLAUDE.md).
    """
    prop_material_id, prop_mu = _parse_block_materials(lines)
    j = next(i for i, l in enumerate(lines) if l.strip().startswith('[NumBlockLabels]'))
    n_labels = int(lines[j].split('=')[1].strip())
    lab = np.array([[float(v) for v in l.split()[:6]] for l in lines[j + 1:j + 1 + n_labels]])
    x, y, magdir = lab[:, 0], lab[:, 1], lab[:, 5]
    prop = lab[:, 2].astype(np.int64) - 1
    if (prop < 0).any():
        raise ValueError("rótulo de bloco sem material ([BlockLabels] com blockprop=0)")

    label_material_id = prop_material_id[prop]
    label_mu = prop_mu[prop]
    label_M = np.zeros(n_labels, dtype=np.float32)
    mag = label_material_id == _MAGNET_ID
    outward = np.cos(np.deg2rad(magdir[mag] - np.degrees(np.arctan2(y[mag], x[mag])))) > 0
    label_M[mag] = np.where(outward, MAGNETIZATION['N35p'], MAGNETIZATION['N35n'])
    return label_material_id, label_mu, label_M


def _parse_label_currents(lines, elems: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Densidade de corrente J [A/mm²] por RÓTULO de bloco (2026-10-14).

    Na magnetostática o FEMM trata o bloco num circuito em série como condutor
    sólido de J uniforme = Turns·TotalAmps / área do bloco (os parâmetros de fio só
    importam em CA) — o mesmo J que entra na EDP. TotalAmps vem de [CircuitProps];
    InCircuit (5ª coluna de [BlockLabels], 1-based, 0 = nenhum) e Turns (8ª); a área
    do bloco é a soma das áreas dos elementos do rótulo. Rótulos fora de circuito e
    .ans sem circuitos -> 0. Validado em tests/proto_currents_sample.py.
    """
    amps, cur = [], None
    for l in lines:
        s = l.strip()
        if s.startswith('<BeginCircuit>'):
            cur = 0.0
        elif cur is not None and s.startswith('<TotalAmps_re>'):
            cur = float(s.split('=', 1)[1])
        elif s.startswith('<EndCircuit>'):
            amps.append(cur)
            cur = None
        elif s.startswith('[NumBlockLabels]'):
            break
    amps = np.array(amps, dtype=np.float64)

    j = next(i for i, l in enumerate(lines) if l.strip().startswith('[NumBlockLabels]'))
    n_labels = int(lines[j].split('=')[1].strip())
    lab = [l.split() for l in lines[j + 1:j + 1 + n_labels]]
    in_circ = np.array([int(t[4]) - 1 for t in lab], dtype=np.int64)
    turns = np.array([float(t[7]) for t in lab], dtype=np.float64)

    label_area = np.bincount(elems[:, 3], weights=area, minlength=n_labels)
    label_J = np.zeros(n_labels, dtype=np.float64)
    has = in_circ >= 0
    label_J[has] = turns[has] * amps[in_circ[has]] / label_area[has]
    return label_J.astype(np.float32)


def _node_current_density(elems: np.ndarray, area: np.ndarray, elem_J: np.ndarray,
                          elem_material_id: np.ndarray,
                          node_material_id: np.ndarray) -> np.ndarray:
    """J por nó (2026-10-14), seguindo o material do nó como M em
    _node_magnet_polarity: nó de cobre -> média de J dos elementos de cobre
    incidentes, ponderada por área (nó na divisa ida/volta da mesma ranhura fica
    com a média dos dois lados); qualquer outro material -> 0."""
    n_nodes = node_material_id.shape[0]
    is_cu = elem_material_id == MATERIAL_ID['copper']
    num = np.zeros(n_nodes, dtype=np.float64)
    den = np.zeros(n_nodes, dtype=np.float64)
    for corner in range(3):
        idx = elems[:, corner]
        np.add.at(num, idx[is_cu], (area * elem_J)[is_cu])
        np.add.at(den, idx[is_cu], area[is_cu])
    node_J = np.divide(num, den, out=np.zeros(n_nodes), where=den > 0)
    node_J[node_material_id != MATERIAL_ID['copper']] = 0.0
    return node_J.astype(np.float32)


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

    return node_material_id, frac_dom.astype(np.float32), _node_dual_area(elems, area, n_nodes)


def _node_dual_area(elems: np.ndarray, area: np.ndarray, n_nodes: int) -> np.ndarray:
    """Área lumped do nó (mm^2): soma de 1/3 da área dos triângulos incidentes."""
    node_dual_area = np.zeros(n_nodes, dtype=np.float64)
    for corner in range(3):
        np.add.at(node_dual_area, elems[:, corner], area / 3.0)
    return node_dual_area.astype(np.float32)


def _node_magnet_polarity(nodes: np.ndarray, elems: np.ndarray, area: np.ndarray,
                           elem_material_id: np.ndarray, elem_M: np.ndarray) -> np.ndarray:
    """M por nó, restrito a nós que tocam ao menos um elemento classificado
    como ímã: votação por área (mesmo princípio de frac_dom) SÓ entre os
    elementos incidentes já classificados como ímã, escolhendo o sinal
    (+-MAGNETIZATION) de maior área somada. Nós que não tocam nenhum ímã
    ficam com M=0 (ferro/ar/cobre já têm MAGNETIZATION=0 em toda a área).

    100% dados do FEMM (elem_M vem de _parse_label_materials) — sem
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
    node_M = np.where(touches_magnet, sign * MAGNETIZATION['N35p'], 0.0)
    return node_M.astype(np.float32)


#: ordem de prioridade de material por nó (ferro > ímã > cobre > ar) --
#: decisão do usuário 2026-10-09, ver CLAUDE.md "Suavização de B do FEMM".
MATERIAL_PRIORITY = (MATERIAL_ID['iron_1008'], _MAGNET_ID,
                     MATERIAL_ID['copper'], MATERIAL_ID['vacuum'])


def _node_material_priority(elems: np.ndarray, elem_material_id: np.ndarray, n_nodes: int,
                             priority=MATERIAL_PRIORITY) -> np.ndarray:
    """Material do nó = o de MAIOR prioridade entre os elementos incidentes
    (alternativa ao voto por área de _node_material_stats, que serrilha as
    interfaces -- ~50% das arestas de interface com lado trocado). 2026-10-09.

    Retorna node_material_id [n_nodes] (int64).
    """
    rank_of = np.empty(_N_MATERIALS, dtype=np.int64)
    rank_of[np.asarray(priority)] = np.arange(len(priority))
    elem_rank = rank_of[elem_material_id]
    node_rank = np.full(n_nodes, len(priority), dtype=np.int64)
    for corner in range(3):
        np.minimum.at(node_rank, elems[:, corner], elem_rank)
    if (node_rank == len(priority)).any():
        raise ValueError("nó sem elemento incidente")
    return np.asarray(priority, dtype=np.int64)[node_rank]


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


def _element_b_from_A(nodes: np.ndarray, elems: np.ndarray, node_A: np.ndarray):
    """B=curl(A) constante por elemento (P1 linear) -- fórmula fechada
    validada em tests/proto_femm_element_b_check.py (bate com
    mo_getb(smooth='off') a ~1e-16). Gabarito exato por elemento do torque
    (femm_mesh_elem.parse_ans_gzip_sample_elem, target_field='B').

    Bx=dA/dy, By=-dA/dx. `nodes[:, :2]` está em mm (convenção mi_probdef do
    projeto) -- convertido pra metro aqui, senão o erro é de ~1000x (achado
    2026-08-07, ver CLAUDE.md "Extração de dados direto do arquivo .ans").

    As coordenadas reais vêm direto do .ans, então B sai no referencial global
    de verdade, sem ambiguidade de rotação por elemento (antes, o
    eval.py::_element_curl_A, já removido, reconstruía o triângulo por lei dos
    cossenos, sem coordenadas absolutas). Vetorizado -- retorna (Bx, By), cada um [n_elems].
    """
    xm = nodes[:, 0].astype(np.float64) * 1e-3
    ym = nodes[:, 1].astype(np.float64) * 1e-3
    x0, y0 = xm[elems[:, 0]], ym[elems[:, 0]]
    x1, y1 = xm[elems[:, 1]], ym[elems[:, 1]]
    x2, y2 = xm[elems[:, 2]], ym[elems[:, 2]]
    A0 = node_A[elems[:, 0]].astype(np.float64)
    A1 = node_A[elems[:, 1]].astype(np.float64)
    A2 = node_A[elems[:, 2]].astype(np.float64)

    area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    b0, b1, b2 = y1 - y2, y2 - y0, y0 - y1
    c0, c1, c2 = x2 - x1, x0 - x2, x1 - x0
    dAdx = (A0 * b0 + A1 * b1 + A2 * b2) / area2
    dAdy = (A0 * c0 + A1 * c1 + A2 * c2) / area2
    return dAdy.astype(np.float32), (-dAdx).astype(np.float32)


def _grid_polar_xy(r_in: float, r_ext: float, ang_1_rad: float, ang_2_rad: float,
                    n_r: int, n_a: int):
    """Coordenadas (x,y) do centro de cada pixel da grade H×W (flattened,
    ordem row-major r depois c) -- usado pelos parsers (femm_mesh_v2,
    femm_mesh_smooth, femm_mesh_elem_smooth) e pelo pós-processamento."""
    dr = (r_ext - r_in) / n_r
    da = (ang_2_rad - ang_1_rad) / n_a
    r_vals = r_in + (np.arange(n_r) + 0.5) * dr
    a_vals = ang_1_rad + (np.arange(n_a) + 0.5) * da
    Rg, Ag = np.meshgrid(r_vals, a_vals, indexing='ij')
    return (Rg * np.cos(Ag)).ravel(), (Rg * np.sin(Ag)).ravel()


def _to_polar(bx: np.ndarray, by: np.ndarray, x: np.ndarray, y: np.ndarray):
    """(Bx, By) -> (Br, Bθ) pelo ângulo θ = atan2(y, x) de cada ponto (nó,
    centróide ou centro do pixel). Gabarito dos parsers smooth desde 2026-10-17
    (antes cartesiano). Rotação em float64; retorna float32."""
    th = np.arctan2(np.asarray(y, np.float64), np.asarray(x, np.float64))
    c, s = np.cos(th), np.sin(th)
    bx, by = np.asarray(bx, np.float64), np.asarray(by, np.float64)
    return (bx * c + by * s).astype(np.float32), (-bx * s + by * c).astype(np.float32)
