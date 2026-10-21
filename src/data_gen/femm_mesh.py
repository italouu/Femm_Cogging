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

ATUALIZAÇÃO 2026-07-23 (mesmo dia, reversão parcial): A/Bx/By de
x_hw_grid/y_hw_grid/a_hw_grid voltam a vir de point-query AO VIVO
(mo_getpointvalues, ver _query_grid_pointvalues abaixo) — decisão do usuário
após benchmark (ver histórico/conversa): validado que A bate exato (MAE=0)
com a versão via malha e Bx/By têm ~1,5% de ruído de discretização, então a
troca não perde qualidade, só custa ~10x mais tempo nesse passo (~0,4s →
~4s por amostra em 135×270; passo pequeno frente ao solve ~4s e ao loop de B
por nó ~3,5s). material_id/mu_r/M CONTINUAM vindo da malha
([BlockProps]/_assign_mesh_to_grid) — testado e confirmado que não dá pra
substituir por point-query: `Mu1` do cobre resolve exatamente 1.0 (idêntico
a vácuo/ar, 0% de acerto por limiar) e não existe campo de magnetização
consultável por ponto (`Jm`, único candidato, sempre retorna 0 mesmo dentro
de ímãs) — limitações da própria API do FEMM, não do método de acesso.
"""

import gzip
import shutil
from pathlib import Path

import numpy as np
import femm

from src.data_gen.motor_constants import stack_length_mm

from src.data_gen.motor_model import (
    BLDC_Process, BLDC_FEMM_Model_Sym120_Annular,
)

# funções puras de parsing/geometria da malha em parsers/ans_parsing.py (sem `import femm`,
# usadas também pelo pipeline femm_mesh_v2, que roda sem FEMM) -- reimportadas aqui
from src.data_gen.parsers.ans_parsing import (
    _N_MATERIALS, _MAGNET_ID, _MU_BY_ID,
    _parse_solution, _parse_block_materials, _block_magnet_polarity,
    _build_edges, _element_areas, _node_material_stats, _node_magnet_polarity,
    _wrap_edge_pairs, _build_bidirectional_edge_attrs,
    _grid_polar_xy, _assign_mesh_to_grid,
)


def _query_grid_pointvalues(r_in: float, r_ext: float, ang_1_rad: float, ang_2_rad: float,
                             n_r: int, n_a: int):
    """A/Bx/By na grade H×W via point-query AO VIVO no FEMM
    (mo_getpointvalues, 1 chamada COM por pixel -- devolve A, B1, B2, ...,
    Mu1, Mu2 num só round-trip, evitando as 2-3 chamadas redundantes de
    mo_geta+mo_getb+mo_getmu separadas). Precisa ser chamada com a sessão do
    FEMM AINDA ABERTA (antes de femm.closefemm()), com mi_loadsolution() já
    feito. 2026-07-23, decisão do usuário após benchmark: A bate exato
    (MAE=0) com a interpolação baricêntrica da malha; Bx/By têm ~1,5% de
    ruído de discretização — mesma ordem de grandeza do ruído de malha↔malha
    entre rodadas independentes, não uma perda de qualidade real. Custo:
    ~10x mais lento que a versão via malha nesse passo (~0,4s → ~4s por
    amostra em 135×270 — pequeno frente ao solve completo, ~4s, e ao loop de
    B por nó do grafo, ~3,5s).

    NÃO cobre Mu_r/M (ver _assign_mesh_to_grid) — point-query não distingue
    cobre de ar (Mu1 resolve exatamente 1.0 pros dois no ponto convergido) e
    não tem magnetização consultável por ponto (Jm sempre 0, mesmo em
    ímãs) — limitações da própria API do FEMM, confirmadas empiricamente.
    """
    Xg, Yg = _grid_polar_xy(r_in, r_ext, ang_1_rad, ang_2_rad, n_r, n_a)
    n = n_r * n_a
    A_hw = np.empty(n, dtype=np.float32)
    Bx_hw = np.empty(n, dtype=np.float32)
    By_hw = np.empty(n, dtype=np.float32)
    for k in range(n):
        z = femm.mo_getpointvalues(Xg[k], Yg[k])
        A_hw[k] = z[0]
        Bx_hw[k] = z[1]
        By_hw[k] = z[2]
    return A_hw.reshape(n_r, n_a), Bx_hw.reshape(n_r, n_a), By_hw.reshape(n_r, n_a)


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
        x_hw         [2,H,W] float32 Mu_r,M (derivado da malha -- trifinder, valor
                                             constante do elemento; ver _assign_mesh_to_grid.
                                             NÃO vem de point-query -- ver nota 2026-07-23)
        y_hw         [2,H,W] float32 Bx,By  (point-query AO VIVO no FEMM --
                                             mo_getpointvalues; ver _query_grid_pointvalues)
        a_hw         [H,W]   float32 A      (idem Bx/By, point-query ao vivo)
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

    # --- grade H×W: A/Bx/By via point-query ao vivo (mo_getpointvalues) --
    # PRECISA rodar com a sessão ainda aberta. Ver _query_grid_pointvalues
    # acima (decisão do usuário 2026-07-23, mesmo dia da troca anterior pra
    # malha -- benchmark mostrou A exato e Bx/By ~1,5% de ruído, sem perda
    # real de qualidade; custo ~10x maior nesse passo, pequeno frente ao
    # solve+loop de B por nó).
    A_hw, Bx_hw, By_hw = _query_grid_pointvalues(
        r_in, r_ext, ang_1_rad, ang_2_rad, n_r, n_a)

    femm.closefemm()

    # Mu_r/M continuam vindo da malha (ver _assign_mesh_to_grid) -- point-query
    # não recupera essas duas colunas corretamente (ver notas acima/topo do
    # arquivo). A/Bx/By já vieram de _query_grid_pointvalues, acima.
    Mu_hw, M_hw = _assign_mesh_to_grid(
        nodes, elems, elem_mu, elem_M, node_mu, node_M,
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
        # 'x_hw_grid'/'y_hw_grid'/'a_hw_grid' -- renomeadas para 'x_hw'/'y_hw'/'a_hw'
        # (2026-07-27): femm_mesh não tem variante "suavizada" separada (ao contrário
        # do qtree, onde x_hw_grid/x_hw coexistem com significados diferentes), então
        # o sufixo _grid só causava confusão/exigia rename tardio no agrupamento em
        # chunks. Já sai com o nome final na fonte.
        'x_hw':       np.stack([Mu_hw, M_hw], axis=0),
        'y_hw':       np.stack([Bx_hw, By_hw], axis=0),
        'a_hw':       A_hw,
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


# ---------------------------------------------------------------------------
# mode='femm_mesh_v2' -- só geração bruta, salva o .ans inteiro (gzip)
# ---------------------------------------------------------------------------

def save_ans_gzip_sample(motor_params: dict, tmp_dir: Path, out_path: Path) -> dict:
    """Gera 1 amostra (desenha + malha + mi_analyze) e salva o `.ans`
    BRUTO -- texto completo escrito pelo FEMM, sem nenhuma extração ou
    derivação -- comprimido em gzip direto em `out_path`
    (`data/raw/<dataset>/sample_XXXXXX.ans.gz`, escrita atômica).

    Existe pra corrigir a lacuna encontrada em 2026-08-07 (ver CLAUDE.md,
    "Extração de dados direto do arquivo .ans"): `generate_mesh_sample`
    (mode='femm_mesh') deriva node_x/edge_index/etc. e descarta a malha
    original (elems, incluindo a conectividade de elementos) -- sem ela,
    não dá pra recalcular nada (ex: B exato por elemento, validado nesta
    sessão) sem resimular. Guardando o `.ans` inteiro, qualquer derivação
    futura parte só do arquivo, sem FEMM.

    SEM `mi_loadsolution()`/chamada de pós-processamento nenhuma -- não
    precisa, o arquivo que `mi_analyze()` já escreveu em disco é
    exatamente o que se quer guardar.

    Compressão: gzip padrão da stdlib, cópia em bytes crus (`'rb'`/`'wb'`,
    sem tradução de encoding/newline) -- formato portável, decompressível
    em qualquer SO. Quem for parsear o conteúdo depois (Linux, ao que tudo
    indica) deve abrir com `encoding='utf-8'` explícito, nunca depender do
    encoding padrão do SO (nesta sessão, prints no console Windows já
    quebraram por causa disso -- cp1252 não é UTF-8).
    """
    tmp_dir = Path(tmp_dir)
    out_path = Path(out_path)
    fem_file = str(tmp_dir / "model.fem")
    ans_file = tmp_dir / "model.ans"

    phase = float(motor_params['rotor_phase']['value']) if 'rotor_phase' in motor_params else 0.0
    model = BLDC_FEMM_Model_Sym120_Annular(motor_params=motor_params, phase=phase)

    femm.openfemm('bHide')
    femm.main_resize(1000, 1000)
    femm.newdocument(0)
    femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, stack_length_mm(motor_params), 200)
    model.draw_motor()
    femm.mi_zoomnatural()
    femm.mi_saveas(fem_file)
    femm.mi_createmesh()
    femm.mi_analyze()
    femm.closefemm()

    # escrita atômica: salva em .tmp e renomeia -- mesma técnica de
    # generate_mesh_sample/process_and_save_sample
    stem = out_path.stem.removesuffix('.ans')   # sample_XXXXXX (out_path = sample_XXXXXX.ans.gz)
    tmp_gz = out_path.parent / f"{stem}.tmp.ans.gz"
    with open(ans_file, 'rb') as f_in, gzip.open(tmp_gz, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    tmp_gz.replace(out_path)

    return {
        'ans_size_bytes': ans_file.stat().st_size,
        'gz_size_bytes': out_path.stat().st_size,
    }
