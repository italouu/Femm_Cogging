"""
parsers/femm_mesh_v3.py
------------------------
Estende parse_ans_gzip_sample (femm_mesh_v2.py) com uma feature adicional de
nó do grafo de vértices, pedida pelo usuário (2026-08-20): node_cell_count --
quantos vértices da malha caem na MESMA célula da grade base H×W que aquele
nó (densidade de amostragem da malha por pixel do FNO).

A segunda parte do pedido -- "projeção das 8 células vizinhas vinda do FNO"
-- NÃO entra aqui: a saída do FNO só existe em tempo de forward do modelo
(depende de pesos treinados), não em tempo de parsing (só temos o .ans.gz
bruto). Essa parte fica inteiramente em
src/neural_op/archs/femm_mesh_v3_gnn.py (_interpolate_fno_to_nodes_v3),
usando r_base/c_base (colunas 0,1 de node_x, que continuam idênticas ao v2)
pra fazer o gather na saída do FNO a cada forward.

node_x passa de [S,2] (r_base,c_base) para [S,3] (r_base,c_base,
node_cell_count). Resto do dict (grafo de elementos, arestas cruzadas,
x_hw/y_hw) idêntico a parse_ans_gzip_sample -- reaproveitado sem duplicação,
só pós-processado.

Índice de célula: floor(r_base*n_r) clipado em [0,n_r) -- SEM wrap radial
(não é periódico); floor(c_base*n_a) mod n_a -- COM wrap angular, mesma
convenção de periodicidade θ=0/120° já usada no grafo (ver
ans_parsing.py::_wrap_edge_pairs). node_cell_count é a contagem de nós
(dentro da MESMA amostra) que caem nessa célula, broadcastada de volta pra
cada nó que a compartilha.
"""
import numpy as np

from src.data_gen.parsers.femm_mesh_v2 import parse_ans_gzip_sample


def parse_ans_gzip_sample_v3(ans_gz_path, r_in, r_ext, ang_1: float = 0.0, ang_2: float = 120.0,
                              n_r: int = 138, n_a: int = 276, tmp_dir=None,
                              target_field: str = 'A') -> dict:
    """Idêntico a parse_ans_gzip_sample (mesmos parâmetros/retorno), com
    node_x estendido de [S,2] para [S,3] -- ver docstring do módulo."""
    d = parse_ans_gzip_sample(ans_gz_path, r_in, r_ext, ang_1=ang_1, ang_2=ang_2,
                               n_r=n_r, n_a=n_a, tmp_dir=tmp_dir, target_field=target_field)

    r_base = d['node_x'][:, 0]
    c_base = d['node_x'][:, 1]
    cell_r = np.clip(np.floor(r_base * n_r), 0, n_r - 1).astype(np.int64)
    cell_c = np.mod(np.floor(c_base * n_a), n_a).astype(np.int64)
    flat_idx = cell_r * n_a + cell_c

    counts = np.bincount(flat_idx, minlength=n_r * n_a)
    node_cell_count = counts[flat_idx].astype(np.float32)

    d['node_x'] = np.concatenate([d['node_x'], node_cell_count[:, None]], axis=1)
    return d
