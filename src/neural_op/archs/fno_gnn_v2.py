from src.neural_op.archs.fno_gnn import FNO_GNN


class FNO_GNN_v2(FNO_GNN):
    """
    Variante de FNO_GNN com edge_attr [E, 5]: adiciona delta_mu direcional
    (mu_r[origem] − mu_r[destino]) às 4 features geométricas de aresta.

    Criada em 2026-07-17 a pedido do usuário, em vez de alterar FNO_GNN
    diretamente — mantém FNO_GNN/FNO_GNN_Field e runs já treinadas intactas.
    Mesma arquitetura, mesmo forward, mesma step_fn e mesmo eval_fn de
    FNO_GNN (herdados); só _EDGE_DIM muda (4 → 5), o que redimensiona a
    primeira camada do GNN (EdgeConvLayer opera sobre node_width + edge_dim).

    Requer chunks gerados pelo parser FNO_GNN_V2_PARSER
    (src/data_gen/parsers/fno_gnn_v2.py) — chunks de FNO_GNN_PARSER têm
    edge_attr [E, 4] e não são compatíveis (mismatch de shape no GNN).
    """
    _EDGE_DIM = 5
