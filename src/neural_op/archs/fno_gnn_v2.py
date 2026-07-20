from src.neural_op.archs.fno_gnn import FNO_GNN


class FNO_GNN_v2(FNO_GNN):
    """
    Variante de FNO_GNN com edge_attr [E, 5]: adiciona delta_mu direcional
    (mu_r[origem] − mu_r[destino]) às 4 features geométricas de aresta.

    Criada em 2026-07-17 a pedido do usuário, em vez de alterar FNO_GNN
    diretamente — mantém FNO_GNN/FNO_GNN_Field e runs já treinadas intactas.

    2026-07-20: edge_dim deixou de ser uma constante de classe (_EDGE_DIM) e
    passou a vir do construtor, auto-detectado a partir do dataset (ver
    FNO_GNNConfig/NnCfg.__post_init__) — na prática esta classe hoje é
    idêntica a FNO_GNN. Mantida (e mantida registrada em ARCH_REGISTRY) só
    por compatibilidade: reconstruir/avaliar runs já treinadas com
    "arch": "FNO_GNN_v2" no config.json depende desse nome existir.

    Requer chunks gerados pelo parser FNO_GNN_V2_PARSER
    (src/data_gen/parsers/fno_gnn_v2.py) — chunks de FNO_GNN_PARSER têm
    edge_attr [E, 4] e não são compatíveis (mismatch de shape no GNN).
    """
    pass
