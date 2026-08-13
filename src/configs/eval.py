from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalCfg:
    run_dir              : str   = 'data/logs/mesh_ans_138x276/FNO_BipartiteGNN/run_0001' # caminho para run_XXXX/ (data/logs/{problem}/{arch}/run_XXXX)
    checkpoint           : str   = 'best'              # 'best', 'latest' ou 'final'
    split                : str   = 'test'               # 'train' ou 'test' — conjunto de onde o chunk vem
    chunk_index          : Optional[int] = 1         # N-ésimo chunk (0-indexed) dentro de `split`, reproduzindo
                                                         # o mesmo shuffle/seed do treino da run; se None, usa chunk_name
    chunk_name           : str   = 'data_chunk_0000'   # sem extensão .pt; usado só se chunk_index=None; dataset lido de run_dir/config.json
    dataset_override     : Optional[str] = None        # nome de pasta em data/torch/data_chunks/; se None, usa dataset do
                                                         # config.json da run (treino); se setado, ignora chunk_index (split de
                                                         # treino não é reproduzível em outro dataset) e usa chunk_name
    sample_idx           : int   = 0
    irrelevance_threshold: float = 0.0001
    show_qtree_overlay   : bool  = False               # overlay de refinamento (FNO_GNN)
    error_cap_enabled    : bool  = False                # capa vmax do colormap de erro em error_cap
    error_cap            : float = 20.0                # vmax do erro normalizado [%] quando error_cap_enabled=True
    error_plot_mode      : str   = 'percent'           # 'percent' (erro norm. por B_ref, %) ou
                                                         # 'absolute' (erro bruto |pred-alvo|, Tesla)
    qtree_metric_enabled : bool  = False                # fno_eval_fn: projeta a predição (grade base) nas
                                                         # folhas da qtree e calcula erro vs node_y (GT fino);
                                                         # requer chunk com node_x/node_y/L (parser FNO_GNN)
    show_b               : bool  = False                # femm_mesh_v2_eval_fn (FNO_BipartiteGNN, alvo=A):
                                                         # False (padrão/'normal') -> plota A cru, sem pós-
                                                         # processamento. True ('show B') -> deriva B=curl(A)
                                                         # por elemento (fórmula P1 fechada, já validada contra
                                                         # mo_getb) a partir de GT/FNO@nós/GNN e plota Br/Bθ/|B|
                                                         # (radial/tangencial, não Bx/By -- malha reconstruída
                                                         # sem coordenadas absolutas, ver eval.py::
                                                         # _element_radial_tangential) em vez de A. B NÃO é o
                                                         # alvo de treino deste arch — é só visualização pós-hoc;
                                                         # reconstrói (x,y) reais em mm só a partir do próprio
                                                         # chunk (sem tocar CSV/raw). |B| exato; Br/Bθ com erro
                                                         # pequeno (~0.1-0.3% em malha de produção).