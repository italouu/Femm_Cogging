from dataclasses import dataclass, asdict
from functools import partial

from src.neural_op.archs.fno            import FNO2d,          fno_step_fn, fno_metric_fn
from src.neural_op.archs.fno_gnn        import FNO_GNN,        fno_gnn_step, fno_gnn_metric_fn
from src.neural_op.archs.gnn_post_base  import (GNN_PostBase,
                                                gnn_post_base_step_fn,
                                                gnn_post_base_metric_fn)
from src.neural_op.archs.fno_bipartite_gnn import (FNO_BipartiteGNN,
                                                  fno_bipartite_gnn_step,
                                                  fno_bipartite_gnn_metric_fn)
from src.neural_op.archs.fno_bipartite_gnn_elem import FNO_BipartiteGNN_Elem
from src.neural_op.archs.fno_split_gnn  import (FNO_SplitGNN,
                                                fno_split_gnn_step,
                                                fno_split_gnn_metric_fn)
# [REMOVIDO] FNO_SplitGNN_Base (saída e loss só em N_base) -- motivo: pior que o FNO_SplitGNN
# em superfície e torque (2,73% contra 2,45% de L1_A no mae); arquivo fno_split_gnn_base.py apagado.
# from src.neural_op.archs.fno_split_gnn_base import (FNO_SplitGNN_Base,
#                                                     fno_split_gnn_base_step,
#                                                     fno_split_gnn_base_metric_fn)
from src.configs.training               import (FNOConfig, FNO_GNNConfig, GNN_PostBaseConfig,
                                                FNO_BipartiteGNNConfig, FNO_BipartiteGNN_ElemConfig,
                                                FNO_SplitGNNConfig)
# [REMOVIDO] FNO_SplitGNN_BaseConfig do import acima -- motivo: FNO_SplitGNN_Base removida.
#                                               FNO_SplitGNNConfig, FNO_SplitGNN_BaseConfig)


@dataclass
class ArchEntry:
    cls         : type      # classe do modelo
    cfg_cls     : type      # dataclass de config correspondente
    loader_mode : str       # 'grid', 'vertex' ou 'bipartite' (dataloaders/loaders.build_loaders)
    model_kwargs: callable  # (arch_cfg) -> dict com kwargs do construtor do modelo
    make_step_fn: callable  # (arch_cfg, loss_cfg) -> step_fn(batch, model, loss_fn, device)
    metric_fn   : callable  # (arch_cfg) -> metric_fn(batch, model, device) -> (mae_hw, mae_graph|None)
                            # MAE bruto (sem máscara) para monitoramento em metrics.jsonl,
                            # em T, só sobre a saída final: FNO2d -> mae_hw (grade H×W),
                            # mae_graph=None; demais -> mae_hw=None (grade intermediária
                            # não supervisionada), mae_graph nos nós (elementos na Elem)

    def make_model(self, arch_cfg):
        return self.cls(**self.model_kwargs(arch_cfg))


def _fno_kwargs(cfg):
    return asdict(cfg)




def _gnn_post_base_kwargs(cfg):
    # base_arch/base_arch_cfg (snapshot, init=False) são passados como fallback ao
    # construtor — não fazem parte do model_kwargs "normal", mas precisam chegar até
    # GNN_PostBase para reconstrução resiliente caso base_run_dir não exista mais.
    return {
        'base_run_dir':     cfg.base_run_dir,
        'base_checkpoint':  cfg.base_checkpoint,
        'gnn_node_width':   cfg.gnn_node_width,
        'gnn_n_layers':     cfg.gnn_n_layers,
        'base_arch':        cfg.base_arch,
        'base_arch_cfg':    cfg.base_arch_cfg,
        'edge_dim':         cfg.edge_dim,
        'node_in_ch':       cfg.node_in_ch,
        'base_out_ch':      cfg.base_out_ch,
        'base_normalize':   cfg.base_normalize,
        'base_norm_stats':  cfg.base_norm_stats,
    }


ARCH_REGISTRY: dict = {
    'FNO2d': ArchEntry(
        cls=FNO2d,
        cfg_cls=FNOConfig,
        loader_mode='grid',
        model_kwargs=_fno_kwargs,
        make_step_fn=lambda cfg, lcfg: fno_step_fn,
        metric_fn=lambda cfg: fno_metric_fn,
    ),
    'FNO_GNN': ArchEntry(
        cls=FNO_GNN,
        cfg_cls=FNO_GNNConfig,
        loader_mode='vertex',
        model_kwargs=_fno_kwargs,
        make_step_fn=lambda cfg, lcfg: fno_gnn_step,
        metric_fn=lambda cfg: fno_gnn_metric_fn,
    ),
    'GNN_PostBase': ArchEntry(
        cls=GNN_PostBase,
        cfg_cls=GNN_PostBaseConfig,
        loader_mode='vertex',
        model_kwargs=_gnn_post_base_kwargs,
        make_step_fn=lambda cfg, lcfg: gnn_post_base_step_fn,
        metric_fn=lambda cfg: gnn_post_base_metric_fn,
    ),
    # FNO_BipartiteGNN: grafo duplo vértices+elementos,
    # mode='bipartite' (ver src/data_gen/parsers/bipartite_structure.py,
    # src/neural_op/archs/fno_bipartite_gnn.py).
    'FNO_BipartiteGNN': ArchEntry(
        cls=FNO_BipartiteGNN,
        cfg_cls=FNO_BipartiteGNNConfig,
        loader_mode='bipartite',
        model_kwargs=_fno_kwargs,
        make_step_fn=lambda cfg, lcfg: fno_bipartite_gnn_step,
        metric_fn=lambda cfg: fno_bipartite_gnn_metric_fn,
    ),
    # FNO_BipartiteGNN_Elem: a rede da bipartite com os papéis trocados no layout
    # dos dados -- elementos = grafo principal (saída (Br, Bθ) por elemento),
    # vértices = auxiliar estático. Chunk com as chaves do bipartite_structure
    # (src/data_gen/parsers/elem_structure.py), então loader/step_fn/metric_fn
    # da bipartite servem sem mudança.
    'FNO_BipartiteGNN_Elem': ArchEntry(
        cls=FNO_BipartiteGNN_Elem,
        cfg_cls=FNO_BipartiteGNN_ElemConfig,
        loader_mode='bipartite',
        model_kwargs=_fno_kwargs,
        make_step_fn=lambda cfg, lcfg: fno_bipartite_gnn_step,
        metric_fn=lambda cfg: fno_bipartite_gnn_metric_fn,
    ),
    # FNO_SplitGNN: grafo dual de elementos (N_aux) + vértices divididos por material
    # (N_base), os dois com estado e saída. Chunk com as chaves do bipartite + elem_y
    # (src/data_gen/parsers/split_layout.py); loader 'bipartite' repassa elem_y.
    # loss_lambda é da loss, não do construtor: fora do model_kwargs, entra no step_fn.
    'FNO_SplitGNN': ArchEntry(
        cls=FNO_SplitGNN,
        cfg_cls=FNO_SplitGNNConfig,
        loader_mode='bipartite',
        # [REMOVIDO] model_kwargs=_fno_kwargs e make_step_fn sem λ -- motivo: loss_lambda
        # no FNO_SplitGNNConfig (não é kwarg do modelo; o step_fn passa a recebê-lo).
        # model_kwargs=_fno_kwargs,
        # make_step_fn=lambda cfg, lcfg: fno_split_gnn_step,
        model_kwargs=lambda cfg: {k: v for k, v in asdict(cfg).items() if k != 'loss_lambda'},
        make_step_fn=lambda cfg, lcfg: partial(fno_split_gnn_step, loss_lambda=cfg.loss_lambda),
        metric_fn=lambda cfg: fno_split_gnn_metric_fn,
    ),
    # [REMOVIDO] FNO_SplitGNN_Base -- motivo: pior que o FNO_SplitGNN em superfície e torque.
    # # FNO_SplitGNN_Base: a mesma rede, saída e loss só em N_base (elem_y); N_aux fica só
    # # como nó de passagem de mensagem. Mesmos chunks do FNO_SplitGNN.
    # 'FNO_SplitGNN_Base': ArchEntry(
    #     cls=FNO_SplitGNN_Base,
    #     cfg_cls=FNO_SplitGNN_BaseConfig,
    #     loader_mode='bipartite',
    #     model_kwargs=_fno_kwargs,
    #     make_step_fn=lambda cfg, lcfg: fno_split_gnn_base_step,
    #     metric_fn=lambda cfg: fno_split_gnn_base_metric_fn,
    # ),
}
