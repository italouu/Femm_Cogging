from dataclasses import dataclass, asdict

from src.neural_op.archs.fno            import FNO2d,          fno_step_fn, fno_metric_fn
from src.neural_op.archs.fno_gnn        import FNO_GNN,        fno_gnn_step, fno_gnn_metric_fn
from src.neural_op.archs.gnn_post_base  import (GNN_PostBase,
                                                gnn_post_base_step_fn,
                                                gnn_post_base_metric_fn)
from src.neural_op.archs.femm_mesh_v2_gnn import (FNO_BipartiteGNN,
                                                  fno_bipartite_gnn_step,
                                                  fno_bipartite_gnn_metric_fn)
from src.neural_op.archs.femm_mesh_elem_gnn import FNO_BipartiteGNN_Elem
from src.configs.training               import (FNOConfig, FNO_GNNConfig, GNN_PostBaseConfig,
                                                FNO_BipartiteGNNConfig, FNO_BipartiteGNN_ElemConfig)


@dataclass
class ArchEntry:
    cls         : type      # classe do modelo
    cfg_cls     : type      # dataclass de config correspondente
    loader_mode : str       # 'grid', 'mesh_v1' ou 'femm_mesh_v2' (dataloaders/grid_loader.build_loaders)
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
        loader_mode='mesh_v1',
        model_kwargs=_fno_kwargs,
        make_step_fn=lambda cfg, lcfg: fno_gnn_step,
        metric_fn=lambda cfg: fno_gnn_metric_fn,
    ),
    'GNN_PostBase': ArchEntry(
        cls=GNN_PostBase,
        cfg_cls=GNN_PostBaseConfig,
        loader_mode='mesh_v1',
        model_kwargs=_gnn_post_base_kwargs,
        make_step_fn=lambda cfg, lcfg: gnn_post_base_step_fn,
        metric_fn=lambda cfg: gnn_post_base_metric_fn,
    ),
    # FNO_BipartiteGNN (2026-08-10): grafo duplo vértices+elementos,
    # mode='femm_mesh_v2' (ver src/data_gen/femm_mesh_v2.py,
    # src/neural_op/archs/femm_mesh_v2_gnn.py).
    'FNO_BipartiteGNN': ArchEntry(
        cls=FNO_BipartiteGNN,
        cfg_cls=FNO_BipartiteGNNConfig,
        loader_mode='femm_mesh_v2',
        model_kwargs=_fno_kwargs,
        make_step_fn=lambda cfg, lcfg: fno_bipartite_gnn_step,
        metric_fn=lambda cfg: fno_bipartite_gnn_metric_fn,
    ),
    # FNO_BipartiteGNN_Elem (PROTÓTIPO 2026-10-07): bipartite com os papéis
    # trocados -- elementos = grafo principal (saída Bx,By por elemento),
    # vértices = auxiliar estático. Chunk com as chaves do femm_mesh_v2
    # (src/data_gen/parsers/femm_mesh_elem.py), então loader/step_fn/metric_fn
    # da bipartite servem sem mudança.
    'FNO_BipartiteGNN_Elem': ArchEntry(
        cls=FNO_BipartiteGNN_Elem,
        cfg_cls=FNO_BipartiteGNN_ElemConfig,
        loader_mode='femm_mesh_v2',
        model_kwargs=_fno_kwargs,
        make_step_fn=lambda cfg, lcfg: fno_bipartite_gnn_step,
        metric_fn=lambda cfg: fno_bipartite_gnn_metric_fn,
    ),
}
