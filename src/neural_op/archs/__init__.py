from dataclasses import dataclass, asdict

from src.neural_op.archs.fno            import FNO2d,          fno_step_fn, fno_metric_fn
from src.neural_op.archs.fno_gnn        import FNO_GNN,        make_fno_gnn_step, fno_gnn_metric_fn
from src.neural_op.archs.fno_gnn_v2     import FNO_GNN_v2
from src.neural_op.archs.fno_gnn_field  import FNO_GNN_Field
from src.neural_op.archs.fno_mat        import (MaskedFNO2d,    masked_fno_step_fn, masked_fno_metric_fn,
                                                FNO2d_SingleMat, make_single_mat_step_fn,
                                                make_single_mat_metric_fn)
from src.neural_op.archs.masked_fno_gnn import (MaskedFNO_GNN, make_masked_fno_gnn_step,
                                                masked_fno_gnn_metric_fn)
from src.neural_op.archs.gnn_post_base  import (GNN_PostBase,
                                                gnn_post_base_step_fn,
                                                make_gnn_post_base_step_fn,
                                                gnn_post_base_metric_fn)
from src.neural_op.archs.eval           import (fno_eval_fn, fno_gnn_eval_fn,
                                                masked_fno_eval_fn, masked_fno_gnn_eval_fn,
                                                single_mat_fno_eval_fn)
from src.configs.training               import (FNOConfig, FNO_GNNConfig, FNO_GNN_v2Config,
                                                MaskedFNO2dConfig, MaskedFNO_GNNConfig,
                                                SingleMatFNOConfig, GNN_PostBaseConfig)
# [REMOVIDO] PhiDeepONet — removido do escopo ativo (2026-05-27)
# from src.neural_op.archs.phi_deeponet import PhiDeepONet,        phi_deeponet_step_fn
# from src.neural_op.archs.eval         import phi_deeponet_eval_fn
# from src.configs.training             import PhiDeepONetConfig


@dataclass
class ArchEntry:
    cls         : type      # classe do modelo
    cfg_cls     : type      # dataclass de config correspondente
    loader_mode : str       # 'grid' ou 'qtree'
    model_kwargs: callable  # (arch_cfg) -> dict com kwargs do construtor do modelo
    make_step_fn: callable  # (arch_cfg, loss_cfg) -> step_fn(batch, model, loss_fn, device)
    eval_fn     : callable  # (model, chunk_data, eval_cfg) -> None
    metric_fn   : callable  # (arch_cfg) -> metric_fn(batch, model, device) -> (mae_hw, mae_graph|None)
                            # MAE bruto (sem máscara) para monitoramento em metrics.jsonl;
                            # mae_hw sempre compara a saída em grade H×W do estágio FNO;
                            # mae_graph compara a saída final nos nós/grafo (None se o
                            # arch não produz saída em grafo — ex: FNO2d, MaskedFNO2d)

    def make_model(self, arch_cfg):
        return self.cls(**self.model_kwargs(arch_cfg))


def _fno_kwargs(cfg):
    return asdict(cfg)


def _fno_gnn_kwargs(cfg):
    # lambda_loss não é parâmetro do construtor do modelo — usado apenas na step_fn
    return {k: v for k, v in asdict(cfg).items() if k != 'lambda_loss'}


# [REMOVIDO] _phi_deeponet_kwargs — removido com PhiDeepONet (2026-05-27)
# def _phi_deeponet_kwargs(cfg):
#     return {k: v for k, v in asdict(cfg).items() if k != 'data_res'}


def _gnn_post_base_kwargs(cfg):
    # base_arch/base_arch_cfg (snapshot, init=False) são passados como fallback ao
    # construtor — não fazem parte do model_kwargs "normal", mas precisam chegar até
    # GNN_PostBase para reconstrução resiliente caso base_run_dir não exista mais.
    return {
        'base_run_dir':    cfg.base_run_dir,
        'base_checkpoint': cfg.base_checkpoint,
        'gnn_node_width':  cfg.gnn_node_width,
        'gnn_n_layers':    cfg.gnn_n_layers,
        'base_arch':       cfg.base_arch,
        'base_arch_cfg':   cfg.base_arch_cfg,
    }


ARCH_REGISTRY: dict = {
    'FNO2d': ArchEntry(
        cls=FNO2d,
        cfg_cls=FNOConfig,
        loader_mode='grid',
        model_kwargs=_fno_kwargs,
        make_step_fn=lambda cfg, lcfg: fno_step_fn,
        eval_fn=fno_eval_fn,
        metric_fn=lambda cfg: fno_metric_fn,
    ),
    # [REMOVIDO] 'phi_DeepONet' — removido do registry (2026-05-27)
    # 'phi_DeepONet': ArchEntry(
    #     cls=PhiDeepONet, cfg_cls=PhiDeepONetConfig, loader_mode='qtree',
    #     model_kwargs=_phi_deeponet_kwargs,
    #     make_step_fn=lambda cfg, lcfg: phi_deeponet_step_fn,
    #     eval_fn=phi_deeponet_eval_fn,
    # ),
    'FNO_GNN': ArchEntry(
        cls=FNO_GNN,
        cfg_cls=FNO_GNNConfig,
        loader_mode='qtree',
        model_kwargs=_fno_gnn_kwargs,
        make_step_fn=lambda cfg, lcfg: make_fno_gnn_step(cfg.lambda_loss, lcfg),
        eval_fn=fno_gnn_eval_fn,
        metric_fn=lambda cfg: fno_gnn_metric_fn,
    ),
    'FNO_GNN_Field': ArchEntry(
        cls=FNO_GNN_Field,
        cfg_cls=FNO_GNNConfig,   # mesma config de FNO_GNN — só o forward muda (campo vs delta)
        loader_mode='qtree',
        model_kwargs=_fno_gnn_kwargs,
        make_step_fn=lambda cfg, lcfg: make_fno_gnn_step(cfg.lambda_loss, lcfg),
        eval_fn=fno_gnn_eval_fn,
        metric_fn=lambda cfg: fno_gnn_metric_fn,
    ),
    # FNO_GNN_v2 (2026-07-17): edge_attr [E,5] com delta_mu direcional
    # (mu_origem - mu_destino). Requer dataset gerado com npz_parser='FNO_GNN_v2'
    # (src/data_gen/parsers/fno_gnn_v2.py) — incompatível com chunks de FNO_GNN
    # (edge_attr [E,4]). Forward/step_fn/eval_fn herdados de FNO_GNN sem alteração.
    'FNO_GNN_v2': ArchEntry(
        cls=FNO_GNN_v2,
        cfg_cls=FNO_GNN_v2Config,
        loader_mode='qtree',
        model_kwargs=_fno_gnn_kwargs,
        make_step_fn=lambda cfg, lcfg: make_fno_gnn_step(cfg.lambda_loss, lcfg),
        eval_fn=fno_gnn_eval_fn,
        metric_fn=lambda cfg: fno_gnn_metric_fn,
    ),
    'MaskedFNO2d': ArchEntry(
        cls=MaskedFNO2d,
        cfg_cls=MaskedFNO2dConfig,
        loader_mode='grid',
        model_kwargs=lambda cfg: asdict(cfg),
        make_step_fn=lambda cfg, lcfg: masked_fno_step_fn,
        eval_fn=masked_fno_eval_fn,
        metric_fn=lambda cfg: masked_fno_metric_fn,
    ),
    'FNO2d_SingleMat': ArchEntry(
        cls=FNO2d_SingleMat,
        cfg_cls=SingleMatFNOConfig,
        loader_mode='grid',
        model_kwargs=lambda cfg: asdict(cfg),   # material_id é parâmetro do construtor
        make_step_fn=lambda cfg, lcfg: make_single_mat_step_fn(cfg.material_id),
        eval_fn=single_mat_fno_eval_fn,
        metric_fn=lambda cfg: make_single_mat_metric_fn(cfg.material_id),
    ),
    'MaskedFNO_GNN': ArchEntry(
        cls=MaskedFNO_GNN,
        cfg_cls=MaskedFNO_GNNConfig,
        loader_mode='qtree',
        model_kwargs=lambda cfg: {k: v for k, v in asdict(cfg).items() if k != 'lambda_loss'},
        make_step_fn=lambda cfg, lcfg: make_masked_fno_gnn_step(cfg.lambda_loss),
        eval_fn=masked_fno_gnn_eval_fn,
        metric_fn=lambda cfg: masked_fno_gnn_metric_fn,
    ),
    'GNN_PostBase': ArchEntry(
        cls=GNN_PostBase,
        cfg_cls=GNN_PostBaseConfig,
        loader_mode='qtree',
        model_kwargs=_gnn_post_base_kwargs,
        make_step_fn=lambda cfg, lcfg: make_gnn_post_base_step_fn(lcfg),
        eval_fn=fno_gnn_eval_fn,   # mesma assinatura de forward que FNO_GNN
        metric_fn=lambda cfg: gnn_post_base_metric_fn,
    ),
}

try:
    from src.neural_op.archs.fno_ref import FNORef
    from src.configs.training import FNORefConfig

    def _fno_ref_kwargs(cfg):
        return {k: v for k, v in asdict(cfg).items() if k != 'data_res'}

    ARCH_REGISTRY['FNO_ref'] = ArchEntry(
        cls=FNORef,
        cfg_cls=FNORefConfig,
        loader_mode='grid',
        model_kwargs=_fno_ref_kwargs,
        make_step_fn=lambda cfg, lcfg: fno_step_fn,
        eval_fn=fno_eval_fn,
        metric_fn=lambda cfg: fno_metric_fn,
    )
except ImportError:
    pass
