from dataclasses import dataclass, asdict

from src.neural_op.archs.fno            import FNO2d,          fno_step_fn
from src.neural_op.archs.fno_gnn        import FNO_GNN,        make_fno_gnn_step
from src.neural_op.archs.fno_mat        import (MaskedFNO2d,    masked_fno_step_fn,
                                                FNO2d_SingleMat, make_single_mat_step_fn)
from src.neural_op.archs.masked_fno_gnn import MaskedFNO_GNN,  make_masked_fno_gnn_step
from src.neural_op.archs.eval           import (fno_eval_fn, fno_gnn_eval_fn,
                                                masked_fno_eval_fn, masked_fno_gnn_eval_fn,
                                                single_mat_fno_eval_fn)
from src.configs.training               import (FNOConfig, FNO_GNNConfig,
                                                MaskedFNO2dConfig, MaskedFNO_GNNConfig,
                                                SingleMatFNOConfig)
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
    make_step_fn: callable  # (arch_cfg) -> step_fn(batch, model, loss_fn, device)
    eval_fn     : callable  # (model, chunk_data, eval_cfg) -> None

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


ARCH_REGISTRY: dict = {
    'FNO2d': ArchEntry(
        cls=FNO2d,
        cfg_cls=FNOConfig,
        loader_mode='grid',
        model_kwargs=_fno_kwargs,
        make_step_fn=lambda cfg: fno_step_fn,
        eval_fn=fno_eval_fn,
    ),
    # [REMOVIDO] 'phi_DeepONet' — removido do registry (2026-05-27)
    # 'phi_DeepONet': ArchEntry(
    #     cls=PhiDeepONet, cfg_cls=PhiDeepONetConfig, loader_mode='qtree',
    #     model_kwargs=_phi_deeponet_kwargs,
    #     make_step_fn=lambda cfg: phi_deeponet_step_fn,
    #     eval_fn=phi_deeponet_eval_fn,
    # ),
    'FNO_GNN': ArchEntry(
        cls=FNO_GNN,
        cfg_cls=FNO_GNNConfig,
        loader_mode='qtree',
        model_kwargs=_fno_gnn_kwargs,
        make_step_fn=lambda cfg: make_fno_gnn_step(cfg.lambda_loss),
        eval_fn=fno_gnn_eval_fn,
    ),
    'MaskedFNO2d': ArchEntry(
        cls=MaskedFNO2d,
        cfg_cls=MaskedFNO2dConfig,
        loader_mode='grid',
        model_kwargs=lambda cfg: asdict(cfg),
        make_step_fn=lambda cfg: masked_fno_step_fn,
        eval_fn=masked_fno_eval_fn,
    ),
    'FNO2d_SingleMat': ArchEntry(
        cls=FNO2d_SingleMat,
        cfg_cls=SingleMatFNOConfig,
        loader_mode='grid',
        model_kwargs=lambda cfg: asdict(cfg),   # material_id é parâmetro do construtor
        make_step_fn=lambda cfg: make_single_mat_step_fn(cfg.material_id),
        eval_fn=single_mat_fno_eval_fn,
    ),
    'MaskedFNO_GNN': ArchEntry(
        cls=MaskedFNO_GNN,
        cfg_cls=MaskedFNO_GNNConfig,
        loader_mode='qtree',
        model_kwargs=lambda cfg: {k: v for k, v in asdict(cfg).items() if k != 'lambda_loss'},
        make_step_fn=lambda cfg: make_masked_fno_gnn_step(cfg.lambda_loss),
        eval_fn=masked_fno_gnn_eval_fn,
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
        make_step_fn=lambda cfg: fno_step_fn,
        eval_fn=fno_eval_fn,
    )
except ImportError:
    pass
