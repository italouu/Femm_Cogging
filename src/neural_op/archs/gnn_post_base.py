import json
from pathlib import Path

import torch

from src.neural_op.archs._blocks import GNN
from src.neural_op.archs.fno_gnn import _interpolate_fno_to_nodes

_SUPPORTED_BASE_ARCHS = ('FNO2d', 'FNO_GNN')


def _load_frozen_base(base_run_dir, base_checkpoint, fallback_arch=None, fallback_arch_cfg=None):
    """
    Reconstrói e congela um modelo já treinado (FNO2d ou FNO_GNN) a partir de seu run_dir.

    Se base_run_dir/config.json não existir mais (run movida/apagada), usa
    fallback_arch/fallback_arch_cfg — snapshot gravado em GNN_PostBaseConfig — para
    reconstruir a arquitetura. Se o checkpoint também não existir, o submódulo fica
    com inicialização aleatória; nesse caso os pesos corretos (congelados) vêm do
    load_state_dict feito por quem chama (resume/eval carregando o checkpoint desta
    própria run de GNN_PostBase, que já embute os pesos do base congelado).
    """
    from src.neural_op.archs import ARCH_REGISTRY  # import tardio — evita ciclo com este módulo

    run_dir  = Path(base_run_dir)
    cfg_path = run_dir / 'config.json'

    if cfg_path.exists():
        cfg_dict      = json.loads(cfg_path.read_text())
        arch          = cfg_dict['arch']
        arch_cfg_dict = cfg_dict['arch_cfg']
    elif fallback_arch:
        print(f"  [GNN_PostBase] {cfg_path} não encontrado — usando snapshot de base_arch_cfg")
        arch          = fallback_arch
        arch_cfg_dict = fallback_arch_cfg
    else:
        raise FileNotFoundError(
            f"{cfg_path} não encontrado e nenhum snapshot de fallback (base_arch) disponível"
        )

    if arch not in _SUPPORTED_BASE_ARCHS:
        raise ValueError(
            f"GNN_PostBase só suporta base_arch em {_SUPPORTED_BASE_ARCHS}, "
            f"run '{base_run_dir}' tem arch='{arch}'"
        )

    entry    = ARCH_REGISTRY[arch]
    arch_cfg = entry.cfg_cls(**arch_cfg_dict)
    model    = entry.make_model(arch_cfg)

    ckpt_path = (run_dir / 'model_final.pth') if base_checkpoint == 'final' \
        else (run_dir / 'checkpoints' / f'{base_checkpoint}.pth')
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd   = {k: v for k, v in ckpt['model_state_dict'].items() if k != '_metadata'}
        model.load_state_dict(sd)
    else:
        print(f"  [GNN_PostBase] {ckpt_path} não encontrado — base inicializado aleatoriamente; "
              f"pesos corretos devem vir do load_state_dict desta run de GNN_PostBase")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return arch, model


class GNN_PostBase(torch.nn.Module):
    """
    GNN treinado isoladamente sobre a saída de um FNO2d ou FNO_GNN já treinado e congelado.

    Stage 2 de um pipeline não end-to-end: o modelo base não recebe gradiente;
    apenas o GNN novo é treinado.

    Pipeline
    --------
    base_arch == 'FNO2d':
        x_hw → FNO2d (frozen) → y_hw_base
                    ↓ _interpolate_fno_to_nodes
        base_at_nodes

    base_arch == 'FNO_GNN':
        (x_hw, node_x, edge_index, edge_attr, L) → FNO_GNN (frozen) → y_hw_base, base_at_nodes
        (base_at_nodes já é a saída pós-correção do GNN interno do FNO_GNN)

    Em ambos os casos:
        [node_x | base_at_nodes] → GNN (treinável) → Δ
        y_nodes = base_at_nodes + Δ
    """

    _NODE_IN_CH  = 5
    _EDGE_DIM    = 4
    _BASE_OUT_CH = 2
    _GNN_IN_CH   = _NODE_IN_CH + _BASE_OUT_CH

    def __init__(self, base_run_dir, base_checkpoint, gnn_node_width, gnn_n_layers,
                 base_arch=None, base_arch_cfg=None):
        super().__init__()
        self.base_arch, self.base_model = _load_frozen_base(
            base_run_dir, base_checkpoint,
            fallback_arch=base_arch, fallback_arch_cfg=base_arch_cfg,
        )
        self.gnn = GNN(
            in_node_features=self._GNN_IN_CH,
            out_node_features=self._BASE_OUT_CH,
            edge_dim=self._EDGE_DIM,
            node_width=gnn_node_width,
            n_layers=gnn_n_layers,
        )

    def _base_pred_at_nodes(self, x_hw, node_x, edge_index, edge_attr, L):
        with torch.no_grad():
            if self.base_arch == 'FNO2d':
                y_hw_base    = self.base_model(x_hw)
                base_at_nodes = _interpolate_fno_to_nodes(y_hw_base, node_x, L)
            else:  # 'FNO_GNN'
                y_hw_base, base_at_nodes = self.base_model(x_hw, node_x, edge_index, edge_attr, L)
        return y_hw_base, base_at_nodes

    def forward(self, x_hw, node_x, edge_index, edge_attr, L):
        y_hw_base, base_at_nodes = self._base_pred_at_nodes(x_hw, node_x, edge_index, edge_attr, L)
        gnn_input = torch.cat([node_x, base_at_nodes], dim=-1)
        delta     = self.gnn(gnn_input, edge_index, edge_attr)
        return y_hw_base, base_at_nodes + delta


def gnn_post_base_step_fn(batch, model, loss_fn, device):
    x_hw       = batch['x_hw'].to(device)
    node_x     = batch['node_x'].to(device)
    edge_index = batch['edge_index'].to(device)
    edge_attr  = batch['edge_attr'].to(device)
    L          = batch['L'].to(device)
    y_node     = batch['node_y'][:, :2].to(device)
    _, y_nodes = model(x_hw, node_x, edge_index, edge_attr, L)
    return loss_fn(y_nodes, y_node)
