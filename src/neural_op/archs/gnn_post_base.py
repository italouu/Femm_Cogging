import json
from pathlib import Path

import torch

from src.neural_op.archs._blocks import GNN
from src.neural_op.archs.fno_gnn import _interpolate_fno_to_nodes

# [REMOVIDO] _SUPPORTED_BASE_ARCHS = ('FNO2d', 'FNO_GNN') — faltava 'FNO_GNN_v2';
# FNO_GNN_v2 é subclasse de FNO_GNN com mesma assinatura de forward (ver
# _base_pred_at_nodes abaixo, branch 'else'), então só faltava liberar aqui.
_SUPPORTED_BASE_ARCHS = ('FNO2d', 'FNO_GNN', 'FNO_GNN_v2')


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

    entry = ARCH_REGISTRY[arch]
    # [REMOVIDO] arch_cfg = entry.cfg_cls(**arch_cfg_dict) — quebrava para arch_cfgs
    # com campo init=False (ex: edge_dim em FNO_GNNConfig/FNO_GNN_v2Config): asdict()
    # grava 'edge_dim' no config.json, mas __init__ não aceita esse kwarg. Mesmo
    # padrão de scripts/eval.py:26-31 (hasattr from_dict) para reconstruir sem
    # rechamar __post_init__/__init__.
    if hasattr(entry.cfg_cls, 'from_dict'):
        arch_cfg = entry.cfg_cls.from_dict(arch_cfg_dict)
    else:
        arch_cfg = entry.cfg_cls(**arch_cfg_dict)
    model = entry.make_model(arch_cfg)

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

    # [REMOVIDO] _NODE_IN_CH/_BASE_OUT_CH/_GNN_IN_CH constantes de classe
    # hardcoded (5/2/7) — mesmo motivo de FNO_GNN: quebravam com datasets/bases
    # de canal diferente (ex: base treinada em A, 1 canal, em vez de B).
    # Viraram parâmetros do construtor (node_in_ch/base_out_ch), auto-
    # detectados em GNN_PostBaseConfig (node_in_ch a partir do dataset desta
    # run, base_out_ch a partir do config.json de base_run_dir).
    #
    # _NODE_IN_CH  = 5
    # _BASE_OUT_CH = 2
    # _GNN_IN_CH   = _NODE_IN_CH + _BASE_OUT_CH

    def __init__(self, base_run_dir, base_checkpoint, gnn_node_width, gnn_n_layers,
                 edge_dim, node_in_ch, base_out_ch, base_arch=None, base_arch_cfg=None,
                 base_normalize=False, base_norm_stats=None):
        super().__init__()
        self.base_arch, self.base_model = _load_frozen_base(
            base_run_dir, base_checkpoint,
            fallback_arch=base_arch, fallback_arch_cfg=base_arch_cfg,
        )
        # Normalização (src/neural_op/normalization.py) usada no TREINO do
        # modelo base — pode diferir da normalização desta run (self.normalizer,
        # atribuído externamente após a construção — ver scripts/train.py/eval.py).
        # _base_pred_at_nodes faz o round-trip decode(normalizer)/encode(base
        # normalizer) só ao redor da chamada a self.base_model, pra ele sempre
        # receber x_hw na escala exata que aprendeu.
        from src.neural_op.normalization import Normalizer
        self._base_normalizer = (
            Normalizer.from_dict(base_norm_stats) if base_normalize and base_norm_stats else None
        )
        self.normalizer = None   # atribuído externamente (ver scripts/train.py/eval.py)
        self.gnn = GNN(
            in_node_features=node_in_ch + base_out_ch,
            out_node_features=base_out_ch,
            edge_dim=edge_dim,
            node_width=gnn_node_width,
            n_layers=gnn_n_layers,
        )

    def _base_pred_at_nodes(self, x_hw, node_x, edge_index, edge_attr, L):
        # Round-trip pra escala do modelo base: x_hw chega normalizado pelas
        # stats DESTA run (self.normalizer, se houver) — decodifica pra bruto e
        # reencoda pelas stats que o base aprendeu (self._base_normalizer, se
        # houver) antes de chamá-lo; o inverso na saída, de volta pra escala
        # desta run (usada na concatenação com node_x pro GNN treinável).
        # Sem normalização em nenhum dos dois lados: no-op, comportamento
        # idêntico ao anterior a essa mudança.
        normalizer = self.normalizer
        base_norm  = self._base_normalizer

        x_hw_base = normalizer.decode(x_hw, 'x_hw') if normalizer is not None else x_hw
        if base_norm is not None:
            x_hw_base = base_norm.encode(x_hw_base, 'x_hw')

        with torch.no_grad():
            if self.base_arch == 'FNO2d':
                y_hw_base     = self.base_model(x_hw_base)
                base_at_nodes = _interpolate_fno_to_nodes(y_hw_base, node_x, L)
                base_key      = 'y_hw'
            else:  # 'FNO_GNN' / 'FNO_GNN_v2'
                y_hw_base, base_at_nodes = self.base_model(x_hw_base, node_x, edge_index, edge_attr, L)
                base_key      = 'node_y'

        if base_norm is not None:
            y_hw_base     = base_norm.decode(y_hw_base, 'y_hw')
            base_at_nodes = base_norm.decode(base_at_nodes, base_key)
        if normalizer is not None:
            y_hw_base     = normalizer.encode(y_hw_base, 'y_hw')
            base_at_nodes = normalizer.encode(base_at_nodes, 'node_y')
        return y_hw_base, base_at_nodes

    def forward(self, x_hw, node_x, edge_index, edge_attr, L, return_components=False):
        y_hw_base, base_at_nodes = self._base_pred_at_nodes(x_hw, node_x, edge_index, edge_attr, L)
        gnn_input = torch.cat([node_x, base_at_nodes], dim=-1)
        delta     = self.gnn(gnn_input, edge_index, edge_attr)
        if return_components:
            return y_hw_base, base_at_nodes, delta
        return y_hw_base, base_at_nodes + delta


def make_gnn_post_base_step_fn(loss_cfg=None):
    """Retorna step_fn fechada sobre loss_cfg."""
    subtract = getattr(loss_cfg, 'subtract_fno', False)

    def gnn_post_base_step(batch, model, loss_fn, device):
        x_hw       = batch['x_hw'].to(device)
        node_x     = batch['node_x'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_attr  = batch['edge_attr'].to(device)
        L          = batch['L'].to(device)
        y_node     = batch['node_y'].to(device)   # [REMOVIDO] [:, :2] — ver fno_gnn_step
        if subtract:
            _, base_at_nodes, delta = model(x_hw, node_x, edge_index, edge_attr, L,
                                             return_components=True)
            # [REMOVIDO] return loss_fn(delta, y_node) — comparava a correção bruta
            # do GNN contra o alvo inteiro, não contra o resíduo real da baseline
            # congelada. Corrigido para residual learning de fato: delta deve
            # aprender (y_node - base_at_nodes), não y_node diretamente.
            return loss_fn(delta, y_node - base_at_nodes)
        else:
            _, y_nodes = model(x_hw, node_x, edge_index, edge_attr, L)
            return loss_fn(y_nodes, y_node)

    return gnn_post_base_step


def gnn_post_base_metric_fn(batch, model, device):
    """
    MAE bruto: mae_hw compara a saída em grade H×W do modelo base congelado
    (FNO2d ou estágio FNO do FNO_GNN) contra y_hw; mae_graph compara a saída
    final do GNN_PostBase (base + delta) nos nós contra node_y.

    batch já chega normalizado (CUDAPrefetcher.encode_batch, se normalize=True)
    — decodifica pred/y de volta pra unidade física (Tesla) antes do MAE, pra
    manter o significado documentado de mae_hw/mae_graph em metrics.jsonl.
    """
    normalizer = getattr(model, 'normalizer', None)
    x_hw       = batch['x_hw'].to(device)
    node_x     = batch['node_x'].to(device)
    edge_index = batch['edge_index'].to(device)
    edge_attr  = batch['edge_attr'].to(device)
    L          = batch['L'].to(device)
    y_hw       = batch['y_hw'].to(device)
    y_node     = batch['node_y'].to(device)   # [REMOVIDO] [:, :2] — ver fno_gnn_step
    with torch.no_grad():
        y_hw_base, y_nodes = model(x_hw, node_x, edge_index, edge_attr, L)
        if normalizer is not None:
            y_hw_base = normalizer.decode(y_hw_base, 'y_hw')
            y_nodes   = normalizer.decode(y_nodes,   'node_y')
            y_hw      = normalizer.decode(y_hw,      'y_hw')
            y_node    = normalizer.decode(y_node,    'node_y')
        mae_hw    = torch.mean(torch.abs(y_hw_base - y_hw)).item()
        mae_graph = torch.mean(torch.abs(y_nodes   - y_node)).item()
    return mae_hw, mae_graph


# [REMOVIDO] gnn_post_base_step_fn plain function — substituída por make_gnn_post_base_step_fn
# para aceitar loss_cfg (subtract_fno). Retrocompatibilidade via fábrica com loss_cfg=None
# (comportamento idêntico ao anterior).
# def gnn_post_base_step_fn(batch, model, loss_fn, device):
#     x_hw       = batch['x_hw'].to(device)
#     node_x     = batch['node_x'].to(device)
#     edge_index = batch['edge_index'].to(device)
#     edge_attr  = batch['edge_attr'].to(device)
#     L          = batch['L'].to(device)
#     y_node     = batch['node_y'][:, :2].to(device)
#     _, y_nodes = model(x_hw, node_x, edge_index, edge_attr, L)
#     return loss_fn(y_nodes, y_node)
gnn_post_base_step_fn = make_gnn_post_base_step_fn()
