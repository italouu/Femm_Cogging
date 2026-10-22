from dataclasses import dataclass, field, fields, MISSING
from typing import Any, Optional

from src.configs.monitor import MonitorCfg
from src.configs.loss    import MaeLossCfg, LOSS_CFG_REGISTRY


def _detect_chunk_dims(dataset: str) -> dict:
    """
    Lê shapes reais do primeiro chunk de data/torch/data_chunks/{dataset}/.

    Fonte de verdade para in/out channels (x_hw/y_hw), features de nó
    (node_x/node_y) e de aresta (edge_attr) de todos os arch_cfgs com
    dimensões dependentes do dataset (FNOConfig, FNO_GNNConfig, GNN_PostBaseConfig,
    FNO_BipartiteGNN*Config) — evita hardcode quando o layout do chunk muda
    (ex: canal J em 2026-10-14). Chamado por NnCfg.__post_init__.
    """
    import glob
    import torch

    chunk_paths = sorted(glob.glob(f'data/torch/data_chunks/{dataset}/data_chunk_*.pt'))
    if not chunk_paths:
        raise FileNotFoundError(
            f"_detect_chunk_dims: nenhum chunk encontrado em "
            f"'data/torch/data_chunks/{dataset}/' — não foi possível detectar dimensões."
        )
    sample = torch.load(chunk_paths[0], map_location='cpu', weights_only=False)

    dims = {
        'x_hw_ch': sample['x_hw'].shape[1],
        'y_hw_ch': sample['y_hw'].shape[1],
        'grid_hw': tuple(sample['x_hw'].shape[-2:]),   # B3 — checagem de data_res
    }
    if 'node_x' in sample:
        dims['node_x_ch'] = sample['node_x'].shape[1]
        dims['node_y_ch'] = sample['node_y'].shape[1]
        if dims['node_y_ch'] != dims['y_hw_ch']:
            raise ValueError(
                f"_detect_chunk_dims: node_y tem {dims['node_y_ch']} coluna(s) mas "
                f"y_hw tem {dims['y_hw_ch']} canal(is) em '{dataset}' — parser "
                f"inconsistente (esperado o mesmo número dos dois lados)."
            )
    if 'edge_attr' in sample:
        dims['edge_dim'] = sample['edge_attr'].shape[-1]
    if 'elem_x' in sample:
        dims['elem_in_ch'] = sample['elem_x'].shape[1]
    if 'cross_edge_attr' in sample:
        dims['cross_edge_dim'] = sample['cross_edge_attr'].shape[-1]
    return dims


def _from_dict_generic(cls, d: dict):
    """
    Reconstrói uma dataclass com campos init=False (ex: edge_dim, base_arch) a
    partir de um dict já salvo (config.json), sem passar por __init__/__post_init__
    — usado por postprocess/inference.load_model (via hasattr(cfg_cls, 'from_dict')) para reconstruir
    arch_cfg sem depender de recursos externos (dataset, base_run_dir) ainda
    existirem no mesmo estado do treino.

    Campo ausente em d (ex: 'edge_dim' em config.json de runs salvas antes desse
    campo existir) cai no default da dataclass, em vez de KeyError — mesmo
    comportamento que cls(**d) teria se o campo fosse init=True.
    """
    obj = cls.__new__(cls)
    for f in fields(cls):
        if f.name in d:
            setattr(obj, f.name, d[f.name])
        else:
            default = f.default if f.default is not MISSING else f.default_factory()
            setattr(obj, f.name, default)
    return obj


# ── Arquiteturas ──────────────────────────────────────────────────────────────

@dataclass
class FNOConfig:
    # in_channels/out_channels: auto-detectados a partir de x_hw/y_hw do dataset
    # em NnCfg.__post_init__ (helper _detect_chunk_dims) — igual edge_dim em
    # FNO_GNNConfig. init=False pra impedir mismatch manual (ex: setar
    # out_channels diferente do nº de canais de y_hw do dataset).
    in_channels: int = field(default=2, init=False)
    out_channels: int = field(default=2, init=False)
    # Defaults = run fno2d_mae da bateria bldc (src/configs/batteries/bldc.py), conferidos campo a campo.
    modes1: int = 69
    modes2: int = 139
    conv_width: int = 8

    conv_layers: int = 4
    lift_width: int = 64
    lift_layers: int = 3
    proj_width: int = 64
    proj_layers: int = 3
    data_res: tuple = (138, 276)

    @classmethod
    def from_dict(cls, d: dict):
        # in_channels/out_channels são init=False — postprocess/inference.load_model detecta este
        # método (hasattr) e o usa em vez de cfg_cls(**arch_cfg_dict), que quebraria
        # por esses campos não serem kwargs do construtor. Mesmo padrão de
        # FNO_GNNConfig.from_dict (edge_dim).
        return _from_dict_generic(cls, d)


@dataclass
class FNO_GNNConfig:
    # Defaults = run fno_gnn_mae da bateria bldc (src/configs/batteries/bldc.py), conferidos campo a campo.
    fno_modes1: int = 69
    fno_modes2: int = 139
    fno_conv_width: int = 6
    fno_conv_layers: int = 4
    fno_lift_width: int = 64
    fno_lift_layers: int = 3
    fno_proj_width: int = 64
    fno_proj_layers: int = 3
    data_res: tuple = (138, 276)
    gnn_node_width: int = 32
    gnn_n_layers: int = 3

    # edge_dim / grid_in_ch / grid_out_ch / node_in_ch — todos detectados
    # automaticamente em NnCfg.__post_init__ (helper _detect_chunk_dims) a
    # partir dos chunks reais do dataset de treino (self.dataset): edge_dim de
    # edge_attr, grid_in_ch de x_hw, grid_out_ch de y_hw, node_in_ch de node_x.
    # Defaults abaixo (4/2/2/5) só são usados se este arch_cfg for reconstruído
    # fora de NnCfg (ex: postprocess/inference.load_model faz cfg_cls.from_dict(arch_cfg_dict)
    # direto do config.json salvo — nesse caso os valores gravados na própria
    # run são usados, não o default).
    edge_dim:     int = field(default=4, init=False)
    grid_in_ch:   int = field(default=2, init=False)
    grid_out_ch:  int = field(default=2, init=False)
    node_in_ch:   int = field(default=5, init=False)

    @classmethod
    def from_dict(cls, d: dict):
        # edge_dim é init=False — postprocess/inference.load_model detecta este método (hasattr) e o usa
        # em vez de cfg_cls(**arch_cfg_dict), que quebraria por edge_dim não ser kwarg
        # do construtor.
        return _from_dict_generic(cls, d)


@dataclass
class FNO_BipartiteGNNConfig:
    """Config de FNO_BipartiteGNN (src/neural_op/archs/fno_bipartite_gnn.py) --
    grafo duplo vértices (iterado) + elementos (estático, injetado via
    arestas cruzadas), mode='bipartite' (ver src/data_gen/parsers/bipartite_structure.py
    e CLAUDE.md/conversa 2026-08-10 sobre a arquitetura)."""
    # Defaults = run bipartite_mae da bateria bldc (src/configs/batteries/bldc.py), conferidos campo a campo.
    fno_modes1: int = 69
    fno_modes2: int = 139
    fno_conv_width: int = 6
    fno_conv_layers: int = 4
    fno_lift_width: int = 64
    fno_lift_layers: int = 3
    fno_proj_width: int = 64
    fno_proj_layers: int = 3
    data_res: tuple = (138, 276)
    gnn_node_width: int = 32
    gnn_n_layers: int = 3

    # edge_dim/grid_in_ch/grid_out_ch/node_in_ch/elem_in_ch/cross_edge_dim --
    # todos auto-detectados em NnCfg.__post_init__ (helper _detect_chunk_dims)
    # a partir dos chunks reais do dataset — ver FNO_GNNConfig para o mesmo
    # padrão. Defaults abaixo batem com o layout fixo de
    # src/data_gen/parsers/bipartite_structure.py (node_x=[r,c], edge_attr=3 colunas,
    # elem_x=5 colunas, cross_edge_attr=1 coluna) — só usados se este
    # arch_cfg for reconstruído fora de NnCfg (ex: postprocess/inference.load_model).
    edge_dim:       int = field(default=3, init=False)
    grid_in_ch:     int = field(default=2, init=False)
    grid_out_ch:    int = field(default=1, init=False)
    node_in_ch:     int = field(default=2, init=False)
    elem_in_ch:     int = field(default=5, init=False)
    cross_edge_dim: int = field(default=1, init=False)

    @classmethod
    def from_dict(cls, d: dict):
        return _from_dict_generic(cls, d)


@dataclass
class FNO_BipartiteGNN_ElemConfig:
    """Config de FNO_BipartiteGNN_Elem (src/neural_op/archs/fno_bipartite_gnn_elem.py)
    -- mesma rede da FNO_BipartiteGNN com os papéis trocados no layout dos dados:
    elementos = grafo principal (saída (Br, Bθ) por elemento), vértices = auxiliar
    estático. Dataset: BLDC_138x276/FNO_BipartiteGNN_Elem
    (scripts/build_datasets.py). Mesmos campos de FNO_BipartiteGNNConfig."""
    # Defaults = run elem_mae da bateria bldc (src/configs/batteries/bldc.py), conferidos campo a campo.
    fno_modes1: int = 69
    fno_modes2: int = 139
    fno_conv_width: int = 6
    fno_conv_layers: int = 4
    fno_lift_width: int = 64
    fno_lift_layers: int = 3
    fno_proj_width: int = 64
    fno_proj_layers: int = 3
    data_res: tuple = (138, 276)
    gnn_node_width: int = 32
    gnn_n_layers: int = 3

    # auto-detectados em NnCfg.__post_init__; defaults = layout de
    # src/data_gen/parsers/elem_structure.py (node_x de elemento com 5 colunas,
    # elem_x de vértice com 2)
    edge_dim:       int = field(default=3, init=False)
    grid_in_ch:     int = field(default=2, init=False)
    grid_out_ch:    int = field(default=2, init=False)
    node_in_ch:     int = field(default=5, init=False)
    elem_in_ch:     int = field(default=2, init=False)
    cross_edge_dim: int = field(default=1, init=False)

    @classmethod
    def from_dict(cls, d: dict):
        return _from_dict_generic(cls, d)


@dataclass
class GNN_PostBaseConfig:
    # Treino em duas etapas (não end-to-end): base_run_dir aponta para um run FNO2d já
    # treinado, congelado; só o GNN novo é treinado.
    # Defaults = run postbase_mae da bateria bldc; nela a base é runs['fno2d_mae'] (resolvida
    # pelo runner) -- aqui, o FNO2d mae local de BLDC_138x276 (alvo polar Br, Bθ, run de
    # 2026-10-19; trocar pela run nova depois que a bateria for refeita).
    base_run_dir   : str = 'data/logs/BLDC_138x276/FNO2d/run_0002'
    base_checkpoint: str = 'best'   # 'best', 'latest' ou 'final'
    gnn_node_width : int = 64
    gnn_n_layers   : int = 6

    # Snapshot de arch/arch_cfg/epoch de base_run_dir, capturado em __post_init__ e
    # gravado em config.json desta run (via NnCfg → ModelManager.open). Garante que os
    # parâmetros do modelo base não se percam mesmo que base_run_dir seja movido/apagado
    # depois — usado como fallback em GNN_PostBase._load_frozen_base.
    base_arch     : str           = field(default='', init=False)
    base_arch_cfg : dict          = field(default_factory=dict, init=False)
    base_epoch    : Optional[int] = field(default=None, init=False)

    # edge_dim / node_in_ch do GNN interno — auto-detectados em NnCfg.__post_init__
    # (helper _detect_chunk_dims), a partir do dataset de treino (self.dataset do
    # NnCfg, não de base_run_dir) — ver FNO_GNNConfig para detalhes.
    edge_dim:   int = field(default=4, init=False)
    node_in_ch: int = field(default=5, init=False)

    # base_out_ch: número de canais de saída do modelo base congelado (Br, Bθ)
    # — lido do PRÓPRIO base_run_dir/config.json (FNO2d grava em 'out_channels'),
    # não do dataset desta run nova. Default 2 cobre configs salvos antes desse
    # campo existir.
    base_out_ch: int = field(default=2, init=False)

    # base_normalize/base_norm_stats: snapshot da normalização (ver
    # src/neural_op/normalization.py) usada no TREINO do modelo base congelado
    # — lido do PRÓPRIO base_run_dir/config.json, não recalculado. Necessário
    # porque o base pode ter sido treinado num dataset diferente do desta run
    # nova (self.dataset), com stats diferentes; GNN_PostBase.forward faz um
    # round-trip decode(normalizer)/encode(base_normalizer) só ao redor da
    # chamada ao modelo base, para ele sempre receber x_hw na escala exata que
    # aprendeu. Default False/{} cobre runs base salvas antes dessa feature.
    base_normalize:  bool = field(default=False, init=False)
    base_norm_stats: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        import json
        from pathlib import Path
        import torch

        run_dir  = Path(self.base_run_dir)
        cfg_dict = json.loads((run_dir / 'config.json').read_text())
        self.base_arch     = cfg_dict['arch']
        self.base_arch_cfg = cfg_dict['arch_cfg']
        self.base_out_ch   = self.base_arch_cfg.get('out_channels') or 2
        self.base_normalize  = cfg_dict.get('normalize', False)
        self.base_norm_stats = cfg_dict.get('norm_stats', {})

        ckpt_path = (run_dir / 'model_final.pth') if self.base_checkpoint == 'final' \
            else (run_dir / 'checkpoints' / f'{self.base_checkpoint}.pth')
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            self.base_epoch = ckpt.get('epoch')

    @classmethod
    def from_dict(cls, d: dict):
        # Reconstrói a partir de um config.json já salvo (eval/resume) sem rechamar
        # __post_init__ — evita depender de base_run_dir ainda existir; usa o snapshot
        # (base_arch/base_arch_cfg/base_epoch/edge_dim) já gravado na própria run.
        return _from_dict_generic(cls, d)


# ── Config principal ──────────────────────────────────────────────────────────
#
# Referência rápida — arch e loss disponíveis
# ┌───────────────────────┬──────────────┬─────────────────────────────┐
# │ arch                  │ loader_mode  │ arch_cfg                    │
# ├───────────────────────┼──────────────┼─────────────────────────────┤
# │ FNO2d                 │ grid         │ FNOConfig                   │
# │ FNO_GNN               │ vertex       │ FNO_GNNConfig               │
# │ GNN_PostBase          │ vertex       │ GNN_PostBaseConfig          │
# │                       │              │   (base FNO2d congelada)    │
# │ FNO_BipartiteGNN      │ bipartite    │ FNO_BipartiteGNNConfig      │
# │ FNO_BipartiteGNN_Elem │ bipartite    │ FNO_BipartiteGNN_ElemConfig │
# └───────────────────────┴──────────────┴─────────────────────────────┘
# Losses — LOSS_REGISTRY (src/neural_op/losses.py) / LOSS_CFG_REGISTRY (src/configs/loss.py):
# 'mse' (MseLossCfg) e 'mae' (MaeLossCfg), sem campos, sobre a saída final do modelo.

@dataclass
class NnCfg:
    # Defaults = run elem_mae da bateria bldc (src/configs/batteries/bldc.py), campo a campo.
    # arch/loss são derivados do TIPO de arch_cfg/loss_cfg em __post_init__ (init=False) e
    # continuam no config.json (asdict) com o mesmo nome.
    dataset: str = 'BLDC_138x276/FNO_BipartiteGNN_Elem'
    arch: str = field(default='', init=False)   # derivado de type(arch_cfg) — ARCH_REGISTRY
    loss: str = field(default='', init=False)   # derivado de type(loss_cfg) — LOSS_CFG_REGISTRY

    problem: str = 'BLDC_138x276'

    # Treino
    lr: float = 0.01
    n_epochs: int = 500
    scheduler: str = 'step'   # só 'step' (StepLR) é implementado em scripts/train.py::run
    scheduler_step: int = 100
    scheduler_gamma: float = 0.6

    # Dataloader
    batch_size: int = 32
    train_split: float = 0.70
    test_split: Optional[float] = None  # None → complemento de train_split (1 - train_split);
                                         # valor explícito → fração fixa de chunks para teste,
                                         # útil para agilizar épocas com train_split pequeno
    buffer_size: int = 64
    num_workers: int = 2
    prefetch_factor: int = 2
    split_seed: int = 12

    # B5 (2026-10-03) — índice da repetição (sem controle de semente) na bateria;
    # só rótulo gravado no config.json. Gravado pelo runner (scripts/run_battery.py,
    # BatteryCfg.n_repeats) — nas definições de bateria fica no default 0.
    # Configs antigos sem o campo = repetição 0.
    repeat: int = 0

    # Resume
    resume_run:        Optional[str] = None     # caminho para run_XXXX/ existente
    resume_checkpoint: str           = 'latest' # 'latest' ou 'best'
    resume_modified:   bool          = False    # True = carrega só pesos; optimizer/scheduler do zero

    # O TIPO de arch_cfg/loss_cfg define a arch/loss (ver arch/loss acima).
    arch_cfg:     Any        = field(default_factory=FNO_BipartiteGNN_ElemConfig)
    monitor_cfg:  MonitorCfg = field(default_factory=MonitorCfg)
    loss_cfg:     Any        = field(default_factory=MaeLossCfg)

    # Normalização (src/neural_op/normalization.py) — z-score por canal de
    # x_hw/y_hw/node_x/node_y, aplicado de forma transparente no dataloader
    # (loss calculada em espaço normalizado — corrige o caso de alvos com
    # escala física pequena, ex: A/potencial vetor, onde MSE bruta cai perto
    # de zero sem a rede aprender a estrutura espacial real). norm_stats é
    # sempre recalculado (ou lido do cache em disco) a partir de self.dataset
    # em __post_init__ — nunca deixado no default, mesmo padrão de
    # _detect_chunk_dims. GNN_PostBase é exceção: usa suas próprias stats
    # (fit sobre self.dataset) para o GNN treinável, e as stats do
    # base_run_dir (arch_cfg.base_norm_stats) só para a chamada ao modelo
    # base congelado — ver GNN_PostBase.forward.
    normalize:  bool = True
    norm_stats: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        from pathlib import Path
        from src.neural_op.archs import ARCH_REGISTRY
        from src.neural_op.losses import LOSS_REGISTRY
        # Tipo EXATO (não isinstance) -- cada arch tem a sua dataclass de config.
        archs = [n for n, e in ARCH_REGISTRY.items() if e.cfg_cls is type(self.arch_cfg)]
        if len(archs) != 1:
            raise TypeError(
                f"arch_cfg do tipo {type(self.arch_cfg).__name__} corresponde a {archs or 'nenhuma'} "
                f"arch(s) no ARCH_REGISTRY — precisa ser exatamente uma")
        self.arch = archs[0]
        losses = [n for n, c in LOSS_CFG_REGISTRY.items() if c is type(self.loss_cfg)]
        if len(losses) != 1 or losses[0] not in LOSS_REGISTRY:
            raise TypeError(
                f"loss_cfg do tipo {type(self.loss_cfg).__name__} corresponde a {losses or 'nenhuma'} "
                f"loss(es) em LOSS_CFG_REGISTRY/LOSS_REGISTRY — precisa ser exatamente uma")
        self.loss = losses[0]
        if self.scheduler != 'step':
            raise ValueError(f"scheduler={self.scheduler!r} não implementado — só 'step' (StepLR)")
        _dim_fields = ('grid_in_ch', 'grid_out_ch', 'node_in_ch', 'edge_dim',
                       'elem_in_ch', 'cross_edge_dim')
        if isinstance(self.arch_cfg, FNOConfig) or any(hasattr(self.arch_cfg, f) for f in _dim_fields):
            dims = _detect_chunk_dims(self.dataset)
            if isinstance(self.arch_cfg, FNOConfig):
                # FNO2d "puro" (mode='grid') — campos com nomes próprios (não
                # grid_in_ch/grid_out_ch) porque são passados direto ao construtor
                # de FNO2d via asdict(cfg), que espera in_channels/out_channels.
                self.arch_cfg.in_channels  = dims['x_hw_ch']
                self.arch_cfg.out_channels = dims['y_hw_ch']
            if hasattr(self.arch_cfg, 'grid_in_ch'):
                self.arch_cfg.grid_in_ch = dims['x_hw_ch']
            if hasattr(self.arch_cfg, 'grid_out_ch'):
                self.arch_cfg.grid_out_ch = dims['y_hw_ch']
            if hasattr(self.arch_cfg, 'node_in_ch'):
                self.arch_cfg.node_in_ch = dims['node_x_ch']
            if hasattr(self.arch_cfg, 'edge_dim'):
                self.arch_cfg.edge_dim = dims['edge_dim']
            if hasattr(self.arch_cfg, 'elem_in_ch'):
                self.arch_cfg.elem_in_ch = dims['elem_in_ch']
            if hasattr(self.arch_cfg, 'cross_edge_dim'):
                self.arch_cfg.cross_edge_dim = dims['cross_edge_dim']
            # B3 (2026-10-03) — data_res só limita modes1/modes2 em FNO2d; se
            # diferir da grade real o espectro pode ficar sobreposto sem aviso
            # (ex: (135,270) numa grade 138×276). Só avisa, não corrige.
            data_res = getattr(self.arch_cfg, 'data_res', None)
            if data_res is not None and tuple(data_res) != dims['grid_hw']:
                import warnings
                warnings.warn(
                    f"NnCfg: arch_cfg.data_res={tuple(data_res)} difere da grade real "
                    f"{dims['grid_hw']} de '{self.dataset}'", stacklevel=2)
        if self.normalize:
            from src.neural_op.normalization import Normalizer
            self.norm_stats = Normalizer.fit(self.dataset, self.arch, self.train_split,
                                             self.split_seed, self.test_split).to_dict()
        if self.resume_run is not None and not Path(self.resume_run).exists():
            resolved = Path('data/logs') / self.problem / self.arch / self.resume_run
            self.resume_run = str(resolved)