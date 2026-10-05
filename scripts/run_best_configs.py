"""
Roda uma sequência de treinos: para cada arquitetura (FNO2d, FNO_GNN,
GNN_PostBase, FNO_BipartiteGNN — v3 e graph_div_b_loss ficam de fora a pedido
do usuário), reconstrói os hiperparâmetros do melhor run já encontrado (ver
levantamento em CLAUDE.md / conversa) e roda duas vezes: loss='mse' e loss='mae'.
Sem baseline mse/mae pra FNO_BipartiteGNN até agora — graph_div_b_loss nunca foi
comparado contra as losses padrão nesse dataset, essa sequência cobre essa lacuna
(mesmo motivo pelo qual v3 fica de fora: ela já tinha mse/mae, faltava era
FNO_BipartiteGNN "puro").

Total: 4 arqs × 2 losses = 8 treinos, sequenciais (um processo, uma GPU).
Uma falha em um treino não aborta os demais — status/erro de cada um é
reportado no resumo final.

2026-10-01 — adaptado para os datasets unificados (ver CLAUDE.md "Datasets
unificados a partir do raw mesh_ans_138x276"): as 4 archs passam a treinar em
data/torch/data_chunks/mesh_ans_138x276_unified/<arch>/ (mesmo raw, mesmo
gabarito B, mesmo split), gerados por scripts/build_unified_ans_chunks_direct.py.
Logs em data/logs/mesh_ans_138x276_unified_best_mse_mae/<arch>/run_XXXX/.
GNN_PostBase usa como base o FNO2d treinado NESTA bateria com a MESMA loss do
próprio GNN_PostBase (mse -> base FNO2d mse, mae -> base FNO2d mae) — a base
antiga (mesh_138x276_FEMM_MESH/FNO2d/run_0001) foi treinada contra o B
suavizado do FEMM (gabarito diferente). Por
isso FNO2d precisa vir antes de GNN_PostBase em BEST_CONFIGS, e o
GNN_PostBaseConfig é construído só na hora do treino (__post_init__ lê o
config.json da base, que ainda não existe na importação do módulo).

2026-10-03 — bateria definitiva (pré-artigo). Correções ligadas por chaves no
topo do módulo, cada uma com o comportamento antigo disponível:
  B1 INTERP_MODE (interpolação grade→nós centrada em células + wrap angular),
  B2 FNO_NODE_RESCALE (FNO@nós recodificado y_hw→node_y antes do resíduo),
  B3 FULL_SPECTRUM (data_res=138×276, modes=(69,139), sem sobreposição),
  B4 METRICS_EVERY_EPOCH (epochs.csv/run_summary.json — ver ModelManager),
  B5 N_REPEATS (repetições sem controle de semente; GNN_PostBase pareado com
     o FNO2d da mesma loss E repetição — NnCfg.repeat), [DECISÃO PENDENTE]
  B6 INCLUDE_REL_L2 (loss relative_l2 nas 4 archs), [DECISÃO PENDENTE].
Antes de rodar: arquivar os logs da bateria anterior (B0,
scripts/archive_old_battery_logs.py) — senão _find_base_run_dir pode achar o
FNO2d antigo. Roteiro completo: docs/bateria_definitiva/ROTEIRO.md.
"""
import json
import traceback
from pathlib import Path

from src.configs.training import (
    NnCfg, FNOConfig, FNO_GNNConfig, GNN_PostBaseConfig, FNO_BipartiteGNNConfig,
)
from src.configs.monitor import MonitorCfg
from src.neural_op.archs.fno import full_spectrum_modes
from scripts.train import run

UNIFIED_ROOT = 'mesh_ans_138x276_unified'       # data/torch/data_chunks/<UNIFIED_ROOT>/<arch>/
PROBLEM      = f'{UNIFIED_ROOT}_best_mse_mae'   # data/logs/<PROBLEM>/<arch>/

# ── Correções da bateria definitiva (2026-10-03) ─────────────────────────────
# Cada uma com o comportamento antigo disponível pela própria chave, pra
# permitir comparar antes × depois. Hiperparâmetros (lr, γ, larguras, camadas,
# batch, n_epochs, train_split, lambda_loss) NÃO mudam.
# [REMOVIDO 2026-10-05, B1b] padding circular mistura Bx/By girados de 120° no corte
# INTERP_MODE      = 'cell_centered'  # B1: 'legacy' (antigo) | 'cell_centered'
INTERP_MODE      = 'cell_centered_border'  # B1b: 'legacy' | 'cell_centered' (obsoleto)
                                           #      | 'cell_centered_border'
FNO_NODE_RESCALE = True             # B2: False = antigo (FNO@nós na escala de y_hw)
FULL_SPECTRUM    = True             # B3: False = antigo (data_res/modes abaixo, comentados)
GRID_HW          = (138, 276)       # grade real dos chunks unificados
N_REPEATS        = 1                # B5: [DECISÃO PENDENTE] 1 = execução única;
                                    #     N>1 = N repetições sem controle de semente
INCLUDE_REL_L2   = False            # B6: [DECISÃO PENDENTE] True adiciona loss='relative_l2'
                                    #     ATENÇÃO: relative_l2_loss normaliza por linha da
                                    #     dim 0 — em nós [S,C] isso é POR NÓ (|e|/|y| com y em
                                    #     z-score, ~0 em muitos nós), não por amostra
METRICS_EVERY_EPOCH = True          # B4: mae_hw/mae_graph em toda época (epochs.csv);
                                    #     custo: +1 forward sobre o test set por época
# [REMOVIDO 2026-10-05] só as archs em grafo — bateria completa retreina também o FNO2d
# (decisão do usuário: as 8 runs na mesma execução, com cell_centered_border)
# ONLY_ARCHS = ('FNO_GNN', 'GNN_PostBase', 'FNO_BipartiteGNN')
N_EPOCHS = 350                      # 2026-10-05: limite de épocas da bateria (antes: default de
                                    #     NnCfg, 500); StepLR com passo fixo (scheduler_step=100)
                                    #     não depende disso; GL/paciência continuam valendo
ONLY_ARCHS = None                   # None = todas de BEST_CONFIGS (8 runs, FNO2d incluído);
                                    #     tupla de archs = só essas

if FULL_SPECTRUM:
    DATA_RES             = GRID_HW
    MODES1, MODES2       = full_spectrum_modes(*GRID_HW)   # (69, 139)
    DATA_RES_BIPARTITE   = GRID_HW
else:
    # valores da bateria anterior: FNO2d/FNO_GNN com data_res=(135,270) (modes1
    # truncado em 135 → weights1/weights2 sobrepostos nas linhas 3..134);
    # Bipartite com (138,276) (modes1=138 → weights2 sobrescreve weights1 inteiro)
    DATA_RES             = (135, 270)
    MODES1, MODES2       = 270, 270
    DATA_RES_BIPARTITE   = (138, 276)
# [REMOVIDO 2026-10-01] base única com loss fixa — substituído por pareamento por loss
# (GNN_PostBase mse -> FNO2d mse, mae -> FNO2d mae), decisão do usuário.
# BASE_LOSS    = 'mae'   # loss do FNO2d desta bateria usado como base do GNN_PostBase
#                        # (mesma loss da base antiga, mesh_138x276_FEMM_MESH/FNO2d/run_0001)


# Cada entrada: (arch, dataset, lr, scheduler_gamma, arch_cfg) — valores
# reconstruídos do config.json do melhor run de cada arquitetura (mae_hw/
# mae_graph mais baixo entre os runs com treino substancial em datasets mesh_*).
BEST_CONFIGS = [
    dict(
        arch='FNO2d',
        # [REMOVIDO 2026-10-01] dataset antigo (raw v1, B suavizado do FEMM) — ver docstring
        # dataset='mesh_138x276_FEMM_MESH',       # melhor run: run_0002 (mae_hw=0.0207 T)
        dataset=f'{UNIFIED_ROOT}/FNO2d',
        lr=0.005,
        scheduler_gamma=0.7,
        # [REMOVIDO 2026-10-03, B3] modes/data_res fixos — agora via FULL_SPECTRUM
        # arch_cfg=FNOConfig(
        #     modes1=270, modes2=270, conv_width=8, conv_layers=4,
        #     lift_width=64, lift_layers=3, proj_width=64, proj_layers=3,
        #     data_res=(135, 270),
        # ),
        arch_cfg=FNOConfig(
            modes1=MODES1, modes2=MODES2, conv_width=8, conv_layers=4,
            lift_width=64, lift_layers=3, proj_width=64, proj_layers=3,
            data_res=DATA_RES, interp_mode=INTERP_MODE,
        ),
    ),
    dict(
        arch='FNO_GNN',
        # [REMOVIDO 2026-10-01] dataset antigo (raw v1, B suavizado do FEMM) — ver docstring
        # dataset='mesh_138x276_FEMM_MESH',       # melhor run: run_0002 (mae_graph=0.1010 T)
        dataset=f'{UNIFIED_ROOT}/FNO_GNN',
        lr=0.003,
        scheduler_gamma=0.8,
        # [REMOVIDO 2026-10-03, B3] modes/data_res fixos — agora via FULL_SPECTRUM
        # arch_cfg=FNO_GNNConfig(
        #     fno_modes1=270, fno_modes2=270, fno_conv_width=6, fno_conv_layers=4,
        #     fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
        #     data_res=(135, 270), gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
        # ),
        arch_cfg=FNO_GNNConfig(
            fno_modes1=MODES1, fno_modes2=MODES2, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=DATA_RES, gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
            interp_mode=INTERP_MODE, fno_node_rescale=FNO_NODE_RESCALE,
        ),
    ),
    dict(
        arch='GNN_PostBase',
        # [REMOVIDO 2026-10-01] dataset antigo (raw v1, B suavizado do FEMM) — ver docstring
        # dataset='mesh_138x276_FEMM_MESH',       # melhor run: run_0004 (mae_graph=0.1323 T)
        dataset=f'{UNIFIED_ROOT}/GNN_PostBase',
        lr=0.001,
        scheduler_gamma=0.6,
        # [REMOVIDO 2026-10-01] base fixa antiga (treinada contra o B suavizado do FEMM) e
        # construção imediata — substituído por fábrica chamada na hora do treino
        # (_make_postbase_cfg), com base = FNO2d desta bateria (ver docstring).
        # arch_cfg=GNN_PostBaseConfig(
        #     base_run_dir='data/logs/mesh_138x276_FEMM_MESH/FNO2d/run_0001',
        #     base_checkpoint='best',
        #     gnn_node_width=64, gnn_n_layers=6,
        # ),
        # [REMOVIDO 2026-10-01] fábrica sem loss (base com BASE_LOSS fixa)
        # arch_cfg=lambda: _make_postbase_cfg(gnn_node_width=64, gnn_n_layers=6),
        # fábrica recebe a loss do treino -> base = FNO2d desta bateria com a mesma loss
        # [REMOVIDO 2026-10-03, B5] fábrica só por loss — agora (loss, repetição)
        # arch_cfg=lambda loss: _make_postbase_cfg(loss, gnn_node_width=64, gnn_n_layers=6),
        arch_cfg=lambda loss, repeat: _make_postbase_cfg(
            loss, repeat, gnn_node_width=64, gnn_n_layers=6, interp_mode=INTERP_MODE),
    ),
    dict(
        arch='FNO_BipartiteGNN',
        # [REMOVIDO 2026-10-01] dataset antigo (mesmo raw, mas fora da pasta unificada)
        # dataset='mesh_ans_138x276_B',           # melhor run: run_0015 (mae_graph=0.0503 T,
        dataset=f'{UNIFIED_ROOT}/FNO_BipartiteGNN',  # melhor run antigo: mesh_ans_138x276_B/run_0015
        lr=0.01,                                # (mae_graph=0.0503 T, lá com graph_div_b_loss — aqui mse/mae puros)
        scheduler_gamma=0.6,
        # [REMOVIDO 2026-10-03, B3] modes/data_res fixos — agora via FULL_SPECTRUM
        # arch_cfg=FNO_BipartiteGNNConfig(
        #     fno_modes1=270, fno_modes2=270, fno_conv_width=6, fno_conv_layers=4,
        #     fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
        #     data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
        # ),
        arch_cfg=FNO_BipartiteGNNConfig(
            fno_modes1=MODES1, fno_modes2=MODES2, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=DATA_RES_BIPARTITE, gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
            interp_mode=INTERP_MODE, fno_node_rescale=FNO_NODE_RESCALE,
        ),
    ),
]

# [REMOVIDO 2026-10-03, B6] lista fixa — relative_l2 entra por INCLUDE_REL_L2
# LOSSES = ['mse', 'mae']
LOSSES = ['mse', 'mae'] + (['relative_l2'] if INCLUDE_REL_L2 else [])


# [REMOVIDO 2026-10-01] default loss=BASE_LOSS — loss agora sempre explícita
# def _find_base_run_dir(arch: str = 'FNO2d', loss: str = BASE_LOSS) -> str:
# [REMOVIDO 2026-10-03, B5] pareamento só por loss — agora (loss, repetição);
# também passa a ignorar runs que não terminaram (status running/failed)
# def _find_base_run_dir(loss: str, arch: str = 'FNO2d') -> str:
#     candidates = []
#     for run_dir in sorted((Path('data/logs') / PROBLEM / arch).glob('run_*')):
#         cfg_path = run_dir / 'config.json'
#         if not cfg_path.exists() or not (run_dir / 'checkpoints' / 'best.pth').exists():
#             continue
#         with open(cfg_path, encoding='utf-8') as f:
#             if json.load(f).get('loss') == loss:
#                 candidates.append(run_dir)
#     if not candidates:
#         raise FileNotFoundError(...)
#     return candidates[-1].as_posix()
def _find_base_run_dir(loss: str, repeat: int = 0, arch: str = 'FNO2d') -> str:
    """Run mais recente de data/logs/<PROBLEM>/<arch>/ treinado com `loss`, na
    repetição `repeat`, terminado (status done/stopped) e com checkpoints/best.pth
    — a base do GNN_PostBase desta bateria."""
    candidates = []
    for run_dir in sorted((Path('data/logs') / PROBLEM / arch).glob('run_*')):
        cfg_path = run_dir / 'config.json'
        if not cfg_path.exists() or not (run_dir / 'checkpoints' / 'best.pth').exists():
            continue
        status_path = run_dir / 'status.txt'
        status = status_path.read_text().strip() if status_path.exists() else ''
        if status not in ('done', 'stopped'):
            continue
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)
        # trava extra: a base tem que ter sido treinada com as MESMAS correções
        # desta bateria (B1/B3) — nunca um FNO2d de configuração antiga
        acfg = cfg.get('arch_cfg', {})
        # [REMOVIDO 2026-10-05, B1b] interp_mode na trava — o FNO2d não interpola
        # no treino (só grade); o GNN_PostBase interpola a saída da base com o SEU
        # interp_mode (GNN_PostBaseConfig.interp_mode), não com o do config da base.
        # Exigir igualdade recusaria os FNO2d da bateria (treinados com 'cell_centered').
        # same_fixes = (acfg.get('interp_mode', 'legacy') == INTERP_MODE
        #               and tuple(acfg.get('data_res', ())) == tuple(DATA_RES)
        #               and acfg.get('modes1') == MODES1 and acfg.get('modes2') == MODES2)
        same_fixes = (tuple(acfg.get('data_res', ())) == tuple(DATA_RES)
                      and acfg.get('modes1') == MODES1 and acfg.get('modes2') == MODES2)
        if cfg.get('loss') == loss and cfg.get('repeat', 0) == repeat and same_fixes:
            candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(
            f"nenhum run {arch} com loss={loss!r}, repeat={repeat}, status done/stopped e "
            f"best.pth em data/logs/{PROBLEM}/{arch}/ — base do GNN_PostBase precisa ser "
            f"treinada antes (ordem de BEST_CONFIGS)")
    return candidates[-1].as_posix()


# [REMOVIDO 2026-10-01] versão sem loss (base com BASE_LOSS fixa)
# def _make_postbase_cfg(**kw) -> GNN_PostBaseConfig:
#     base_run_dir = _find_base_run_dir()
# [REMOVIDO 2026-10-03, B5] versão sem repetição
# def _make_postbase_cfg(loss: str, **kw) -> GNN_PostBaseConfig:
#     base_run_dir = _find_base_run_dir(loss)
def _make_postbase_cfg(loss: str, repeat: int = 0, **kw) -> GNN_PostBaseConfig:
    base_run_dir = _find_base_run_dir(loss, repeat)
    print(f"  GNN_PostBase base_run_dir = {base_run_dir}  (loss={loss}, repeat={repeat})",
          flush=True)
    return GNN_PostBaseConfig(base_run_dir=base_run_dir, base_checkpoint='best', **kw)


# [REMOVIDO 2026-10-03, B5] sem repetições
# def build_run_list():
#     runs = []
#     for base in BEST_CONFIGS:
#         for loss in LOSSES:
#             runs.append(dict(base, loss=loss))
#     return runs
def build_run_list(n_repeats: int = None):
    """Repetição externa: dentro de cada repetição, FNO2d vem antes de
    GNN_PostBase (ordem de BEST_CONFIGS), garantindo a base da mesma
    (loss, repetição)."""
    n_repeats = N_REPEATS if n_repeats is None else n_repeats
    runs = []
    for rep in range(n_repeats):
        for base in BEST_CONFIGS:
            if ONLY_ARCHS is not None and base['arch'] not in ONLY_ARCHS:
                continue
            for loss in LOSSES:
                runs.append(dict(base, loss=loss, repeat=rep))
    return runs


def make_nn_cfg(spec, problem: str = None, **overrides) -> NnCfg:
    """NnCfg de um item de build_run_list(). Fatorado do __main__ pra ser
    reaproveitado pelo smoke test (scripts/verify_battery.py, V3), que troca
    problem/n_epochs sem duplicar a montagem."""
    arch_cfg = spec['arch_cfg']
    if callable(arch_cfg):
        arch_cfg = arch_cfg(spec['loss'], spec['repeat'])
    kw = dict(
        dataset=spec['dataset'],
        problem=problem or PROBLEM,
        arch=spec['arch'],
        loss=spec['loss'],
        lr=spec['lr'],
        scheduler_gamma=spec['scheduler_gamma'],
        arch_cfg=arch_cfg,
        repeat=spec['repeat'],
        n_epochs=N_EPOCHS,
        # critério de parada = defaults atuais de MonitorCfg (gl_patience=3,
        # min_epochs=100, early_stop_patience=10); só liga as métricas por época
        monitor_cfg=MonitorCfg(metrics_every_epoch=METRICS_EVERY_EPOCH),
    )
    kw.update(overrides)
    return NnCfg(**kw)


if __name__ == '__main__':
    summary = []
    for spec in build_run_list():
        label = f"{spec['arch']} / {spec['dataset']} / loss={spec['loss']} / rep={spec['repeat']}"
        print(f"\n{'='*80}\n{label}\n{'='*80}", flush=True)
        try:
            _nn = make_nn_cfg(spec)
            # [REMOVIDO 2026-10-03, B5] montagem inline — fatorada em make_nn_cfg
            # (repetição + reaproveitamento pelo smoke test V3)
            # _nn = NnCfg(
            #     dataset=spec['dataset'],
            #     # sufixo só no problem (diretório de logs, data/logs/{problem}/{arch}/) —
            #     # dataset continua apontando pros chunks reais; separa estas runs
            #     # (best hparams × mse/mae) das demais já registradas em data/logs/{dataset}/.
            #     # [REMOVIDO 2026-10-01] problem derivado do dataset — com dataset em
            #     # subpasta (<UNIFIED_ROOT>/<arch>) viraria um problem por arch; um único
            #     # PROBLEM agrupa a bateria em data/logs/<PROBLEM>/<arch>/.
            #     # problem=f"{spec['dataset']}_best_mse_mae",
            #     problem=PROBLEM,
            #     arch=spec['arch'],
            #     loss=spec['loss'],
            #     lr=spec['lr'],
            #     scheduler_gamma=spec['scheduler_gamma'],
            #     # arch_cfg pode ser fábrica (GNN_PostBase — base só existe após o FNO2d;
            #     # recebe a loss do treino pra parear com o FNO2d de mesma loss)
            #     # [REMOVIDO 2026-10-01] fábrica sem argumento
            #     # arch_cfg=spec['arch_cfg']() if callable(spec['arch_cfg']) else spec['arch_cfg'],
            #     arch_cfg=(spec['arch_cfg'](spec['loss']) if callable(spec['arch_cfg'])
            #               else spec['arch_cfg']),
            # )
            status = run(_nn)
            summary.append((label, status, None))
        except Exception as e:
            traceback.print_exc()
            summary.append((label, 'failed', str(e)))

    print(f"\n{'='*80}\nResumo\n{'='*80}")
    for label, status, err in summary:
        line = f"  [{status:8s}] {label}"
        if err:
            line += f"  -- {err}"
        print(line)
