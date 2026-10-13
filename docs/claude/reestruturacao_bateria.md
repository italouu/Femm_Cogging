# Reestruturação: bateria de treinos dentro da arquitetura (2026-10-13 — CONCLUÍDA)

Objetivo: mover a funcionalidade "executar uma bateria (sequência de treinos)" dos scripts
avulsos de `scripts/` para a arquitetura do projeto (`src/configs` + `src/training`). Ao
final, os scripts abaixo são **eliminados** (pelo método: comentar → novo → testar → confirmar
com o usuário → remover). Nada é movido/apagado sem confirmação.

## Scripts de bateria a eliminar ao final

| Arquivo | Papel hoje | Situação |
|---|---|---|
| `scripts/run_best_configs.py` | núcleo: `BEST_CONFIGS`, `LOSSES`, chaves B1–B6, `N_EPOCHS`, `_find_base_run_dir`, `build_run_list`, `make_nn_cfg` (bateria unificada) | importado por quase todos abaixo |
| `scripts/run_best_configs_smooth.py` | bateria smooth (troca dataset/problem/test_split; monkeypatch `rbc.PROBLEM`) | **em uso** |
| `scripts/run_elem_proto.py` | Elem unificado: `make_nn_cfg`, `precheck`, `ARCH`, `LOSSES` | importado pelo smooth |
| `scripts/run_elem_proto_smooth.py` | Elem smooth | **em uso** |
| `scripts/rerun_battery_run.py` | refaz 1 run (Bipartite mse, unificado) | pontual, já executado |
| `scripts/archive_old_battery_logs.py` | B0: arquiva bateria antiga | pontual, já executado |
| `scripts/run_bateria_definitiva.py` | pipeline ponta a ponta (unificado), chama `tests.*` | histórico |
| `scripts/run_vm_pipeline.sh` | idem, para a VM | histórico |
| `scripts/run_vm_elem_proto.sh`, `run_vm_elem_proto_chunks.sh`, `run_vm_elem_proto_train.sh` | descontinuados (`exit 2`) | mortos |

## Dependentes (precisam ser migrados ANTES de eliminar)

| Arquivo | Dependência |
|---|---|
| `tests/resultados/common.py:35` | `from scripts.run_best_configs import INTERP_MODE` — **avaliação ativa** |
| `tests/proto_elem_smoke.py:65,120,125` | `make_nn_cfg` (run_elem_proto), `METRICS_EVERY_EPOCH` |
| `tests/pos_bateria_common.py:36` | `INTERP_MODE` (avaliação antiga) |
| `tests/verify_battery.py:279` | `import scripts.run_best_configs as rbc` (smoke V3) |
| `tests/phase0_measurements.py`, `tests/count_battery_params.py`, `tests/eval_surface_integral_table.py` | citam/usam a bateria unificada (verificar) |
| `scripts/train.py` | `run()` é chamado por todos os runners (fica, mas deve migrar para `src/training`) |
| `scripts/build_elem_smooth_chunks.py`, `scripts/build_elem_proto_chunks.py` | só citam em comentário (verificar) |
| Docs: `CLAUDE.md`, `docs/claude/estrutura_arquivos.md`, `pipelines_e_archs.md`, `pendencias_historicas.md` | comandos `python -m scripts.run_*` |

## Progresso

- [x] **2026-10-13 — `arch`/`loss` derivados do tipo de `arch_cfg`/`loss_cfg`** (`NnCfg`,
  `init=False`; `config.json` inalterado). Defaults de `NnCfg`/arch configs/`MonitorCfg` =
  melhores runs smooth; `scripts/train.py` `__main__` = `run(NnCfg())`. `FNO_GNN_Field` fora do
  registro. `make_nn_cfg` de `run_best_configs`/`run_elem_proto` adaptados (`loss_cfg=` no
  lugar de `arch=`/`loss=`). Testes: as 10 runs smooth reconstruídas só com defaults batem
  campo a campo com o `config.json`; idem via scripts de bateria; smoke de 1 época ok;
  `load_model` ok. Trechos `# [REMOVIDO 2026-10-13]` aguardam confirmação para serem apagados.
  Testes obsoletos desativados: `tests/proto_femm_mesh_v2_arch_smoke.py`, `tests/smoke_test_refactor.py`.
- [x] **2026-10-13 — bateria implementada** (só arquivos novos):
  - `src/configs/battery.py` — `BatteryRun(name, build, needs)`, `BatteryCfg(name, runs, n_repeats)`;
    `build(runs) -> NnCfg` escrito por inteiro, chamado só na hora do treino; `needs` declara
    dependência (GNN_PostBase recebe `runs['fno2d_mae']` = run_dir da base desta bateria).
  - `src/configs/batteries/smooth.py` (8 runs) e `elem_smooth.py` (2 runs).
  - `scripts/run_battery.py` — pré-checagem (constrói toda run sem dependência antes de
    treinar), repetições (`NnCfg.repeat` gravado pelo runner), manifesto
    `data/logs/_batteries/<nome>/manifest.json` (retomada pula runs concluídas com config
    idêntica; definição alterada → aborta com diff; Ctrl+C interrompe a bateria). Usa
    `scripts/train.py::run` sem alteração.
  - Testes: as 10 definições reproduzem campo a campo os `config.json` das runs smooth/Elem
    existentes; `--dry-run` local recusa (datasets só na VM); `tests/battery_smoke.py`
    (2 runs com dependência × 2 repetições, 1 época) ok, retomada ok, mudança de `lr` abortada
    com diff. Não testado: Ctrl+C no meio de uma run.
- [x] **2026-10-13 — uma única bateria (`smooth`) e limpeza dos logs.** Apagados todos os logs
  não-smooth (bateria unificada, `_smoke_*`, `_verificacao_*`, `_regen_raw_fase`, `_batteries/_smoke`).
  Pasta `..._smooth_best_mse_mae` → `data/logs/mesh_ans_138x276_smooth_battery/` ("best configs"
  eram só configs escolhidas), com `problem`/`base_run_dir` reescritos nos `config.json`;
  Elem continua em `..._smooth_elem_proto/`. Elem juntada em `src/configs/batteries/smooth.py`
  (10 runs); `elem_smooth.py` comentado. `run_best_configs_smooth`/`run_elem_proto_smooth`
  desativados (`raise SystemExit`). `tests/resultados/common.py` não importa mais de
  `scripts.run_best_configs`. Verificado: 10 definições = `config.json` renomeados; `load_model` ok.
- [ ] Rodar `python -m scripts.run_battery smooth --dry-run` na VM
- [x] **2026-10-13 — eliminação (confirmada pelo usuário).** Apagados os 11 scripts da tabela acima e
  `src/configs/batteries/elem_smooth.py`. Dependentes: `tests/resultados/common.py` e
  `tests/pos_bateria_common.py` passam a definir `INTERP_MODE = 'cell_centered_border'`; os da bateria
  unificada foram apagados (`verify_battery`, `proto_elem_smoke`, `phase0_measurements`,
  `count_battery_params`, `pos_bateria_eval/figures/figures_dente/report/timing`). Re-testado: 10
  definições = `config.json`; `tests/battery_smoke.py` ponta a ponta (+ retomada).
- [ ] `python -m scripts.run_battery smooth --dry-run` na VM (datasets só lá).

Na mesma sessão (fora da bateria): removidos o pós-processamento de cogging (`tests/cogging_torque_*`,
`tests/resultados/cogging_torque_smooth.py`), o Darcy (`src/bench/datasets/`, `scripts/parsers/`) e a
arch `FNO_ref` (`FNORefConfig`, `fno_ref.py`, registro, `bench/metrics.py`).
