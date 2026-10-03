# Bateria definitiva `mesh_ans_138x276_unified` — roteiro de execução

Preparado em 2026-10-03. Todos os comandos rodam **da raiz do projeto**. Nada
aqui dispara a bateria sozinho — o passo 6 é o único que treina de verdade.

## Execução na VM (Linux) — caminho recomendado

Tudo dos passos 2–5 abaixo (e opcionalmente o B0 e a bateria) está encadeado em
`scripts/run_vm_pipeline.sh`, que para na primeira falha e é retomável:

```
git pull
# 1) copiar para a VM (não estão no git — ver "Pré-requisitos" abaixo):
#      data/raw/mesh_ans_138x276/                         (4000 .ans.gz + valid_designs.csv)
#      data/logs/mesh_ans_138x276_unified_best_mse_mae/   (8 runs antigas — Fase 0 e V2)
#      [opcional] data/torch/data_chunks/mesh_ans_138x276_unified/  (senão são gerados)
# 2) revisar as decisões pendentes (seção 1) no topo de scripts/run_best_configs.py
bash scripts/run_vm_pipeline.sh                    # precheck, chunks, Fase 0, V1, V2, params, V3
# 3) revisar docs/bateria_definitiva/*.json/*.md
bash scripts/run_vm_pipeline.sh --archive          # B0 (move logs antigos -> ..._oldcriterion)
bash scripts/run_vm_pipeline.sh --battery          # bateria (exige archive concluído)
```

- Logs por etapa e marcadores de conclusão em `data/logs/_vm_pipeline/`
  (`<etapa>.log`, `.done_<etapa>`). Etapa com `.done_*` é pulada; apagar o
  marcador (ou usar `--only <etapa>`) para refazer.
- `PYTHON=python3.x bash scripts/run_vm_pipeline.sh` para escolher o interpretador.
- O precheck avisa se a GPU tiver < 26 GiB: no smoke desta máquina (RTX 4080,
  16 GB) o GNN_PostBase (width 64 × 6, batch 32) teve pico de ~25 GiB e falhou.
- `docs/bateria_definitiva/phase0_results.*` e `verify_results.json` que estão
  nesta máquina são **parciais** (Fase 0 em 1 chunk; V3 incompleto) e não foram
  commitados — serão gerados de novo na VM.

## 0. Pré-requisitos na máquina de execução

- `git pull` até o commit com este arquivo (commits `F0`, `B1`, `B2`, `B3`, `B4`, `B5+B6`, `B0`, `V`).
- Raw `data/raw/mesh_ans_138x276/` (4000 `.ans.gz` + `valid_designs.csv`).
- Logs da bateria anterior em `data/logs/mesh_ans_138x276_unified_best_mse_mae/`
  (8 runs com `checkpoints/best.pth` e `split.json`) — necessários para a Fase 0.
- Chunks unificados (`data/torch/data_chunks/mesh_ans_138x276_unified/<arch>/`).
  Se ainda não existirem: `python -m scripts.build_unified_ans_chunks_direct`
  (~22 min, ~105 GB, retomável). Fase 0/V2 funcionam sem eles (remontam do raw,
  ~17 s/chunk), mas a bateria (passo 6) precisa.

## 1. Decisões pendentes — editar o topo de `scripts/run_best_configs.py`

| chave | atual | decisão |
|---|---|---|
| `N_REPEATS` (B5) | `1` | execução única (1) ou N repetições sem semente — GNN_PostBase já pareia por (loss, repetição) |
| `INCLUDE_REL_L2` (B6) | `False` | incluir `relative_l2` nas 4 archs. **Atenção**: `relative_l2_loss` normaliza por linha da dim 0 → em nós `[S,C]` é erro relativo POR NÓ (\|e\|/\|y\| com y em z-score, ~0 em muitos nós), não por amostra |
| `INTERP_MODE` (B1) | `'cell_centered'` | `'legacy'` = antigo |
| `FNO_NODE_RESCALE` (B2) | `True` | `False` = antigo |
| `FULL_SPECTRUM` (B3) | `True` | `False` = data_res/modes antigos |
| `METRICS_EVERY_EPOCH` (B4) | `True` | +1 forward sobre o test set (70% dos dados) por época — ver tempo no smoke (V3) |

## 2. Fase 0 — medições com os checkpoints atuais (ANTES do B0)

```
python -m scripts.phase0_measurements
```
Saída: `docs/bateria_definitiva/phase0_results.{json,md}` (T0a, T0b, T0c com os
dois estimadores de área). ~17 s/chunk se remontar do raw (~88 chunks de teste),
bem menos com os chunks em disco.

## 3. B0 — arquivar os logs antigos

```
python -m scripts.archive_old_battery_logs            # simulação
python -m scripts.archive_old_battery_logs --execute  # move para ..._oldcriterion/
```
Recusa se a Fase 0 completa não existir. (O `_find_base_run_dir` também
rejeita qualquer FNO2d com correções diferentes — segunda trava.)

## 4. Verificação (Fase 2)

```
python -m scripts.verify_battery v1      # interpolação (sintético)
python -m scripts.verify_battery v2      # escala do FNO@nós com B2
python -m scripts.verify_battery v3      # smoke: 4 archs × losses, 3 épocas, pasta temporária
```
`v3` cria `data/torch/data_chunks/_smoke_bateria_definitiva/` (hardlinks dos 4
primeiros chunks, ou remontados do raw) e `data/logs/_smoke_bateria_definitiva/`,
e apaga os dois no fim (`--keep` para inspecionar). Confere: treino sem erro,
GNN_PostBase acha a base do smoke com a mesma loss/repetição, `epochs.csv`
(1 linha/época com mae), `run_summary.json`, `best.pth`, `model_final.pth`,
`config.json` com `n_params_real`/`git_commit`/`postbase_base`.
Resultados em `docs/bateria_definitiva/verify_results.json`.

## 5. Contagem de parâmetros (B3)

```
python -m scripts.count_battery_params
```

## 6. Bateria (só depois de 1–5 revisados)

```
python -m scripts.run_best_configs
```
Logs em `data/logs/mesh_ans_138x276_unified_best_mse_mae/<arch>/run_XXXX/`:
`config.json`, `split.json`, `metrics.jsonl` (heartbeat), `epochs.csv` (por
época), `run_summary.json`, `checkpoints/best.pth`, `model_final.pth`.

**GNN_PostBase (width 64 × 6 camadas, batch 32)** estourou 16 GB numa RTX 4080
na bateria anterior (pico 16,2 GiB mesmo com batch 16). O B3 reduz só o FNO
base (congelado); a memória é dominada pela GNN — ver pico no V3 antes de rodar.
