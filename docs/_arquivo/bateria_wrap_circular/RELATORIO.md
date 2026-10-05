# Relatório — bateria definitiva `mesh_ans_138x276_unified_best_mse_mae`

Gerado em 2026-10-04 a partir dos arquivos de cada run
(`config.json`, `run_summary.json`, `epochs.csv`, `metrics.jsonl`, `split.json`) em
`data/logs/mesh_ans_138x276_unified_best_mse_mae/` e de
`docs/bateria_definitiva/{verify_results,param_counts_b3}.json`.

- Execução: VM Linux, GPU NVIDIA L40S-48Q, `python -m scripts.run_bateria_definitiva`,
  de 2026-10-03 19:48 a 2026-10-04 13:12 (≈ 17,4 h de treino somadas).
- Código: commit `2cab72d` com `git_dirty: true` em todos os runs (ver §5).
- Dataset: `mesh_ans_138x276_unified/<arch>` (4000 amostras, 125 chunks × 32), gabarito B
  único (curl(A) por elemento P1 + média simples nos nós).
- Split: `split_seed=12`, `train_split=0.30` (treino 30% / teste 70%), `split.json`
  idêntico nos 8 runs.
- Execução única por configuração, sem semente de treino fixa (só o split é determinístico).

## 1. Configuração dos runs

Hiperparâmetros idênticos aos da bateria anterior (conferido campo a campo contra os
`config.json` antigos). Comum a todos: `n_epochs=500`, `batch_size=32`,
scheduler `step` a cada 100 épocas, `normalize=True`, `lambda_loss=0`.

| Arch | lr | γ | FNO (conv width × camadas) | GNN (width × camadas) | data_res | modos |
|---|---|---|---|---|---|---|
| FNO2d | 0,005 | 0,7 | 8 × 4 | — | 138×276 | 69×139 |
| FNO_GNN | 0,003 | 0,8 | 6 × 4 | 32 × 3 | 138×276 | 69×139 |
| GNN_PostBase | 0,001 | 0,6 | base FNO2d congelada | 64 × 6 | (da base) | (da base) |
| FNO_BipartiteGNN | 0,01 | 0,6 | 6 × 4 | 32 × 3 | 138×276 | 69×139 |

Lift/proj do FNO: 64 × 3 em todas. Critério de parada: GL 5% sustentado por 3 heartbeats,
paciência de 10 heartbeats sem novo best (Δ 1e-6), nenhum dos dois antes da época 100;
heartbeat a cada 10 épocas.

GNN_PostBase pareado por loss: `mse` → base `FNO2d/run_0001` (best, época 139);
`mae` → base `FNO2d/run_0002` (best, época 209). Caminho e checkpoint registrados em
`config.json → postbase_base`.

## 2. Resultados (checkpoint `best`, conjunto de teste)

`mae_hw`/`mae_graph`: MAE em unidade física (T), na grade H×W e nos nós, na época do best.
`test_loss` está no espaço normalizado (não comparável entre mse e mae).

| Arch | Loss | Run | Parada | Última ép. | Best ép. | test_loss | mae_hw (T) | mae_graph (T) |
|---|---|---|---|---|---|---|---|---|
| FNO2d | mse | run_0001 | paciência | 239 | 139 | 0,00656 | 0,0191 | — |
| FNO2d | mae | run_0002 | paciência | 309 | 209 | 0,04194 | **0,0170** | — |
| FNO_GNN | mse | run_0001 | GL ⚠ | 249 | 209 | 0,06789 | 0,198 | 0,0761 |
| FNO_GNN | mae | run_0002 | GL ⚠ | 109 | 79 | 0,14234 | 0,164 | 0,0755 |
| GNN_PostBase | mse | run_0001 | GL ⚠ | 129 | 99 | 0,03243 | 0,0191 | 0,0522 |
| GNN_PostBase | mae | run_0002 | n_epochs | 499 | 479 | 0,08858 | 0,0170 | **0,0470** |
| FNO_BipartiteGNN | mse | run_0001 | paciência | 189 | 89 | 0,04053 | 0,212 | 0,0541 |
| FNO_BipartiteGNN | mae | run_0002 | paciência | 229 | 129 | 0,09302 | 0,248 | 0,0493 |

Leituras:
- Nos nós, o melhor é GNN_PostBase mae (0,0470 T), seguido de FNO_BipartiteGNN mae
  (0,0493 T); FNO_GNN fica em ~0,076 T nas duas losses. A loss `mae` dá `mae_graph`
  menor que `mse` em todas as archs com grafo.
- `mae_hw` de FNO_GNN e FNO_BipartiteGNN (0,16–0,25 T) **não mede qualidade**: com
  `lambda_loss=0` a saída do estágio FNO na grade não é supervisionada. Para essas duas archs
  só `mae_graph` é métrica de resultado.
- `mae_hw` do GNN_PostBase é exatamente o do FNO2d base da mesma loss (0,0191/0,0170) — a
  base está de fato congelada.
- Estas são as métricas registradas durante o treino (MAE por componente). A métrica de
  resultado é a da §2.1.

### 2.1 Erro por integral de superfície contra o gabarito na malha

`python -m scripts.eval_surface_integral_table` (2026-10-04, RTX 4080; chunks de teste
remontados do raw), saída em
`data/logs/mesh_ans_138x276_unified_best_mse_mae/surface_integral_table.json`.

**Cálculo**
- Teste inteiro: 88 chunks, 2816 amostras, 89.743.872 nós; checkpoint `best.pth`.
- Gabarito: `node_y` nos vértices da malha real do FEMM (curl(A) por elemento P1 + média
  simples nos nós). **Todas as archs são comparadas nesse mesmo gabarito**, inclusive o
  FNO2d: a saída do FNO2d na grade é decodificada para T e interpolada nos nós pela mesma
  função do B1 (`interpolate_grid_to_nodes`, modo `cell_centered`, gravado no run), usando
  `r_base`/`c_base` de cada nó.
- Erro de módulo por nó: `e = | |B|_pred − |B|_true |`, com `|B| = hypot(Bx, By)`.
- Peso de área por nó `A_n`: área *lumped* (1/3 da área de cada triângulo incidente) —
  `node_dual_area` para FNO2d/FNO_GNN/GNN_PostBase; área do elemento/3 somada por vértice
  (via `cross_edge_index`) para FNO_BipartiteGNN. As duas são a mesma grandeza.
- Métricas, acumuladas globalmente sobre todo o teste (não média de médias por amostra):
  - L1_área = Σ e·A_n / Σ A_n
  - L2_área = √( Σ e²·A_n / Σ A_n )
  - MAE ponto a ponto = média simples de `e` sobre os nós (sem peso de área)
  - % = relativo a B_ref = √( Σ |B_true|²·A_n / Σ A_n ) = **0,5961 T**

| Arch | Loss | Best ép. | L1 área (T) | L1 área (%B_ref) | L2 área (T) | L2 área (%B_ref) | MAE pt (T) | MAE pt (%B_ref) |
|---|---|---|---|---|---|---|---|---|
| FNO2d | mse | 139 | 0,0291 | 4,88 | 0,0566 | 9,50 | 0,1372 | 23,01 |
| FNO2d | mae | 209 | 0,0268 | 4,50 | 0,0552 | 9,26 | 0,1365 | 22,90 |
| FNO_GNN | mse | 209 ⚠ | 0,0485 | 8,13 | 0,0693 | 11,63 | 0,0771 | 12,93 |
| FNO_GNN | mae | 79 ⚠ | 0,0431 | 7,23 | 0,0659 | 11,05 | 0,0808 | 13,56 |
| GNN_PostBase | mse | 99 ⚠ | 0,0256 | 4,29 | 0,0405 | 6,79 | 0,0583 | 9,78 |
| GNN_PostBase | mae | 479 | **0,0205** | **3,43** | **0,0360** | **6,04** | 0,0538 | 9,03 |
| FNO_BipartiteGNN | mse | 89 | 0,0368 | 6,18 | 0,0512 | 8,58 | 0,0508 | 8,52 |
| FNO_BipartiteGNN | mae | 129 | 0,0247 | 4,15 | 0,0365 | 6,12 | **0,0417** | **6,99** |

⚠ = run com divergência de treino (ver abaixo).

Leituras:
- Por área (L1/L2), o melhor é GNN_PostBase mae (3,43% / 6,04%), seguido de
  FNO_BipartiteGNN mae (4,15% / 6,12%). Ponto a ponto, o melhor é FNO_BipartiteGNN mae
  (6,99%).
- `mae` é melhor que `mse` nas 4 archs, em todas as métricas (única exceção: MAE pt do
  FNO_GNN).
- **MAE ponto a ponto e L1 por área contam histórias diferentes.** A malha é muito mais
  densa nas interfaces (entreferro, aberturas de ranhura), onde o erro é maior. O MAE pt dá
  o mesmo peso a cada nó e é dominado por essas regiões pequenas; o L1 por área pondera pela
  área física. O FNO2d mostra o efeito extremo: 4,5% por área, mas 22,9% ponto a ponto — o
  erro dele se concentra nos nós pequenos das interfaces, que é justamente o que as GNNs
  corrigem (FNO_BipartiteGNN mae: 7,0% ponto a ponto).
- FNO_GNN fica **pior que o FNO2d por área** (7,2–8,1% contra 4,5–4,9%), embora melhor
  ponto a ponto. Somado à divergência dos dois runs, é a arch menos confiável da bateria.
- Para referência (não é métrica de resultado): o FNO2d contra o gabarito **na grade**
  (`y_hw`, peso r·dr·dθ) dá L1 3,58–3,98% / L2 6,45–6,67%; a diferença para a malha é o custo
  de levar a grade 138×276 até os nós.

### ⚠ Runs com divergência de treino

Nos três runs marcados o `train_loss` explode junto com o `test_loss` (instabilidade da
otimização, não overfitting) e o GL encerra o treino; o `best.pth` é o estado anterior ao
pico. O mesmo tipo de pico já ocorria na bateria anterior.

| Run | Época do pico | train_loss antes → depois | Observação |
|---|---|---|---|
| FNO_GNN mse | 224 | 0,039 → 8,54 | recuperando devagar quando parou (test 0,082 vs best 0,068) |
| FNO_GNN mae | 88 | 0,128 → 3,77 | best na época 79 — antes de `min_epochs` |
| GNN_PostBase mse | 102 | 0,025 → 0,166 | logo após o corte de lr (ép. 100); trava em ~0,166 e não recupera |

Decisão (2026-10-04): usar o `best.pth` desses runs; investigação adiada para depois do
trabalho pronto. GNN_PostBase mae parou por `n_epochs` ainda melhorando lentamente
(best na 479) — decisão: sem mais épocas.

## 3. Custo computacional

| Arch | Loss | Parâmetros reais | Treináveis | Pico GPU (GiB) | Tempo/época mediano (s) | Tempo total (h) |
|---|---|---|---|---|---|---|
| FNO2d | mse | 9.831.210 | 9.831.210 | 3,39 | 5,2 | 0,35 |
| FNO2d | mae | 9.831.210 | 9.831.210 | 3,50 | 5,1 | 0,44 |
| FNO_GNN | mse | 5.547.157 | 5.547.157 | 14,08 | 25,5 | 1,78 |
| FNO_GNN | mae | 5.547.157 | 5.547.157 | 14,11 | 25,8 | 0,79 |
| GNN_PostBase | mse | 9.931.506 | 100.296 | 33,66 | 54,4 | 1,97 |
| GNN_PostBase | mae | 9.931.506 | 100.296 | 33,85 | 54,5 | 7,58 |
| FNO_BipartiteGNN | mse | 5.550.544 | 5.550.544 | 17,48 | 38,4 | 2,03 |
| FNO_BipartiteGNN | mae | 5.550.544 | 5.550.544 | 17,53 | 38,5 | 2,46 |

Parâmetros reais contam pesos complexos em dobro (o campo `n_params` do `config.json`
conta cada complexo uma vez e vale ~metade — usar `n_params_real`). O tempo por época inclui
o cálculo de `mae_hw`/`mae_graph` sobre o teste a cada época (B4). GNN_PostBase (~34 GiB)
não cabe numa GPU de 16 GB com batch 32.

## 4. Verificações antes da bateria

**V1 — interpolação (sintético): ok.** Campo linear reproduzido com erro máximo
4,4e-16 no modo corrigido (contra 4,9e-3 no antigo). Campo periódico em θ: valores em
0°, 0,1°, 119,9° e 120° iguais ao esperado com erro ≤ 4,4e-16, contínuos através do corte.

**V2 — escala do FNO nos nós (B2): ok.** Com `y_hw` interpolado no lugar do FNO
(chunk 0, 32 amostras, 1.012.094 nós), o valor recodificado reproduz a referência física a
4,8e-7 T (sem a correção: erro de até 0,85 T). Média no espaço codificado ~0,01 em ambos os
canais; desvio 0,87 contra 1,02 de `node_y` — a diferença restante é o piso de representação
da grade, não escala.

**V3 — smoke (4 chunks, 3 épocas, 4 archs × mse/mae, L40S-48Q): ok nos 8.** Treino
sem erro, GNN_PostBase encontrou a base certa (`FNO2d/run_0001` para mse, `run_0002` para
mae) e todos os arquivos do B4 foram gerados.

## 5. Pontos em aberto

- **`git_dirty: true`** em todos os runs: não há como saber pelos logs o que estava
  modificado na VM. Hipótese provável: os arquivos versionados de `docs/bateria_definitiva/`
  reescritos pelas etapas V1/V2/V3/params antes da bateria. Confirmar com
  `git status`/`git diff` na VM.
- **Sem conjunto de validação**: o mesmo teste (70%) escolhe o `best.pth`, decide a parada e
  reporta o erro. Resultado levemente otimista; assumir no texto ou rever o split numa
  bateria futura.
- **Sem semente de treino**: não reproduzível bit a bit.

## 6. Mudanças efetivamente aplicadas nesta bateria

Em relação à bateria anterior (mesmos dados, split e hiperparâmetros). Todas têm flag para
o comportamento antigo.

| Item | O que mudou | Onde | Commit |
|---|---|---|---|
| Critério de parada | GL sustentado (`gl_patience=3`, antes 1) e warm-up `min_epochs=100` | `src/configs/monitor.py`, `src/neural_op/monitor.py` | `6cf77e4` |
| Critério de parada | Paciência por estagnação ligada por padrão: `early_stop_patience=10` (antes desligada) | `src/configs/monitor.py` | `3346aa6` |
| B1 — interpolação grade→nós | `align_corners=False` (grade centrada em células), padding circular de 1 coluna no eixo angular (wrap 0°/120°), `border` no radial; função única `interpolate_grid_to_nodes` usada no treino (FNO_GNN, FNO_BipartiteGNN, GNN_PostBase) e na avaliação. Ativo: `interp_mode='cell_centered'` | `src/neural_op/archs/interp.py` + archs + `eval.py` | `a011d2e`, `7729a45` |
| B2 — escala do resíduo | FNO interpolado nos nós decodificado com stats de `y_hw` e recodificado com as de `node_y` antes de somar o resíduo da GNN (FNO_GNN e FNO_BipartiteGNN). Ativo: `fno_node_rescale=True` | `fno_gnn.py`, `femm_mesh_v2_gnn.py`, `gnn_post_base.py` | `fae21a5` |
| B3 — espectro do FNO | `data_res=(138,276)` em todas as archs (antes 135×270 no FNO2d/FNO_GNN) e modos = espectro completo sem sobreposição (69×139; antes 270×270, com blocos sobrepostos) | `fno.py`, `run_best_configs.py` | `39900ab`, `2291def` |
| B4 — instrumentação | `epochs.csv` por época; `run_summary.json` (pico GPU, parâmetros reais/treináveis, tempo, best, motivo da parada, commit); `postbase_base` no `config.json`; `best` e `final` sempre salvos | `model_manager.py`, `training_utils.py`, `monitor.py`, `train.py` | `eb5ce58` |
| B5 — repetições | Infraestrutura: campo `repeat`, pareamento do GNN_PostBase por (loss, repetição), base só aceita se `done/stopped` e com as mesmas correções. Usado com `N_REPEATS=1` | `run_best_configs.py`, `training.py` | `2291def` |
| Execução | Pipeline ponta a ponta (`run_bateria_definitiva.py`) e verificações V1/V2/V3 (`verify_battery.py`) | `scripts/` | `9c5e342`, `2cab72d` |

Efeito do B3 na contagem de parâmetros (reais, complexo × 2):

| Arch | Antes | Depois | Pesos espectrais mortos antes |
|---|---|---|---|
| FNO2d | 18.810.666 | 9.831.210 | 9.191.424 |
| FNO_GNN | 10.598.101 | 5.547.157 | 5.170.176 |
| GNN_PostBase (total / treináveis) | 18.910.962 / 100.296 | 9.931.506 / 100.296 | 9.191.424 |
| FNO_BipartiteGNN | 11.074.960 | 5.550.544 | 5.524.416 |

A redução vem quase toda da eliminação dos pesos sobrepostos (mortos); a capacidade
efetiva do FNO praticamente não muda.

**Implementado mas não usado nesta bateria**: `relative_l2` (B6, `INCLUDE_REL_L2=False` —
adiado para depois do pipeline garantido), repetições (`N_REPEATS=1`), Fase 0 e arquivamento
B0 (a VM não tinha os logs antigos; esta bateria não usa nenhum checkpoint antigo).
