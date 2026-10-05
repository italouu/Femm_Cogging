# Pós-bateria — `mesh_ans_138x276_unified_best_mse_mae`

Gerado em 2026-10-04. Nada foi treinado e o código de treino não foi alterado. Scripts novos
(`scripts/pos_bateria_*.py`):

| Script | O que faz |
|---|---|
| `pos_bateria_common.py` | parse do raw, arco do entreferro, saturação, harmônicos, forward dos modelos |
| `pos_bateria_eval.py` | passada única no teste: T2, T4a, T5b e L1_A por amostra (19 min) |
| `pos_bateria_timing.py femm` / `infer` | T3 (3a+3b / 3c) |
| `pos_bateria_figures.py` | figuras T4b, T5a, T5b |
| `pos_bateria_report.py` | `tabelas.md` + `pos_bateria.json` (todos os números) |

Conjunto de teste: o mesmo de `eval_surface_integral_table.py` (88 chunks do `split.json`,
2816 amostras, 89.743.872 nós). Os chunks unificados não existem nesta máquina, por isso cada
amostra foi remontada do raw com `parse_ans_gzip_sample_unified`.

**Conferência de consistência:** as métricas globais recalculadas amostra a amostra reproduzem
`surface_integral_table.json` com diferença relativa ≤ 1,3e-9 nos 8 runs.

---

## 1. Tarefa 1 — `git_dirty`

**Não executada na VM** (sem acesso a partir daqui). O que dá para afirmar sem ela:

- `git_dirty` é calculado em `src/training/model_manager.py` com
  `git status --porcelain --untracked-files=no`, ou seja, só arquivos **versionados**.
- Antes do primeiro treino, a bateria roda as etapas V1/V2/V3. `scripts/verify_battery.py`
  reescreve `docs/bateria_definitiva/verify_results.json`, que é versionado. Também escrevem em
  arquivos versionados `count_battery_params.py` (`param_counts_b3.json`) e
  `phase0_measurements.py` (`phase0_results.{json,md}`). Isso **basta para `git_dirty=true` em
  todos os runs, mesmo com o código intocado**.
- O commit seguinte, `6c765d1` (03/10 19:47:29), foi feito 70 s antes do início do 1º run
  (19:48:39). Ele altera só `scripts/run_bateria_definitiva.py` (grava a saída de cada etapa em
  `data/logs/_bateria_definitiva/`), não o treino. Os runs registram `2cab72d`. Se essa mudança
  estava aplicada sem commit na VM, ela também aparece no diff, sem efeito no treino.
- Nesta máquina, os modificados são só `docs/bateria_definitiva/*`. `git diff 2cab72d` mostra
  `verify_results.json` e `run_bateria_definitiva.py` (= `6c765d1`).

Para fechar, rodar na VM, no diretório do projeto:

```
git rev-parse HEAD
git status --porcelain --untracked-files=no
git diff --stat 2cab72d
git diff 2cab72d -- src scripts
```

Resultado esperado: só `docs/bateria_definitiva/*`, e possivelmente
`scripts/run_bateria_definitiva.py`. **Qualquer outro arquivo em `src/` ou `scripts/` deve ser
sinalizado.**

## 2. Tarefa 2 (T0a) — piso de representação da grade

Linha extra: `y_hw` (gabarito na grade, T) interpolado nos nós com
`interpolate_grid_to_nodes(cell_centered)` e comparado com `node_y`. Mede o erro que um FNO2d
que acertasse `y_hw` exatamente teria nos nós.

| Modelo | Loss | Best ép. | L1_A (T) | L1_A (%B_ref) | L2_A (T) | L2_A (%B_ref) | Pontual (T) | Pontual (%B_ref) |
|---|---|---|---|---|---|---|---|---|
| FNO2d | mse | 139 | 0,0291 | 4,88 | 0,0566 | 9,50 | 0,1372 | 23,01 |
| FNO2d | mae | 209 | 0,0268 | 4,50 | 0,0552 | 9,26 | 0,1365 | 22,90 |
| FNO_GNN | mse | 209 ⚠ | 0,0485 | 8,13 | 0,0693 | 11,63 | 0,0771 | 12,93 |
| FNO_GNN | mae | 79 ⚠ | 0,0431 | 7,23 | 0,0659 | 11,05 | 0,0808 | 13,56 |
| GNN_PostBase | mse | 99 ⚠ | 0,0256 | 4,29 | 0,0405 | 6,79 | 0,0583 | 9,78 |
| GNN_PostBase | mae | 479 | 0,0205 | 3,43 | 0,0360 | 6,04 | 0,0538 | 9,03 |
| FNO_BipartiteGNN | mse | 89 | 0,0368 | 6,18 | 0,0512 | 8,58 | 0,0508 | 8,52 |
| FNO_BipartiteGNN | mae | 129 | 0,0247 | 4,15 | 0,0365 | 6,12 | 0,0417 | 6,99 |
| **gabarito da grade interpolado** | — | — | **0,0127** | **2,13** | **0,0416** | **6,98** | **0,1220** | **20,46** |

B_ref = 0,5961 T. ⚠ = run com divergência de treino (relatório anterior).

Leituras:
- **O erro pontual do FNO2d (22,9%) é quase todo piso da grade (20,5%).** Por área (L1), o piso é
  2,13%, menos da metade do erro do FNO2d (4,50%).
- **Só os modelos em grafo ficam abaixo do piso em L2 e no erro pontual**: GNN_PostBase mae
  (6,04% L2), FNO_BipartiteGNN mae (6,12% L2; 6,99% pontual, contra 20,46%). Nenhum modelo que
  passe pela grade 138×276 chegaria a esse nível nos nós.
- O piso não é um limite inferior estrito para o FNO2d (uma grade diferente de `y_hw` poderia
  interpolar melhor), mas é o erro de um FNO2d "perfeito" contra o seu próprio alvo de treino.
- O piso inclui o efeito do wrap cartesiano (§6, item 1): ~0,5 mT dos 12,7 mT de L1.

## 3. Tarefa 3 (T7) — custo computacional

**Condições de medição** (`timing.json → hardware`):

| Item | Valor |
|---|---|
| Máquina | Windows 11 Pro 10.0.26200, Intel Core i9-13900KF (32 threads), 31,8 GiB RAM |
| GPU | NVIDIA GeForce RTX 4080 16 GB, driver 610.88 |
| Software | Python 3.12.2, PyTorch 2.3.1+cu121 (CUDA 12.1, cuDNN 8.9.7), NumPy 1.26.4, SciPy 1.14.0, Matplotlib 3.8.2 |
| FEMM | FEMM 4.2 (`d:\femm42\bin\femm.exe`, arquivo de 2019-04-21), pyfemm 0.1.3 |

Execução sequencial, sem outras cargas: FEMM primeiro, inferência depois.

**3a. FEMM**: 50 amostras de teste, igualmente espaçadas na ordem do teste. Mesmas
configurações de `save_ans_gzip_sample`: `Sym120_Annular`, `phase=0`,
`mi_probdef(0,'millimeters','planar',1e-8,0,200)`, uma sessão FEMM por amostra. Em todas as 50, a
malha regenerada é **idêntica à do raw** (mesmo nº de nós/elementos, A nodal com diferença
máxima 0), o que confirma a reprodução fiel da geração.

| Etapa | mediana (ms) | IQR (ms) |
|---|---|---|
| openfemm (fora do custo por amostra) | 47,9 | 2,4 |
| geometria (newdocument + probdef + draw_motor [desenho + materiais] + saveas) | 253,4 | 11,5 |
| malha (`mi_createmesh`) | 390,2 | 19,3 |
| solução (`mi_analyze`) | 3396,9 | 539,7 |
| leitura do .ans + B nodal (curl(A) P1 + média simples) | 81,3 | 18,6 |
| closefemm (fora do custo por amostra) | 9,3 | 3,9 |
| **malha + solução** | **3778,8** | 558,0 |

Malha: mediana de 32.182 nós.

**3b. Pré-processamento** a partir do `.ans` recém-resolvido, com as mesmas funções dos parsers,
cronometradas por etapa. A saída é **idêntica** (`np.array_equal`) à de
`parse_ans_gzip_sample_unified` nas 50 amostras.

| Etapa | Usada por | mediana (ms) |
|---|---|---|
| leitura do .ans (nós/elementos/blocos) | todos | 60,9 |
| materiais por elemento (μ_r, M, área) | todos | 2,6 |
| **x_hw**: centros de pixel nos triângulos (trifinder) + cópia de μ_r, M | todos | **403,0** |
| r_base/c_base dos nós | todos | 0,8 |
| arestas da malha + wrap | grafos | 85,9 |
| node_x/edge_attr v1 (voto de material por área) | FNO_GNN, GNN_PostBase | 13,4 |
| node_x/edge_attr/elem_x/arestas cruzadas | FNO_BipartiteGNN | 16,6 |
| numpy→GPU + z-score | FNO2d / FNO_GNN / GNN_PostBase / Bipartite | 1,2 / 3,4 / 3,7 / 5,1 |

> **No pipeline atual, a entrada x_hw do FNO2d é derivada da malha do FEMM.** μ_r e M de cada
> pixel vêm do triângulo que contém o centro do pixel. Isso exige `mi_createmesh` mesmo para o
> FNO2d, e é a etapa mais cara do pré-processamento (~85% dele, por causa da construção do
> trifinder do matplotlib). Em princípio, x_hw poderia vir direto da geometria (point-in-polygon,
> que é 100% idêntico à malha, ver CLAUDE.md), o que tiraria o FNO2d da dependência do FEMM.
> Isso não foi medido aqui.

**3c. Inferência**: `best.pth`, `torch.inference_mode()`, 10 aquecimentos + 100 repetições,
`torch.cuda.synchronize()` antes e depois de cada medição. Inclui a desnormalização e, no FNO2d,
a interpolação grade→nós. A entrada já está codificada na GPU (a codificação entra em 3b).
- **Batch 1:** amostra 2409 (nº de nós mediano do 1º chunk de teste, 31.994 nós).
- **Batch máximo:** maior potência de 2 que coube. São as amostras do 1º chunk de teste repetidas
  em ciclo.
- **Limite de memória:** o alocador foi limitado a 95% da VRAM dedicada (ver §6, item 6).

**Tabela final** (ms por amostra, medianas):

| Arch | Loss | FEMM malha+solução | Pré-proc. | Infer. batch 1 (IQR) | Batch máx. | Infer./amostra no batch máx. | Pico GPU b1 / bmáx (GiB) | Speedup sobre a solução | Speedup inferência pura |
|---|---|---|---|---|---|---|---|---|---|
| FNO2d | mse | 3779 | 472 | 2,20 (1,78) | ≥512 ¹ | 0,83 | 0,06 / 10,38 | 4,37× | 1543× |
| FNO2d | mae | 3779 | 472 | 2,08 (1,52) | ≥512 ¹ | 0,82 | 0,06 / 10,38 | 4,37× | 1635× |
| FNO_GNN | mse | 3779 | 574 | 2,83 (0,58) | 128 | 2,39 | 0,09 / 7,47 | 3,92× | 1199× |
| FNO_GNN | mae | 3779 | 574 | 3,16 (0,60) | 128 | 2,40 | 0,09 / 7,47 | 3,92× | 1075× |
| GNN_PostBase | mse | 3779 | 574 | 6,32 (0,75) | 64 | 6,61 | 0,15 / 6,94 | 3,90× | 537× |
| GNN_PostBase | mae | 3779 | 574 | 6,36 (0,93) | 64 | 6,62 | 0,15 / 6,94 | 3,90× | 534× |
| FNO_BipartiteGNN | mse | 3779 | 578 | 3,41 (0,35) | 128 | 3,27 | 0,10 / 8,35 | 3,90× | 996× |
| FNO_BipartiteGNN | mae | 3779 | 578 | 3,51 (0,64) | 128 | 3,26 | 0,10 / 8,35 | 3,90× | 968× |

¹ 512 foi o teto da varredura (10,4 GiB). Pode caber mais.

- Speedup sobre a solução = (malha + solução) / (malha + pré-proc. + inferência batch 1), mediana
  por amostra.
- Speedup de inferência pura = solução / inferência batch 1.
- Geometria (253 ms) e leitura do .ans não entram em nenhum dos dois lados.

Leituras:
- **Os modelos em grafo continuam dependendo da malha do FEMM** (`mi_createmesh`, 390 ms), e o
  FNO2d também, no pipeline atual (x_hw). O ganho real é ~4×, não ~1000×: o custo de ponta a ponta
  é dominado por malha (390 ms) + x_hw (403 ms), não pela inferência (2–6 ms).
- A inferência pura é 530–1640× mais rápida que `mi_analyze`. O GNN_PostBase é o mais lento
  (FNO2d base + GNN de 6 camadas × 64).
- No FNO2d com batch 1, o IQR é grande (~75% da mediana): a execução é dominada pela latência de
  lançamento de kernels. No batch máximo, o custo cai para 0,8 ms/amostra.

## 4. Tarefa 4 (O1) — campo no entreferro

Arco em r_m = stator_outer_d/2 + gap/2, com gap = (rotor_inner_d − stator_outer_d)/2
(`valid_designs.csv` não tem coluna de gap). São 1200 pontos em θ = k·0,1°, k = 0…1199.
- **Gabarito e modelos em grafo:** interpolação baricêntrica do B nodal no triângulo que contém
  o ponto (trifinder; nenhum ponto caiu fora da malha).
- **FNO2d:** `interpolate_grid_to_nodes(cell_centered)` direto da grade.
- **Harmônicos:** FFT de B_r com período base de 120° (ordem 7 = fundamental). THD =
  √(Σ_{k=1..100, k≠7} A_k²) / A_7, isto é, todas as ordens exceto DC e a fundamental, incluindo as
  de ranhura.

Gabarito no arco (média / mediana / p95): RMS |B| 0,551 / 0,538 / 0,741 T; amplitude da
fundamental 0,732 / 0,714 / 1,010 T; THD 9,34 / 9,07 / 16,48 %; r_m 38,55 / 38,56 / 40,86 mm;
gap 1,24 / 1,24 / 1,91 mm.

**4a. Erros ao longo do arco** (média / mediana / p95 sobre as 2816 amostras; % = relativo ao
RMS de |B| do gabarito no arco da própria amostra):

| Arch | Loss | L1 B_r (%) | L2 B_r (%) | L1 B_θ (%) | L2 B_θ (%) | L1 B_r (mT) | L1 B_θ (mT) |
|---|---|---|---|---|---|---|---|
| FNO2d | mse | 3,40 / 3,08 / 5,60 | 4,47 / 4,18 / 6,75 | 2,76 / 2,61 / 4,08 | 3,67 / 3,50 / 5,20 | 18,6 / 16,6 / 32,8 | 15,4 / 13,6 / 27,4 |
| FNO2d | mae | 3,11 / 2,72 / 5,73 | 4,10 / 3,75 / 6,69 | 2,42 / 2,24 / 3,81 | 3,29 / 3,08 / 4,92 | 17,1 / 14,9 / 32,9 | 13,6 / 11,9 / 25,7 |
| FNO_GNN | mse | 8,11 / 7,97 / 10,25 | 10,62 / 10,45 / 13,59 | 6,07 / 5,93 / 7,82 | 7,77 / 7,60 / 9,96 | 45,2 / 42,2 / 71,2 | 33,1 / 31,2 / 47,7 |
| FNO_GNN | mae | 7,33 / 7,14 / 9,43 | 9,63 / 9,44 / 12,27 | 5,08 / 4,95 / 6,66 | 6,61 / 6,44 / 8,71 | 40,4 / 37,9 / 61,0 | 27,6 / 26,6 / 37,3 |
| GNN_PostBase | mse | 3,38 / 3,07 / 5,53 | 4,18 / 3,87 / 6,53 | 2,81 / 2,72 / 3,79 | 3,54 / 3,43 / 4,75 | 18,7 / 16,7 / 33,5 | 15,6 / 14,2 / 25,0 |
| GNN_PostBase | mae | **2,99** / 2,60 / 5,58 | **3,58** / 3,20 / 6,31 | **2,25** / 2,10 / 3,45 | **2,83** / 2,66 / 4,28 | 16,5 / 14,4 / 31,7 | 12,6 / 11,2 / 23,1 |
| FNO_BipartiteGNN | mse | 5,21 / 4,77 / 8,44 | 6,45 / 6,00 / 10,24 | 4,25 / 3,80 / 7,15 | 5,23 / 4,75 / 8,51 | 28,7 / 24,8 / 53,5 | 22,7 / 21,6 / 33,1 |
| FNO_BipartiteGNN | mae | 3,89 / 3,64 / 5,95 | 4,86 / 4,60 / 7,21 | 3,23 / 3,11 / 4,54 | 4,05 / 3,92 / 5,62 | 21,3 / 19,6 / 35,0 | 17,9 / 16,3 / 29,0 |

**Harmônicos de B_r** (média / mediana / p95; o viés é a média com sinal):

| Arch | Loss | \|erro rel.\| amplitude fund. (%) | viés amplitude (%) | \|erro fase\| fund. (°) | \|erro THD\| (p.p.) | viés THD (p.p.) |
|---|---|---|---|---|---|---|
| FNO2d | mse | 2,51 / 2,16 / 5,85 | −0,71 | 0,266 / 0,225 / 0,656 | 0,70 / 0,59 / 1,78 | +0,30 |
| FNO2d | mae | 2,56 / 2,20 / 6,22 | −0,62 | 0,272 / 0,230 / 0,687 | 0,56 / 0,49 / 1,32 | +0,20 |
| FNO_GNN | mse | 2,75 / 2,29 / 6,63 | −0,49 | 0,607 / 0,502 / 1,531 | 5,07 / 5,07 / 7,62 | +5,06 |
| FNO_GNN | mae | 3,24 / 2,76 / 7,88 | −1,35 | 0,657 / 0,544 / 1,674 | 3,53 / 3,42 / 6,49 | +3,52 |
| GNN_PostBase | mse | 2,50 / 2,18 / 5,84 | −0,87 | 0,267 / 0,219 / 0,664 | 0,63 / 0,55 / 1,53 | +0,41 |
| GNN_PostBase | mae | 2,51 / 2,15 / 6,15 | −0,23 | 0,262 / 0,220 / 0,649 | 0,49 / 0,40 / 1,20 | +0,18 |
| FNO_BipartiteGNN | mse | 4,15 / 3,73 / 9,36 | −3,74 | 0,389 / 0,317 / 0,984 | 1,54 / 1,41 / 3,48 | +1,39 |
| FNO_BipartiteGNN | mae | 2,49 / 2,12 / 6,16 | +0,05 | 0,434 / 0,371 / 1,045 | 1,35 / 1,31 / 2,70 | +1,27 |

Leituras:
- **No entreferro, o ranking muda em relação ao domínio inteiro.** GNN_PostBase mae é o melhor,
  e o FNO2d fica praticamente empatado com ele. FNO_BipartiteGNN mae, melhor ponto a ponto no
  domínio, é pior que o FNO2d no arco (3,9% contra 3,1% em L1 de B_r). O entreferro é uma
  região de malha fina, mas de campo suave e bem representado pela grade.
- **Fundamental:** erro de amplitude de ~2,5% e de fase < 0,7° (p95) em quase todos os modelos.
  Exceção: Bipartite mse, que subestima a amplitude em 3,7% (viés).
- **As GNNs adicionam conteúdo harmônico espúrio:** a THD sai com viés positivo em todos os
  modelos. No FNO_GNN o viés é +3,5 a +5 p.p. (a THD real é ~9%); no Bipartite, +1,3 p.p.; no
  FNO2d e no GNN_PostBase, ≤ 0,4 p.p. Esse é o "ruído" de alta ordem visível em B_θ na
  figura 4b. Para cogging (que depende das harmônicas de ranhura), esse é um ponto fraco das
  GNNs.

**4b. Figuras** (`figuras/4b_entreferro_<tag>_amostra<idx>.{png,pdf}`; modelos com loss mae).
Amostras escolhidas pelo L1_A por amostra da FNO_BipartiteGNN mae, tomando a amostra com valor
mais próximo de cada percentil:

| Tag | Amostra (índice global, `sample_XXXXXX`) | L1_A Bipartite mae | Fração do ferro > 1,6 T | r_m / gap (mm) |
|---|---|---|---|---|
| P5 | **2275** | 20,1 mT | 0,006% | 38,42 / 1,64 |
| P50 | **2906** | 23,6 mT | 0,026% | 38,87 / 0,97 |
| P95 | **921** | 33,4 mT | 9,12% | 39,56 / 0,78 |
| maior saturação (só T5) | **3690** | 31,5 mT | 17,44% | 36,71 / 0,63 |

Cada figura tem 5 painéis: B_r(θ), B_θ(θ), erro de B_r, erro de B_θ (mesma escala nos dois) e o
espectro de B_r até a ordem 100 (gabarito em barras, modelos em pontos, escala log).

## 5. Tarefa 5 — saturação (só gabarito do FEMM)

|B| é calculado por elemento (curl(A) P1). μ_r efetivo = |B| / (μ0·H(|B|)), com H interpolado
**linearmente** na curva BH de 19 pontos do `[BlockProps]` (a mesma em todos os blocos de ferro,
conferido). O ferro foi identificado pelo material do bloco (`iron_1008`).

**5b. Estatística** (média / mediana / p95 entre as 2816 amostras):

| Limiar | % da área de ferro acima do limiar |
|---|---|
| 1,4 T | 3,59 / 0,44 / 15,06 |
| 1,6 T | 1,97 / 0,022 / 11,34 |
| 1,8 T | 1,16 / 0,005 / 8,75 |

- **A saturação é de cauda.** Na mediana, menos de 0,03% do ferro passa de 1,6 T, mas 5% das
  amostras têm mais de 11%. Pelas figuras 5a, ela se concentra na coroa externa do rotor (culatra
  fina atrás dos ímãs) e nas pontas dos dentes do estator.
- **μ_r efetivo nunca chega a 5000.** Ponderado por área, os quantis são p5 = p25 = 1202,
  p50 = 1660, p75 = 1995, p95 = 2138; a fração de área com μ_r ≥ 5000 é 0. O pico em 1202 é
  artefato da interpolação linear no 1º segmento da curva BH (0 a 0,24 T dá μ constante); ver §6,
  item 3. A cauda saturada desce até μ_r ≈ 5. Figura: `5b_histograma_mu_efetivo` (eixo y log).
- **Correlação de Spearman** entre a fração de ferro > 1,6 T e o L1_A por amostra (n = 2816):
  **FNO2d mae ρ = 0,744**; **FNO_BipartiteGNN mae ρ = 0,485** (p < 1e-160 nos dois). O erro do
  FNO2d é muito mais sensível à saturação: a GNN bipartite (que recebe μ_r e M por elemento)
  absorve parte do efeito. Na dispersão aparecem dois grupos (≈0% e ≈10% de ferro saturado), com
  o FNO2d deslocado para cima no grupo saturado. Figura: `5b_dispersao_saturacao_L1A`.

**5a. Figuras** (`figuras/5a_saturacao_<tag>_amostra<idx>`), para P5, P50, P95 e a amostra de
maior saturação (3690, 17,4% do ferro > 1,6 T):
- |B| por elemento no ferro (escala fixa de 0 a 2,5 T em todas), com os demais materiais em
  cinza e contornos de 1,6 T e 1,8 T (marcados também na barra de cores);
- μ_r efetivo no ferro em escala log (3 a 1e4, fixa), com a referência μ_r = 5000 marcada na barra.

## 6. Inconsistências encontradas

1. **Wrap circular em componentes cartesianas (B1, `interp_mode='cell_centered'`).**
   `interpolate_grid_to_nodes` aplica padding circular de 1 coluna em θ, misturando as colunas 0 e
   W−1. Isso vale para campos escalares e para B_r/B_θ, mas **não para Bx/By**: no corte de 120°
   o vetor está girado de 120°, então Bx(0°) ≠ Bx(120°). Medido em 64 amostras
   (`wrap_check.json`), sobre o gabarito da grade:
   - |Bx,By(col 0) − Bx,By(col W−1)| médio = 0,373 T (0,163 T após girar a coluna 0 de 120°);
   - na faixa de meio pixel junto ao corte, o L1 do piso é **0,131 T, contra 0,012 T no resto**
     (11×); interpolar em componentes polares e voltar ao cartesiano baixa para 0,058 T;
   - essa faixa tem 0,35% da área e contribui com ~0,5 mT do L1 global do piso (12,7 mT), ou seja,
     ~4%.

   Afeta o FNO2d nos nós (os picos em θ = 0°/120° na figura 4b) e a entrada FNO→nós de FNO_GNN,
   FNO_BipartiteGNN e GNN_PostBase **durante o treino** (que a GNN pode aprender a corrigir). A
   verificação V1 testou só campos escalares/periódicos, por isso não pegou. **Não corrigido**
   (mudaria o treino). Correção possível: interpolar em (B_r, B_θ) e girar de volta, ou girar as
   colunas de padding de ±120° antes do `grid_sample`.
2. **|B| do gabarito acima da curva BH.** O |B| P1 por elemento passa do último ponto da curva
   (2,585 T) em 105.862 elementos de ferro, em 2016 das 2816 amostras. O máximo por amostra tem
   mediana de 2,76 T e p95 de 3,10 T. São picos locais em cantos (a solução P1 de primeira ordem
   não é suave nessas singularidades geométricas). μ_eff ali vem de extrapolação linear da curva.
   Afeta pouca área, mas é preciso cuidado ao citar o |B| máximo.
3. **Interpolação da curva BH.** Usamos interpolação linear por partes. O FEMM suaviza a curva
   internamente, então para |B| < 0,24 T o μ_eff daqui é constante (1202) e difere do μ
   diferencial/secante que o FEMM usa. Os quantis baixos de μ_eff são aproximação; os limiares de
   |B| (5b) não dependem disso.
4. **Entrada μ_r = 5000 contra ferro real.** A feature de entrada de todos os modelos (x_hw,
   node_x, elem_x) usa μ_r = 5000 constante para o ferro, mas a solução do FEMM é não linear, e o
   μ_r efetivo fica em 1200–2200 no ferro não saturado (e até ~5 no saturado). Os modelos precisam
   inferir a saturação só da geometria. Isso é coerente com a ρ = 0,74 do FNO2d.
5. **`rotor_phase` em `valid_designs.csv` não é usado.** A coluna varia de 0,006° a 17,1°, mas
   `BLDC_FEMM_Model_Sym120` desenha com `phase=0`. As 50 malhas regeneradas com `phase=0` batem
   bit a bit com o raw. A coluna é metadado sem efeito e não deve ser citada como variável do
   dataset.
6. **Memória de GPU no Windows.** Sem limite, o driver (WDDM) transborda para a memória
   compartilhada do sistema em vez de dar OOM. Na primeira tentativa, o batch 256 do FNO_GNN
   "coube" e ficou ordens de grandeza mais lento. A medição final usa
   `set_per_process_memory_fraction(0,95)`, então o batch máximo é o que cabe em ~15,2 GiB de
   VRAM dedicada. Os picos medidos durante o treino na L40S (§3 do relatório anterior) não são
   comparáveis.
7. **`git_dirty`:** ver §1. O mecanismo explica o flag sem mudança de código, mas falta a
   confirmação na VM.

## Arquivos

Nesta pasta (`docs/bateria_definitiva/pos_bateria/`, versionada): este relatório,
`pos_bateria.json` (todos os números) e `figuras/` (PNG 300 dpi + PDF). Cópia do original em
`data/logs/mesh_ans_138x276_unified_best_mse_mae/pos_bateria/` (gitignored), que tem também os
arquivos intermediários:

| Arquivo | Conteúdo |
|---|---|
| `pos_bateria.json` | **todos os números** (git, eval, timing, figuras, wrap) |
| `eval_pass.json` / `eval_per_sample.npz` | agregados / métricas por amostra (8 runs + piso + arco + saturação) |
| `timing.json` | por amostra e resumos de T3, hardware |
| `figures.json` | amostras escolhidas, Spearman, quantis de μ_eff |
| `git_check.json`, `wrap_check.json` | §1 e §6, item 1 |
| `tabelas.md` | tabelas geradas automaticamente |
| `figuras/` | 9 figuras em PNG (300 dpi) e PDF |
| `*.log` | saída das execuções |
