# Avaliação e pós-processamento

O executável é `python -m scripts.postprocess <subcomando>` e o código fica em
`src/postprocess/`. Os dados saem em `resultados/BLDC_138x276/dados/` (versionado) e as
figuras em `resultados/BLDC_138x276/plots/` (fora do git). Tudo é gerado a partir do
**raw**, com parse na hora (`samples.parse_smooth`), sem usar os chunks.

**Runs avaliadas** (`inference.find_runs`/`load_set`): uma por (arch, loss) em
`data/logs/BLDC_138x276/<arch>/`, com o `best.pth`. `load_model` reconstrói a arch pelo
`config.json` e as stats de normalização. `predict` faz a codificação de entrada, o forward e
a decodificação da saída. No FNO2d, a grade de saída também é interpolada nos nós.

**Amostras** (`samples.samples_with_subset`): os chunks de um subset do split da run
FNO2d mse (`split.json`, conferido contra `split_chunk_paths`):
- `test` (padrão): 19 chunks, 608 amostras, nunca vistos no treino;
- `val`: 56 chunks, 1792 amostras, os que escolheram o `best.pth` e a parada; só com
  `--subset val`.

`eval`, `regions`, `saturation`, `timing` e `torque` aceitam `--subset {test,val}`; `figures`
e `surface --pct` usam o subset gravado no `.npz` do `eval`. `N` = as $N$ primeiras amostras
do subset.

---

## 1. Erro de superfície (`eval`) — métrica de comparação entre modelos

Compara a superfície discreta de cada modelo com a do FEMM, no domínio da malha
$\Omega$ (setor $r\in[28{,}5;46{,}5]$ mm, $\theta\in[0°,120°]$).

**Superfície do FEMM**: $\mathbf B$ suavizado **P1 descontínuo**, com os valores `b1`, `b2`
por vértice de cada triângulo (`dados.md` §2.2), interpolados linearmente nas componentes
cartesianas.

**Superfície do modelo**, por arch (`scripts/postprocess.py::_surface_points`):

| Arch | Tipo | Valor num ponto $\mathbf x$ | Integração |
|---|---|---|---|
| FNO2d | `grade` | constante na célula da grade que contém $\mathbf x$; célula = quadrilátero de lados retos com os 4 cantos polares | interseção exata célula ∩ triângulo (`metrics.grid_mesh_overlay`) |
| FNO_GNN, GNN_PostBase, FNO_BipartiteGNN | `P1_nos` | interpolação linear no triângulo das componentes **polares** nodais da saída, depois módulo (`surface_values_p1`) | na malha do FEMM (`mesh_surface_quadrature`) |

A superfície `P1_nos` (modelos nodais e o piso `P1_nos`) interpola as componentes polares
$(B_r, B_\theta)$, cada nó na base do seu ângulo, enquanto a superfície do FEMM interpola as
cartesianas `b1`, `b2`. A base polar gira dentro do triângulo, então as duas interpolações não
coincidem exatamente; o efeito no $L1_A$/$L2_A$ não foi medido. O mesmo vale para
`fields.elem_to_nodes` (média componente a componente, em polar, dos elementos vizinhos).
| FNO_BipartiteGNN_Elem | `P0_elem` | constante no elemento | na malha do FEMM |

**Definições** (`metrics.Accum.add_surface`, `report`), com $e(\mathbf x) = \big||\mathbf B|_{mod}(\mathbf x) - |\mathbf B|_{FEMM}(\mathbf x)\big|$ [T]:

$$ L1_A = \frac{\int_\Omega e\,dA}{\int_\Omega dA},\qquad L2_A = \sqrt{\frac{\int_\Omega e^2\,dA}{\int_\Omega dA}},\qquad B_{ref} = \sqrt{\frac{\int_\Omega |\mathbf B|^2_{FEMM}\,dA}{\int_\Omega dA}} $$

Em % de $B_{ref}$. As integrais são acumuladas sobre todos os pontos de todas as amostras
do subset (não é média de médias). $B_{ref}$ é o mesmo para todos os modelos.

**Quadratura**:
- Malha × malha (`mesh_surface_quadrature`, `n_sub=2`): cada triângulo é dividido em
  $4^2 = 16$ subtriângulos, com 3 pontos médios de aresta em cada → $K = 48$ pontos de peso
  $A_e/K$.
- Grade × malha (`grid_mesh_overlay`, `n_sub=1`): candidatos célula–triângulo pela caixa
  $(r,\theta)$ do triângulo, com folga para a flecha $L^2/(8r)$ da aresta reta.
  - Interseção por `shapely`; pedaços com área $\le 10^{-14}$ são descartados.
  - Cada pedaço é triangulado em leque e cada subtriângulo dividido em 4, com 3 pontos médios
    de peso área/3.
  - O valor do FEMM em cada ponto vem das baricêntricas do triângulo.

**Pisos** (`_floor_points`): o próprio gabarito em cada representação. Medem o erro mínimo
de cada tipo de superfície contra o P1 descontínuo do FEMM.

| Piso | Superfície |
|---|---|
| `grade` | $|y_{hw}|$ constante na célula |
| `P1_nos` | P1 dos `node_y` do layout de vértices |
| `P0_elem` | `node_y` da Elem (suavizado no centróide), constante no elemento |

O piso P1 não é limite inferior rigoroso.

**Erro nodal (secundário)**: $e$ nos nós contra o `node_y` do layout de vértices
(`Accum.add`), com peso = área dual.
- FNO2d: grade interpolada nos nós.
- Elem: levada aos nós por `fields.elem_to_nodes`, que faz a média, componente a componente,
  dos elementos incidentes do material do nó.
- Piso: `y_hw` do gabarito interpolado nos nós (`inference.floor_nodes`).
- Saídas: $L1_A$, $L2_A$, $\mathrm{MAE}_{pt} = \overline e$ e $L2_{pt} = \sqrt{\overline{e^2}}$.

**Saída**:
- `dados/eval_smooth.json`: por run e por piso, por subset e total; inclui o `nodal` e a
  `mesh_stats`.
- `dados/eval_smooth_per_sample.npz`: $L1_A$ e $L2_A$ por amostra, `nodal_*`, `n_nodes` e
  `n_elems`.

## 2. Regiões (`regions`, `src/postprocess/regions.py`)

Cada triângulo do FEMM é classificado pela prioridade **canto > interface > interior**
(`classify_triangles`). As distâncias são exatas (`shapely.dwithin`).
- **Interface**: a até $d_{int}$ de uma aresta entre elementos de materiais diferentes
  (`interface_edges`).
- **Canto**: a até $d_{canto}$ de um vértice de canto. Vértice de canto (`corner_nodes`) é um
  vértice de fronteira com 2 arestas de interface e mudança de direção $> 30°$, ou com
  $\ge 3$ arestas de interface, ou que toca $\ge 3$ materiais.
- Padrões: $d_{int} = \Delta r = 18/138$ mm, $d_{canto} = 2\Delta r$, ângulo 30°.

Os pontos de quadratura de §1 são acumulados por região do triângulo (`RegionAccum`). Para
cada região são reportados:
- $L1_A$ e $L2_A$ normalizados pela área **da região**, em % do $B_{ref}$ **global**;
- a fração da área;
- a fração de $\int e\,dA$.

Saída: `dados/regions.json` e `plots/regions_sample_*.png`.

## 3. Torque (`torque {theta-e,batch,scatter}`, `src/postprocess/torque.py`)

**Referência FEMM**: `torque.femm_solve` redesenha o design com o $\theta_e$ pedido, resolve
com profundidade $L$ = stack length e calcula
$T_{22}$ = `mo_blockintegral(22)` (torque pelo tensor de Maxwell ponderado) sobre os rótulos
com $r > r_m = (SOD/2 + RID/2)/2$: coroa do rotor, ímãs e margem externa. O rótulo do ar do
entreferro fica exatamente em $r = (RID + SOD)/4 = r_m$; a comparação estrita o exclui
porque rótulo e $r_m$ saem da mesma conta (raio relido do `.ans` igual a $r_m$ em 100
amostras conferidas). O torque da máquina é $3\,T_{22}$.

**Torque pelo campo** (`band_torque`): média do tensor de Maxwell na faixa de ar do
entreferro, com $\mathbf B$ por elemento sobre os elementos de **ar** com centróide em
$r_1 < r_c < r_2$ ($r_1 = SOD/2$, $r_2 = RID/2$):

$$ T = -\,\frac{3\,L}{\mu_0\,(r_2 - r_1)}\sum_{e} r_e\, B_{r,e}\, B_{\theta,e}\, A_e $$

- Unidades SI: $r$ e $r_2 - r_1$ em m, $A_e$ em m², $L$ em m.
- $B_r, B_\theta$ são obtidos de $(B_x, B_y)$ pelo ângulo do centróide.
- O sinal $-$ dá o torque no rotor externo, que é o oposto do torque no conteúdo do círculo.
- O fator 3 converte o setor de 120° para a máquina inteira.

**Campos usados**:
- `gt` = curl(A) exato P0;
- `smooth` = $\mathbf B$ suavizado no centróide;
- `pred_<arch>`:
  - Elem: a saída por elemento, convertida para $(B_x,B_y)$ no centróide;
  - demais: a saída nodal convertida para $(B_x,B_y)$ no nó e depois a média dos 3 vértices
    (`fields.nodes_to_elem`).

| Modo | Amostras | Saída |
|---|---|---|
| `theta-e` | `--idx`; padrão: 5 amostras do conjunto de **teste** do `split.json` da run de referência, 1 por faixa de 2 A de corrente, `seed=0` (`select_batch_samples`; no split atual: 2263, 2265, 3889, 3940, 5395); `--n-theta` 24 ângulos $\theta_e = 360k/n$ | `theta_e_s<IDX>_<loss>.json`: curvas e RMS de cada uma contra o FEMM e contra o `gt` |
| `batch` | 80 amostras do subset estratificadas por corrente (16 por faixa de 2 A, `seed=0`) × 16 ângulos | `theta_e_batch_<loss>{,_summary}.json` |
| `scatter` | 8 primeiras amostras de cada chunk do subset, com $\theta_e$ do CSV, sem FEMM | `scatter_<loss>.json` |

As soluções FEMM ficam em cache em `data/temp/torque_theta_e{,_batch}/`. `--femm` resolve
de novo, e caches com outra profundidade são descartados (`femm_ref_ok`).

**Métricas das curvas** (`curve_row`, em mN·m):
- Ajuste por mínimos quadrados $T(\theta_e) \approx T_0 + A\cos(\theta_e - \varphi)$
  (`fundamental`).
- `rmse` = $\sqrt{\overline{(P-F)^2}}$; `rms_femm`; `rmse_gt_femm` (erro do próprio método
  da faixa).
- `A_ratio` = $A_P/A_F$; `dphi` = $\varphi_P - \varphi_F$ em $(-180°, 180°]$; `dT0`;
  `peak_ratio` = amplitude pico a pico de $P$ / de $F$.
- `batch_summary`: medianas e `rel` = $100\,\mathrm{rmse}/\mathrm{rms}_{femm}$ por faixa de
  corrente.
- `scatter_stats`: RMSE, MAE, viés, p95, máximo, $R^2 = 1 - \sum e^2/\sum(T_{gt} - \overline{T_{gt}})^2$
  e RMSE do `smooth` contra o `gt`.

## 4. Saturação (`saturation`, `src/postprocess/saturation.py`)

Só o gabarito, sem modelos.
- **Curva B-H**: blocos do `.ans` com `BHPoints > 0` (`parse_bh_curves`). O código confere
  que a curva é igual em todas as amostras.
- **Interpolação $H(B)$** (`FemmBH`, porte de `CMMaterialProp::GetSlopes/GetH`):
  - inclinações $s_i = dH/dB$ por um sistema tridiagonal de spline cúbica com condição
    natural nas pontas;
  - se algum trecho cúbico não for monótono, os pontos internos são substituídos pela média
    móvel de 3 pontos (até 100 vezes);
  - Hermite cúbico entre os pontos;
  - extrapolação linear com $s_{n}$ acima do último ponto.
- **Ponto de operação**: $|\mathbf B|$ suavizado no centróide de cada elemento de aço
  (`steel_operating_points`), e
  $$ \mu_{eff} = \frac{|\mathbf B|}{\mu_0\,H(|\mathbf B|)} $$
- Percentis p5/p50/p95 ponderados pela área (`weighted_percentile`) e histograma de 0 a 2,8 T
  com 140 classes.
- **Joelho**: o menor $B$ acima do máximo de $\mu_{eff}$ em que $\mu_{eff} = f\cdot\max$
  (padrão $f = 0{,}5$).
- `--check-femm` compara $\mu_{eff}$ com o $\mu$ do pós-processador do FEMM
  (`mo_getpointvalues`).
- Saída: `dados/saturation.json` e `plots/saturacao_*`.

## 5. Custo computacional (`timing`, `src/postprocess/timing.py`; Windows com FEMM e GPU)

**FEMM** (`femm_solve_timed`): para 50 amostras igualmente espaçadas do conjunto do `eval`,
cronometra à parte `openfemm`, a geometria, `mi_createmesh` (`mesh`), `mi_analyze` (`solve`)
e `closefemm`. A comparação usa só `solve`. O `check_mesh_vs_raw` confere que a malha é igual
à do raw e mede $|\Delta A|$. O tempo `solve` é o do `mi_analyze`; se ele refaz a malha,
`solve` = malha + solução. [VERIFICAR] se o `mi_analyze` remalha
(`tests/check_analyze_remesh.py`, ver lista).

**Pré-processamento** (`preprocess_timed`): do `.ans` resolvido até as entradas, por etapa.
São as mesmas funções dos parsers, sem o gabarito; `check_preprocess` confere igualdade
exata.
- Etapas: `read_ans` (`_parse_solution` sobre o `.ans` não comprimido), `materials`, `x_hw`,
  `node_pos`, `graph_topology`, `v1_graph`, `bip_graph` e `elem_graph`.
- O custo de cada arch é a soma das etapas de `ARCH_STAGES[arch]` mais `encode_timed`
  (CPU → GPU e z-score).

**Inferência** (`time_batch`): `best.pth`, `torch.inference_mode`.
- Mede o forward mais a desnormalização (e a interpolação grade → nós no FNO2d).
- Lote 1 com a amostra de nº de nós mediano entre 32 do 1º chunk do subset, e lotes
  $2, 4, \ldots$ até OOM ou 512.
- 10 aquecimentos e 100 repetições com `cuda.synchronize`; mediana e IQR.
- Limite de 95% da VRAM.

**Acelerações** (`_timing_table`, medianas em ms por amostra):

$$ \text{acel}_1 = \frac{t_{solve}}{t_{pre} + t_{inf,1}},\quad \text{acel}^{pura}_1 = \frac{t_{solve}}{t_{inf,1}},\quad
\text{acel}_{max} = \frac{t_{solve}}{t_{pre} + t_{inf,max}/b_{max}},\quad \text{acel}^{pura}_{max} = \frac{t_{solve}}{t_{inf,max}/b_{max}} $$

Saída: `dados/timing.json`, com hardware, `femm_preproc`, `inference` e `table`.

## 6. Outros subcomandos

| Subcomando | O que faz | Saída |
|---|---|---|
| `figures [--data] [--out]` | curvas de treino (`mae_graph`; FNO2d `mae_hw`); barras de $L1_A$/$L2_A$ com o piso de cada representação; $|\mathbf B|$ nos nós dos 5 modelos mae nas amostras P50/P95 do $L1_A$ de `FNO_BipartiteGNN_mae` (setor e zoom no entreferro) | `plots/1_curvas_treino`, `2_barras_erro`, `3_campo_*`; `dados/figures_smooth.json` |
| `surface IDX [LOSS]` / `surface --pct P [LOSS]` | §1 em 1 amostra (ou na amostra do percentil P do eval) e mapa da densidade do erro por triângulo $\sum e\,w/\sum w$ (`surface_error_per_triangle`), com o piso da grade e os contornos de material | `dados/surface_error/sample_*.{json,npz}`, `plots/surface_error/*.png` |
| `train-table` | tabela das 10 runs só a partir dos logs (`config.json`, `run_summary.json`, `epochs.csv`): parâmetros, pico de GPU, medianas de tempo por época, melhor/última época, motivo da parada, hardware (`device`, `gpu_name`, versões de torch/CUDA/cuDNN, `hostname`) | `dados/train_table.json` |
