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

As regiões são **faixas por distância**, com a largura $d$ aberta, e os elementos são
**cortados**: cada ponto de quadratura de §1 é atribuído pela **sua** distância, e não pelo
triângulo.

**Definição**
- $\Gamma$ = união das arestas de interface (`interface_edges`: arestas compartilhadas por
  elementos de materiais diferentes; a borda externa do domínio fica de fora).
- $\mathcal V_c$ = vértices de canto (`corner_nodes`): vértice de fronteira com 2 arestas de
  interface e mudança de direção $> 30°$ (`--ang-canto`), ou com $\ge 3$ arestas de interface,
  ou que toca $\ge 3$ materiais.
- $\Omega_f(d) = \{x : \mathrm{dist}(x,\Gamma) \le d\}$ (fronteira),
  $\Omega_q(d) = \{x : \mathrm{dist}(x,\mathcal V_c) \le d\}$ (quinas),
  $\Omega_{int}(d) = \Omega \setminus \Omega_f(d)$ (interior). O mesmo $d$ vale para as duas.
- dist = distância euclidiana exata no plano (ponto↔segmento, ponto↔ponto).
- Como $\mathcal V_c$ são vértices de $\Gamma$, $\mathrm{dist}_f \le \mathrm{dist}_q$ ponto a
  ponto e $\Omega_q(d) \subseteq \Omega_f(d)$ para todo $d$.

**Distâncias** (`band_distances`, uma vez por amostra e por conjunto de pontos: quadratura da
malha, com xy = baricêntricas · vértices, e overlay da grade, com `grid_mesh_overlay['xy']`;
reutilizadas por todas as runs e pisos):
- $\mathcal V_c$: `cKDTree` (exato).
- $\Gamma$ (`_dist_to_segments`): cada aresta é dividida em sub-segmentos de no máximo
  0,02 mm. A união é o mesmo conjunto, então isso não é densificar. Para cada ponto, calcula
  a distância exata aos sub-segmentos dos $k$ pontos médios mais próximos ($k$ = 16, depois
  64). O resultado é aceito quando o limite inferior dos demais, $|p - m_k| - h/2$, é
  $\ge$ o mínimo. Sobra um fallback `shapely.STRtree.query_nearest`, que não foi acionado na
  amostra conferida.
- Pontos com $\mathrm{dist}_f > d_{max}$ **garantido** ficam com um limite superior, já que só
  caem no bin de excesso.
- Conferido contra o shapely em 200 mil pontos: diferença máxima de 4e-15 mm.
- $\mathrm{dist}_f$ é levado a $\min(\mathrm{dist}_f, \mathrm{dist}_q)$ para que a inclusão
  valha também no arredondamento (violação medida: 0).

**Acumulação** (`BandAccum`), com $e = ||B|_{modelo} - |B|_{FEMM}|$ e peso $w$ do ponto:
- **Histogramas** em $\mathrm{dist}_f$ e em $\mathrm{dist}_q$ de $\Sigma w$, $\Sigma e w$ e
  $\Sigma e^2 w$ (`band_edges`):
  - bins uniformes (`--bin-scale lin`, padrão) ou log entre 0 e `--d-max` (padrão 2 mm), com
    `--n-bins` (padrão 400, ou seja, 0,005 mm);
  - os `--d` são inseridos como bordas exatas;
  - bin $k$ = $\{e_{k-1} < \mathrm{dist} \le e_k\}$, e o último bin é $\{\mathrm{dist} > d_{max}\}$.
  - A soma acumulada dá $\Omega_f(d)$ e $\Omega_q(d)$ em qualquer borda, sem reprocessar.
- **Regiões disjuntas por $d$ da lista**: quinas $\Omega_q$, fronteira sem quinas
  $\Omega_f \setminus \Omega_q$ ($\mathrm{dist}_f \le d < \mathrm{dist}_q$) e interior.
  - São acumuladas **diretamente** pelo código do ponto (`band_points`), com os dois limiares
    no mesmo ponto.
  - Pela inclusão acima, também valem como diferença dos dois histogramas 1-D; o teste
    confere que as duas formas coincidem.
- **Global**: $\Sigma w$, $\Sigma e w$, $\Sigma e^2 w$ e $\Sigma |B|_{FEMM}^2 w$ ($B_{ref}$).

**Saída**
- `dados/regions.json`, com `params` (todos os valores usados), `timing`, `checks` (violação da
  inclusão, pontos no fallback) e, por run (`runs`) e por piso (`floor_surface`):
  - `point_set` (`mesh` ou `overlay`);
  - `B_ref_global` e `total` ($L1_A$/$L2_A$ em %, iguais aos do `eval` nas mesmas amostras);
  - `by_d[d]` com as regiões `quinas`, `fronteira_sem_quinas`, `interior` (disjuntas) e
    `fronteira` ($\Omega_f$ inteira). Cada uma traz $L1_A$ e $L2_A$ normalizados pela área
    **da região**, em % do $B_{ref}$ **global** (e em T), `area_frac` e `L1_contrib_frac`
    (fração de $\int e\,dA$ total).
- `dados/regions_curves.npz`: por chave (run ou `piso_<tipo>`) e campo,
  `<chave>__d` (bordas) e, para `f` ($\Omega_f$) e `q` ($\Omega_q$):
  - `_area_frac`, `_eta` (fração do erro total) e `_L1_pct`/`_L2_pct` da região;
  - as somas cruas `_cum_w`/`_cum_e`/`_cum_e2`.
- `plots/regions_curvas.{png,pdf}`: colunas fronteira e quinas, linhas mse e mae. Em cada
  painel, a fração da área (cinza tracejado) e η por modelo; os $d$ da tabela ficam marcados e
  o eixo $d$ é log.
- `plots/regions_bands_sample_<idx>_d<d>.png` (ponto decimal como `p`, ex. `d0p05`), na 1ª amostra e para cada $d$: as faixas reais
  (buffer de $\Gamma$ e de $\mathcal V_c$ recortados ao domínio, `band_polygons`) sobre a
  malha, com zoom no entreferro.

**Resolução da quadratura** (`tests/check_regions_bands.py`, amostra 5408): erro relativo da
área de $\Omega_f(d)$ e $\Omega_q(d)$ pela soma dos pesos × área exata (shapely: buffer ∩
domínio).

| $d$ [mm] | $\Omega_f$ malha | $\Omega_f$ overlay | $\Omega_q$ malha | $\Omega_q$ overlay |
|---|---|---|---|---|
| 0,05 | −8,2% | +4,4% | −0,10% | +0,11% |
| 0,1 | +0,70% | +4,5% | +0,27% | −0,27% |
| 0,2 | −0,39% | +0,81% | +0,09% | +0,18% |
| 0,5 | −0,10% | +0,48% | −0,05% | −0,01% |

Com $d \le 0{,}1$ mm, $\Omega_f$ passa de 1%: a faixa é mais estreita que os subtriângulos da
quadratura nos elementos grandes ao longo de $\Gamma$ (malha n_sub=2, overlay n_sub=1). O n_sub
não foi aumentado.

**Tempo** (1 processo, só a parte das regiões): 0,93 s/amostra antes (`classify_triangles` +
`RegionAccum`) contra cerca de 30 s depois (7,4 M distâncias exatas por amostra: 3,2 M da
malha e 4,2 M do overlay). As distâncias rodam nos workers do parse.
`regions` completo, 32 amostras do teste, 4 workers (RTX 4080, 32 GB): 3,5 s/amostra de parede antes contra 11,0 s
depois (faixas 29,6 s por amostra no worker; acumulação de 10 runs + 3 pisos 4,3 s no processo
principal). Validação nas mesmas 32 amostras: total e soma das 3 regiões disjuntas = `eval`
(diferença ≤ 1e-7 pp em $L1_A$/$L2_A$, $B_{ref}$ idêntico); curvas monótonas e
$\Omega_q \subseteq \Omega_f$ sem violação.

`classify_triangles` (região fixa por triângulo) continua no módulo só para
`scripts/artigo_v1.py` (mapa F3a e pontos de zoom); `RegionAccum` está comentado.

## 3. Torque (`torque {theta-e,batch,scatter}`, `src/postprocess/torque.py`)

**Referência = `smooth`** (desde 2026-11-10): o $\mathbf B$ suavizado do `.ans` no centróide
de cada elemento, que é o alvo de treino da Elem, passado pela **mesma** fórmula da faixa
usada nas predições.
- O FEMM só gera a malha e o `.ans` das varreduras (`torque.femm_solve`, com cache em
  `data/temp/torque_theta_e{,_batch}/`).
- O T22 do `mo_blockintegral` não é mais referência. Ele ficou fora do caminho padrão: só é
  calculado com `--femm-ref`, que grava a curva `femm` (batch) ou `femm_T22` (theta-e) sem
  entrar em nenhum resumo.
- O `method_check` (gt × T22) foi removido.

**T22, só com `--femm-ref`**: `femm_solve` redesenha o design com o $\theta_e$ pedido,
resolve com profundidade $L$ = stack length e calcula $T_{22}$ = `mo_blockintegral(22)`
sobre os rótulos com $r > r_m = (SOD/2 + RID/2)/2$: coroa do rotor, ímãs e margem externa.
- O rótulo do ar do entreferro fica exatamente em $r_m$, e a comparação estrita o exclui.
- O torque da máquina é $3\,T_{22}$.
- O cache sem T22 (`T22_sector = null`) é ressolvido quando se pede `--femm-ref`.

**Torque pelo campo** (`band_torque`): média do tensor de Maxwell na faixa de ar do
entreferro, com $\mathbf B$ por elemento sobre os elementos de **ar** com centróide em
$r_1 < r_c < r_2$ ($r_1 = SOD/2$, $r_2 = RID/2$):

$$ T = -\,\frac{3\,L}{\mu_0\,(r_2 - r_1)}\sum_{e} r_e\, B_{r,e}\, B_{\theta,e}\, A_e $$

- Unidades SI: $r$ e $r_2 - r_1$ em m, $A_e$ em m², $L$ em m.
- $B_r, B_\theta$ são obtidos de $(B_x, B_y)$ pelo ângulo do centróide.
- O sinal $-$ dá o torque no rotor externo, que é o oposto do torque no conteúdo do círculo.
- O fator 3 converte o setor de 120° para a máquina inteira.

**Campos** (todos pela mesma `band_torque`):
- `smooth` = **referência**;
- `gt` = curl(A) exato P0. A comparação `gt` × `smooth` mede o **efeito da suavização** e é
  reportada à parte, uma vez;
- `pred_<arch>`, a saída do modelo levada aos elementos (`pred_elem_xy`):
  - Elem: a saída por elemento, convertida para $(B_x,B_y)$ no centróide;
  - FNO_SplitGNN: a saída da **N_base** (cópias vértice×material; não a N_aux). Em cada
    elemento entram as 3 cópias do material do elemento (`fields.split_corner_values`, as
    mesmas do P1_mat do `regions`). Cada uma é convertida para $(B_x,B_y)$ no canto do
    elemento, e depois se tira a média (`split_elem_xy`). O
    `parse_torque_sample(..., with_split=True)` monta o layout só quando a Split está entre
    as runs;
  - demais: a saída nodal convertida para $(B_x,B_y)$ no nó e depois a média dos 3 vértices
    (`fields.nodes_to_elem`);
- `floor_<arch>`: o **piso** da arch (`target_elem_xy`), isto é, o alvo de treino dela levado
  à sua reconstrução por elemento pelo mesmo caminho da predição:
  - FNO2d: `y_hw` interpolado nos vértices (`inference.floor_nodes`), depois a média;
  - nós: `node_y` com prioridade de material, depois a média;
  - Elem: `node_y` direto, que é igual ao smooth (piso nulo);
  - Split: `elem_y` por `split_elem_xy`.

**Runs** (`_torque_runs`): as da bateria `bldc` da loss mais a FNO_SplitGNN da bateria
`split` (`find_split_runs`). `--archs` filtra; as do artigo são FNO2d, FNO_GNN,
FNO_BipartiteGNN_Elem e FNO_SplitGNN.

| Modo | Amostras | Saída |
|---|---|---|
| `theta-e` | `--idx`; padrão: 5 amostras do **teste** (1 por faixa de 2 A, `seed=0`, `select_batch_samples`; no split atual: 2263, 2265, 3889, 3940, 5395); `--n-theta` 24 ângulos $\theta_e = 360k/n$ | `theta_e_s<IDX>_<loss>.json`: curvas `T_Nm` (`smooth`, `gt`, `pred_*`, `floor_*`) e `rms_err_vs_ref_mNm` |
| `batch` | 80 amostras do subset estratificadas por corrente (16 por faixa de 2 A, `seed=0`) × 16 ângulos; `--first K` usa só as $K$ primeiras (placeholder) | `theta_e_batch_<loss>.json` (curvas) + `_summary.json` (`summarize_batch`) |
| `scatter` | 8 primeiras amostras de cada chunk do subset, com $\theta_e$ do CSV, sem FEMM | `scatter_<loss>.json`: `stats[pred_*/floor_*/gt]` contra o smooth |

`--femm` resolve as varreduras de novo, e caches com outra profundidade são ressolvidos
(`femm_ref_ok`).

**Métricas das curvas** (`curve_row(th, R, P)`, em mN·m, com R = smooth)
- Ajuste por mínimos quadrados $T(\theta_e) \approx T_0 + A\cos(\theta_e - \varphi)$
  (`fundamental`).
- `rmse` = $\sqrt{\overline{(P-R)^2}}$; `rms_ref`.
- `A_ratio` = $A_P/A_R$; `dphi` = $\varphi_P - \varphi_R$ em $(-180°, 180°]$; `dT0`;
  `peak_ratio`.
- `batch_summary`: medianas, `rel` = $100\,\mathrm{rmse}/\mathrm{rms}_{ref}$ (mediana, p90 e por
  faixa de corrente), média ± desvio de `A_ratio`, `dphi` e `dT0`.
- `summarize_batch(samples)` gera:
  - `summary[arch]`: preds;
  - `floor[arch]`: pisos;
  - `smoothing_effect`: gt × smooth;
  - `per_sample`.
  Ele roda direto sobre as curvas gravadas, sem refazer o campo.
- `scatter_stats(ref, pr, alt)`: RMSE, MAE, viés, p95, máximo, $R^2$ contra o smooth e RMSE
  gt × smooth.

**Conferência do caminho da Split** (`python -m tests.check_split_torque_path [N]`): aplica o
caminho ao gabarito da Split (`elem_y`) em 80 soluções do batch.
- **Elementos com os 3 vértices "normais"** (vizinhos do mesmo material, fora da costura):
  $|\Delta\mathbf B|$ contra o smooth ≤ 2,2e-7 T, o float32 do `elem_y`.
- **Vértices de interface e da costura:** o FEMM dá um valor nodal por elemento, e o N_base
  usa a média. A diferença chega a 0,49 T num elemento, e o torque difere ≤ 0,44 mN·m.
- Por isso o piso da Split é cerca de 0,004% (não exatamente zero).

**Resultados PLACEHOLDER** (2026-11-10; `resultados/BLDC_138x276/placeholder/dados/torque/`)
- **Pisos e Split:** 10 primeiras amostras da seleção × 16 θe.
- **Archs bldc nas 80 amostras:** recalculadas sem rodar nada, a partir do
  `dados/torque/theta_e_batch_<loss>.json` (`recalc80_ref_smooth_<loss>.json`).
- **Erro relativo mediano contra o smooth:**

  | | FNO2d | FNO_GNN | Elem | Split |
  |---|---|---|---|---|
  | mse | 21,9% | 37,7% | 33,7% | 29,3%* |
  | mae | 20,7% | 22,3% | 25,6% | 15,0%* |

  \* Split só nas 10 amostras. Nessas mesmas 10, as outras archs têm, em mse/mae:
  FNO2d 35,7/31,6%, FNO_GNN 60,5/38,2%, Elem 36,4/44,7%.
- **Pisos (10 amostras):** FNO2d 15,9%, FNO_GNN 20,4%, Elem 0, Split 0,004%. A
  reconstrução nodal e a de grade, sozinhas, já perdem 16–20% do torque na faixa.
- **Efeito da suavização** (gt × smooth, 80 amostras): `rel` 3,3% e `A_ratio` 0,972. Por
  faixa de corrente: 5,0% (0–2 A), 3,4% (2–4 A), 3,2% (4–6 A), 3,3% (6–8 A) e 3,0%
  (8–10 A).

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

## 7. Execução com memória controlada (`scripts/run_postprocess.py`)

Script de partida para as long runs na máquina local (32 GB de RAM). Ele só orquestra: chama
as mesmas funções do `postprocess.py` (`regions_setup/state/track/add_floors/add_run/
finalize`, que o subcomando `regions` também usa), sem cálculo próprio. Por ora, a única etapa
é `--steps regions`. As demais entram no dicionário `STEPS`, chamando os `run_*` existentes.

```
python -m scripts.run_postprocess --steps regions --n 10 --workers 2      # placeholder
python -m scripts.run_postprocess --steps regions --all --workers 3 [--max-ram-gb 26]
python -m scripts.run_postprocess --steps regions --all --resume          # continua
```

**Amostras**
- `--n N` (padrão 10) usa as $N$ primeiras amostras do subset, na ordem do split.
- `--all` usa o subset inteiro.
- Os índices usados vão no JSON (`sample_idx`).

**Execução em lotes** (`--chunk`, padrão 8 amostras)
- **Workers** (`--workers`, padrão 2): calculam layout, Ω, Γ, 𝒱_c e distâncias
  (`_regions_sample`).
  - Com `--tasks-per-child`, padrão 1, cada processo é reciclado a cada amostra.
  - O lote seguinte é submetido enquanto o principal processa o atual, então há no máximo 2
    lotes na RAM.
- **Processo principal:** calcula os pisos e depois cada run, **um modelo de cada vez** na
  GPU sobre o lote. A predição não é guardada: vai direto para `BandAccum.add`.
  - A ordem das amostras em cada acumulador é a mesma do `regions`.
- **Fim do lote:**
  - `gc.collect` e `torch.cuda.empty_cache`;
  - checkpoint (pickle atômico de `dados/regions_checkpoint.pkl`);
  - log em `dados/run_postprocess.log`, com o tempo do lote, a espera pelos workers, a
    estimativa do restante e os picos de RAM (principal + workers, RSS amostrado a cada
    0,25 s) e de VRAM.

**Checkpoint e retomada**
- `--resume` confere a assinatura (parâmetros, índices, `--chunk`, runs/épocas) e continua do
  lote seguinte.
- O JSON final é sempre montado das somas acumuladas.
- `--max-ram-gb`: se o pico do lote passar do limite, o lote termina, o checkpoint é gravado e
  o script para.
- `--stop-after K` interrompe após $K$ lotes; serve para testar a retomada.
- O checkpoint é apagado no fim, a menos que se use `--keep-checkpoint`.

**Saída**
- `--out`. O padrão é `resultados/BLDC_138x276/placeholder/` quando $N$ < subset, com
  `placeholder: true` no JSON; com `--all`, é `resultados/BLDC_138x276/`.
- O `regions.json` traz ainda:
  - `execution`: lotes, picos, área de Ω por amostra × exata e as checagens;
  - `timing.n_workers`, `tasks_per_child` e `chunk`.

**Validação (10 amostras, 2 workers, 2026-11-09)**
- **Retomada:** parar após o 1º lote e retomar dá o mesmo resultado que a execução contínua e
  que o `regions` antigo.
  - Pisos e FNO2d saem idênticos bit a bit.
  - As GNNs diferem em até 1e-7 pp. A causa é que a própria inferência delas na GPU não é
    determinística (somas atômicas em float32: cerca de 1e-6 T entre 3 inferências
    iguais).
- **Regiões:** a soma das 3 regiões disjuntas reproduz o total com erro de 7e-12 pp, e
  máx(dist_f − dist_q) = 1e-16 mm.
- **Área de Ω** × exata ($(r_{ro}^2 - r_{si}^2)\,\theta/2$): −0,005%.

**Custo (10 amostras, 2 workers)**
- **Tempo:** 20,5 s/amostra de parede, dominado pelos workers (cerca de 25 s de distâncias +
  parse por amostra e worker). O processo principal leva cerca de 4 s por amostra.
- **Memória:** pico de RAM de 5,2 GiB (principal 2,9 + workers 2,9) e de VRAM de 0,32 GiB.
