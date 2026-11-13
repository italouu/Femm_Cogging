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

As regiões são **faixas por distância**, de largura $d$, e os elementos são **cortados**:
cada ponto de quadratura de §1 é atribuído pela **sua** distância, sem subdivisão adicional.

**Domínio Ω = só o motor** (`omega_mask`):
- O que sai: os elementos dos rótulos `vacuum` cuja **posição do rótulo** tem
  $r < $ stator_inner/2 (ar do furo interno) ou $r > $ rotor_outer/2 (ar externo ao back
  iron), com os raios da linha do CSV de cada amostra.
- O que fica: o entreferro e o ar das ranhuras.
- Pontos fora de Ω não entram em nenhuma região, nem no total, nem no $B_{ref}$.
- No teste: rótulos 40/41 (interno) e 42/43 (externo), com 91,9 e 143,8 mm² excluídos. A área
  de Ω contra a exata $(r_{ro}^2 - r_{si}^2)\,\theta/2$ difere em −0,005%.

**Definição**
- $\Gamma$ = arestas de interface (`interface_edges`) e $\mathcal V_c$ = vértices de canto
  (`corner_nodes`: mudança de direção $> 30°$, ramificação ou $\ge 3$ materiais). As duas são
  calculadas na malha **inteira**.
- $\Omega_f(d) = \{x \in \Omega : \mathrm{dist}(x,\Gamma) \le d\}$,
  $\Omega_q(d) = \{x \in \Omega : \mathrm{dist}(x,\mathcal V_c) \le d\}$ e
  $\Omega_{int}(d) = \Omega \setminus \Omega_f(d)$.
- Regiões disjuntas: **quinas** $\Omega_q$, **fronteira sem quinas**
  $\Omega_f \setminus \Omega_q$ e **interior**; a **fronteira** é $\Omega_f$ inteira.
- **$d$ padrão** = $R_{in}\,\theta_{setor}/N_A$ = 0,2163 mm, o passo angular da grade no menor
  raio. `--d` aceita outros valores.
- **Costura periódica de 120°** (`SeamGeometry`): os segmentos de Γ e os cantos a até $d_{max}$
  das retas θ = 0 e θ = 120° são copiados, girados de ±120°. Um ponto perto de 0° enxerga a
  interface perto de 120° pelo outro lado. O custo é cerca de 10% a mais nas distâncias.
- dist = euclidiana exata (`_dist_to_segments` com `cKDTree`, conferida contra o shapely com
  diferença de 4e-15 mm; fallback shapely nos raros pontos sem prova de exatidão).
  $\mathrm{dist}_f = \min(\mathrm{dist}_f, \mathrm{dist}_q)$ garante $\Omega_q \subseteq \Omega_f$.

**Acumulação** (`BandAccum`), com $e = ||B|_{modelo} - |B|_{FEMM}|$ e peso $w$ do ponto:
- **Histogramas** em $\mathrm{dist}_f$ e $\mathrm{dist}_q$ de $\Sigma w$, $\Sigma e w$ e
  $\Sigma e^2 w$: 400 bins até `--d-max` 2 mm, com os $d$ inseridos como bordas. Eles dão as
  curvas em qualquer $d$ sem reprocessar.
- **Regiões disjuntas** acumuladas diretamente pelo código do ponto (`band_points`).
- **Global:** $\Sigma w$, $\Sigma e w$, $\Sigma e^2 w$ e $\Sigma |B|_{FEMM}^2 w$, que dá o
  $B_{ref}$ **em Ω**.

**Runs e pisos**
- Runs: FNO2d, FNO_GNN e FNO_BipartiteGNN_Elem da bateria `bldc`, mais FNO_SplitGNN da
  bateria `split`, × mse/mae, sempre com o `best.pth`.
- Pisos: `grade`, `P1_nos`, `P0_elem` e `P1_mat`; este último é a N_base do gabarito,
  `fields.split_corner_values`.
- Superfície da Split = P1 por material (`_split_points`).

**Saída**
- `dados/regions.json`:
  - `params`, `omega` (rótulos, áreas por anel × exata), `timing`, `checks`, `sample_idx`;
  - `execution` (lotes, picos, área de Ω por amostra), quando rodado pelo
    `scripts/run_postprocess.py`;
  - por run e por piso: `B_ref_global` (em Ω), `total` e `by_d[d][região]` com `L1_area_pct`,
    `L2_area_pct` (normalizados pela área da região, em % do $B_{ref}$), `area_frac` e
    `L1_contrib_frac` (η, a fração de $\int e\,dA$).
- `dados/regions_curves.npz` e `plots/regions_curvas.{png,pdf}`: curvas acumuladas em $d$.
- `plots/regions_map_sample_<idx>_d<d>.png`: verificação numa amostra. Os subtriângulos da
  quadratura são pintados pela região (quina vermelho, fronteira sem quinas verde, interior
  azul, fora de Ω cinza), com Γ e $\mathcal V_c$ em preto. Há um painel do setor e um zoom
  num canto da ranhura.
- Junto, se rodado com saturação: `dados/saturation_error.json` (§4.1).

**Validação**
- Com a máscara desligada, o total reproduz o `eval` (32 amostras, ≤ 4e-8 pp).
- No resultado final (608 amostras):
  - a soma das 3 regiões disjuntas reproduz o total com erro de 7,7e-12 pp;
  - máx($\mathrm{dist}_f - \mathrm{dist}_q$) = 4,4e-16 mm;
  - 8857 pontos foram pelo fallback shapely (exato);
  - os resultados por lote batem com uma execução contínua; as GNNs diferem em até 1e-7 pp,
    pelo não determinismo da inferência na GPU.

**Resolução da quadratura** (amostra 5408, nível 0): no $d$ padrão, a área de $\Omega_f$ tem
erro de −0,03% na malha e −0,5% no overlay. Em $d \le 0{,}1$ mm passa de 1% na malha (−8,2% em
0,05 mm); o n_sub não foi aumentado.

**Resultados** (2026-11-10; 608 amostras de teste, $d$ = 0,2163 mm, $B_{ref}$ em Ω =
0,6829 T): $L1_A$ da região em % do $B_{ref}$ [fração da área | η].

| run / piso | quinas | fronteira sem quinas | interior | total L1 / L2 |
|---|---|---|---|---|
| FNO2d mse | 20,79 [0,018 \| 0,071] | 12,48 [0,193 \| 0,465] | 3,03 [0,790 \| 0,464] | 5,17 / 17,00 |
| FNO2d mae | 20,55 [0,018 \| 0,072] | 12,32 [0,193 \| 0,472] | 2,90 [0,790 \| 0,456] | 5,03 / 16,97 |
| FNO_GNN mse | 10,27 [0,018 \| 0,028] | 15,40 [0,193 \| 0,466] | 4,10 [0,789 \| 0,506] | 6,39 / 15,79 |
| FNO_GNN mae | 10,22 [0,018 \| 0,031] | 14,92 [0,193 \| 0,491] | 3,56 [0,789 \| 0,478] | 5,88 / 15,58 |
| Elem mse | 6,72 [0,018 \| 0,024] | 5,15 [0,193 \| 0,202] | 4,84 [0,789 \| 0,774] | 4,93 / 7,02 |
| Elem mae | 6,52 [0,018 \| 0,027] | 4,46 [0,193 \| 0,203] | 4,14 [0,789 \| 0,770] | 4,24 / 6,34 |
| Split mse | 5,32 [0,018 \| 0,028] | 3,62 [0,193 \| 0,212] | 3,18 [0,789 \| 0,759] | 3,30 / 4,62 |
| Split mae | 4,94 [0,018 \| 0,031] | 3,12 [0,193 \| 0,214] | 2,69 [0,789 \| 0,754] | 2,81 / 4,17 |
| piso grade | 19,59 [0,018 \| 0,099] | 10,77 [0,193 \| 0,591] | 1,38 [0,790 \| 0,310] | 3,51 / 16,62 |
| piso P1_nos | 4,74 [0,018 \| 0,029] | 12,09 [0,193 \| 0,811] | 0,58 [0,789 \| 0,160] | 2,88 / 15,03 |
| piso P0_elem | 1,69 [0,018 \| 0,013] | 1,74 [0,193 \| 0,149] | 2,39 [0,789 \| 0,837] | 2,25 / 4,16 |
| piso P1_mat | 0,01 [0,018 \| 0,005] | 0,01 [0,193 \| 0,105] | 0,02 [0,789 \| 0,890] | 0,02 / 0,05 |

O que a tabela mostra:
- **Representações contínuas (grade e P1 nos nós):** concentram o erro na fronteira: 47–54%
  de η em 21% da área.
- **Representações por material (Elem e Split):** o erro fica no interior (75–77% de η).
- A Tabela 1 em Ω está em `artigo_v1/tabelas/tab_main_omega.tex` e `dados/tab_main_omega.md`.

**Tempo** (`scripts/run_postprocess.py`, 2 workers, lotes de 8; RTX 4080, 32 GB):
- cerca de 18 s/amostra de parede, ou 3 h para as 608 amostras (a run levou 7,1 h porque o PC
  ficou suspenso cerca de 4 h);
- pico de RAM de 5,7 GiB e de VRAM de 0,33 GiB.

`classify_triangles` (região fixa por triângulo) fica no módulo só para `scripts/artigo_v1.py`
(mapa F3a). `band_distances`, `band_polygons` e `mesh_domain` ficam para
`tests/check_regions_bands.py`.

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
- Por isso o piso da Split é cerca de 0,001% (não exatamente zero).

**Resultados** (2026-11-10)
- **Seleção:** 80 amostras de teste estratificadas por corrente × 16 θe, só com o cache do
  FEMM (`--cache-only`, nenhum ângulo pulado); 4 archs e pisos.
- **Arquivos:** `dados/torque/theta_e_batch_<loss>{,_summary}.json`, com os anteriores
  (referência T22) em `dados/torque/_antes_ref_smooth/`.
- **Tabelas:** `artigo_v1/tabelas/tab_torque_smooth.tex` e `dados/torque/tab_torque_smooth.md`.
- **Recálculo das 5 archs `bldc`** com a referência smooth, a partir do batch anterior:
  `dados/torque/recalc_ref_smooth_bldc_<loss>.json`.

Erro relativo mediano (%) contra o smooth, e por faixa de corrente:

| | perda | rel. | p90 | A_ratio | 0–2 A | 2–4 A | 4–6 A | 6–8 A | 8–10 A |
|---|---|---|---|---|---|---|---|---|---|
| FNO2d | MSE | 21,9 | 52,5 | 1,123 | 41,9 | 29,3 | 23,8 | 19,3 | 16,1 |
| FNO2d | MAE | 20,7 | 58,6 | 1,103 | 52,8 | 26,9 | 14,9 | 13,1 | 11,3 |
| FNO_GNN | MSE | 37,7 | 85,0 | 1,188 | 84,1 | 55,4 | 26,7 | 31,6 | 27,2 |
| FNO_GNN | MAE | 22,3 | 55,8 | 1,149 | 36,5 | 38,2 | 19,7 | 20,4 | 18,3 |
| Elem | MSE | 33,7 | 79,0 | 0,840 | 72,9 | 47,3 | 38,7 | 30,0 | 28,0 |
| Elem | MAE | 25,6 | 100,2 | 0,950 | 96,0 | 41,4 | 24,8 | 20,5 | 19,2 |
| Split | MSE | 22,3 | 49,6 | 0,847 | 50,2 | 20,9 | 22,5 | 20,0 | 19,1 |
| Split | MAE | **13,9** | 30,5 | 0,933 | 28,7 | 13,8 | 12,2 | 13,3 | 11,9 |
| piso FNO2d (grade) | — | 7,4 | 38,3 | 1,094 | 15,5 | 9,9 | 7,3 | 5,3 | 7,3 |
| piso FNO_GNN (nós) | — | 18,3 | 41,6 | 1,175 | 22,5 | 19,5 | 17,0 | 21,3 | 16,8 |
| piso Elem | — | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| piso Split | — | 0,001 | 0,009 | 1,000 | 0,004 | 0,003 | 0,001 | 0,001 | 0,001 |
| suavização (gt × smooth) | — | 3,3 | 6,2 | 0,972 | 5,0 | 3,4 | 3,2 | 3,3 | 3,0 |

O que a tabela mostra:
- **Pisos:** a reconstrução nodal com prioridade de material perde cerca de 18% do torque na
  faixa só pela representação, e a grade cerca de 7%. Elem e Split não perdem nada.
- **Split MAE:** é a melhor em todas as faixas de corrente.
- **Corrente baixa:** o erro relativo cresce, porque o torque é menor.
- **Custo:** 68–71 min por loss com 2 processos de parse; pico de RAM de 4,0 GiB e de VRAM de
  2,4 GB.

### 3.1 Torque do FNO_SplitGNN pelo método híbrido (fluxo concatenado + cogging por simetria)

**Método adotado para o torque do Split** (2026-11-11). Script:
`python -m tests.check_split_torque_fluxlinkage [IDX ...] [--n 10] [--run DIR]` (sem FEMM; usa o
cache `data/temp/torque_theta_e_batch/`).

**Por que não só o tensor de Maxwell.** O fluxo do Split é bom ponto a ponto (MAE ~9 mT no
entreferro, viés ~0), mas `band_torque` amplifica o erro:
- depende do produto $B_r B_\theta$ e é dominado pelo termo $B_r\,\Delta B_\theta$ ($|B_r| \approx 0{,}39$ T
  contra $|B_\theta| \approx 0{,}14$ T); o erro de $B_\theta$ do Split é coerente com o sinal de $B_r$
  (correlação +0,44 na amostra 2252), então soma em vez de cancelar;
- o torque líquido é só ~26% da soma das contribuições locais em módulo
  ($\int B_r B_\theta\,d\theta / \int |B_r B_\theta|\,d\theta \approx -0{,}26$), o que amplifica o erro relativo ~4×.

**Forma da curva.** Com o rotor fixo e $|I|$ fixo, $T(\theta_e) = T_0 + A\cos(\theta_e - \varphi)$ explica o GT
com resíduo mediano de 0,15% do rms (52 amostras); a 2ª harmônica (relutância) é 0,12% da
fundamental ($L_d \approx L_q$, ímãs de superfície). $T_0$ = cogging na posição do rotor. O erro do
Split por Maxwell está nos parâmetros, não na forma: $|\Delta T_0|$ ~11% e $\Delta A$ ~−7% (mediana).

**Passos:**
1. **A nodal a partir do B do Split:** mínimos quadrados P1,
   $\min \sum_e S_e \lVert \nabla A_e - (-B_y, B_x)_e \rVert^2$ (xy em mm, B em T, A em Wb/m; 1 nó fixo, a
   constante cancela em ψ). B do Split por elemento = média das 3 cópias N_base do material do
   elemento (o mesmo do torque). Também remove a parte não rotacional do B previsto.
   Conferência: com o B exato curl(A) do GT, reproduz o A do FEMM (torque idêntico).
2. **Fluxo concatenado** por circuito, como o FEMM: $\psi_c = L\,N/S_c \int_{S_c} A\,dS$; por fase,
   $\psi_{fase} = \sum_k s_k\,(\psi_{k,0} - \psi_{k,1})$ com `WINDING_SECTOR` (dente $k$: `coil_k_0` ida,
   `coil_k_1` volta).
3. **Torque de excitação** (sem cogging), Clarke amplitude-invariante:
   $T_{exc} = 3 \cdot \tfrac{3}{2}\,p\,(\psi_\alpha i_\beta - \psi_\beta i_\alpha)$, $p = 21$, 3 setores.
4. **Cogging** $T_0$ = média em θe de $\tfrac12(T^{Maxwell}_{Split} + T^{Maxwell}_{Split,RS})$: a variante RS
   (espelho na reta de 60° + ímãs invertidos, $T' = -T$) erra o $T_0$ com sinal oposto ao original
   e a média cancela o viés (ver abaixo).
5. **Híbrido:** $T(\theta_e) = T_{exc}(\theta_e) + T_0$.

Referência = smooth (Maxwell no B suavizado). O fluxo concatenado aplicado ao GT difere ~3% do
Maxwell no GT (diferença entre métodos, entra no erro do híbrido).

**Resultado (Split mae run_0002, 10 amostras de 0,7 a 9,8 A, erro rel. mediano rmse/rms):**

| | erro |
|---|---|
| excitação: Split Maxwell (sem a média) | 11,4% |
| excitação: Split fluxo concatenado | **4,1%** |
| excitação: GT fluxo concatenado × GT Maxwell (diferença de método) | 3,1% |
| total: Split original (Maxwell) | 18,8% |
| total: (Split original + Split RS)/2 (Maxwell) | 11,5% |
| total: **híbrido** | **5,2%** |

- Melhora 9 de 10 amostras; a #3954 (8,8 A) piora (5,3% → 9,2%: o fluxo concatenado
  superestima o pico) e a #3942 (0,7 A) fica em 14% (torque quase todo cogging).
- O achatamento do pico em corrente alta (ex. #1883: Maxwell 23% → híbrido 6,2%) era do
  cálculo por Maxwell, não do fluxo previsto.
- Saída: `dados/torque/fluxlink_split_<loss>.json`, `plots/torque/fluxlink_split_<loss>_{excitacao,total}.png`.

**Simetrias exatas usadas (TTA)** — `tests/check_split_symmetry_torque.py` monta as variantes em
memória a partir do texto do `.ans` (mesma malha ou espelhada; sem FEMM), elemento $e$ da variante =
elemento $e$ (espelhado) da original:

| variante | o que muda | campo no referencial original | torque |
|---|---|---|---|
| S | correntes −I, ímãs +180° | $B' = -B$ | $T' = T$ |
| R | espelho na reta de 60° + correntes −I | no ponto espelhado: $B_r' = B_r$, $B_\theta' = -B_\theta$ | $T' = -T$ |
| RS | espelho + ímãs +180° | no ponto espelhado: $B_r' = -B_r$, $B_\theta' = B_\theta$ | $T' = -T$ |

- Relações conferidas resolvendo cada variante no FEMM (amostra 2252): S exato; R/RS dentro do
  piso de troca de malha (~4 mT, torque a 0,8%). O estator é simétrico em 60°.
- Torque por Maxwell, 52 amostras: cada variante sozinha ~15% (rel. mediano), mas
  original/S com viés médio −72 mN·m e R/RS +69 mN·m; original+R, original+RS ou as 4 (pesos ½–½)
  zeram o viés e dão ~9,6–9,8% (47/52 melhoram). Dados em `data/temp/symmetry_batch/mae/`
  (52 amostras; o lote foi interrompido, é retomável).
- Mexer só na malha (permutar a entrada, ruído nos nós, segmento solto) quase não descorrelaciona
  o erro; mudar a densidade da malha no entreferro (6,6×) quebra o Split (erro ~5×).

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

### 4.1 Erro × ponto de operação (dentro do `regions`)

O erro de superfície no aço em função do $|\mathbf B|$ da referência no ponto, acumulado na
**mesma passagem** do `regions`, sem nova leitura das amostras (`saturation.SatAccum`;
ligado por padrão no `scripts/run_postprocess.py`, `--no-saturation` desliga).

**Conjuntos**
- Pontos de quadratura no aço (rótulo `iron_1008`) em Ω.
- `aco_int`: aço ∩ $\Omega_{int}(d)$, ou seja, $\mathrm{dist}_f > d$ (o mesmo $d$ do
  `regions`).
- `aco`: o aço inteiro.

**Acumulação e intervalos**
- **Variável:** $|\mathbf B|$ da referência suavizada no ponto (o $|B|_{FEMM}$ da métrica de
  superfície).
- **Histograma**, por run e por piso: 520 bins de 0,005 T em [0; 2,6] T mais um bin de
  overflow, com $\Sigma w$, $\Sigma e w$ e $\Sigma e^2 w$.
- **Intervalos** (`sat_report`): 50 de mesma área de aço, na resolução dos bins. Cada um traz
  o $|B|$ mediano (interpolado no bin), a faixa, a área e $L1$/$L2$ em % do $B_{ref}$ em Ω da
  mesma run.

**Curva B-H:** lida do `.ans` no worker; é igual em todas as amostras (0 diferentes). Dela
saem $\mu_{eff}(|B|) = |B|/(\mu_0 H(|B|))$ (`FemmBH`) e o joelho, onde $\mu_{eff}$ cai à
metade do máximo: **1,431 T** (máximo de 2171 em 0,83 T).

**Saída:** `dados/saturation_error.json`, com definições, `bh` (pontos, curva $\mu_{eff}$,
joelho), `checks`, `runs`/`floors` (intervalos e total por conjunto) e `hist` (somas cruas para
reagrupar). A figura é `plots/saturacao_erro_L1_vs_B.{png,pdf}`: L1 × $|B|$ por arch (MAE
contínuo, MSE tracejado), pisos P0_elem/P1_mat pontilhados, $\mu_{eff}$ no eixo secundário e
o joelho marcado.

**Validação:** a soma do histograma contra a soma feita direto nos pontos, com as máscaras
recodificadas à parte (`tests/check_night_prep.py`), difere em ≤ 1,5e-15 relativo. O overflow
($|B| \ge 2{,}6$ T) é ≤ 2e-6 da área.

**Resultados** (608 amostras; $L1$ total, % do $B_{ref}$ em Ω, `aco_int` / `aco`):

| | FNO2d | FNO_GNN | Elem | Split | piso grade | piso P0_elem | piso P1 (nós/mat) |
|---|---|---|---|---|---|---|---|
| MSE | 4,01 / 5,79 | 4,42 / 4,63 | 5,99 / 6,15 | 3,82 / 4,01 | 1,80 / 3,53 | 3,43 / 3,32 | 0,03 / 0,03 |
| MAE | 3,96 / 5,76 | 4,01 / 4,23 | 5,58 / 5,72 | 3,56 / 3,76 | | | |

O que se vê:
- **Abaixo de 1 T:** o erro de todas as archs cresce suavemente com $|B|$.
- **Perto do joelho:** sobe de 1,5 a 2 vezes. O último intervalo, acima do joelho, tem 2% da
  área de aço e fica com L1 de 6,4–8,7% em `aco_int`.
- **FNO2d no aço inteiro:** chega a cerca de 22% no alto $|B|$, porque o aço muito saturado
  fica junto das interfaces (dentes) e a grade não representa o salto.
- **Split:** é a menor em todo o intervalo.

## 5. Custo computacional (`timing`, `src/postprocess/timing.py`; Windows com FEMM e GPU)

Compara a solução do FEMM com a inferência de cada modelo, com a **mesma malha** dos dois
lados; a geração da malha não entra em nenhum tempo. Tabela `tab:cost` do artigo.

```
python -m scripts.postprocess timing [remesh|femm|infer|table|all]   # padrão: all
```

**Amostras e runs.** As mesmas 50 amostras igualmente espaçadas do subset `test` nos três
tempos (índices em `sample_idx` do JSON; 29 292–35 198 nós, mediana 32 135). Runs **MAE**
de FNO2d, FNO_GNN, FNO_BipartiteGNN_Elem (= "FNO_BipartiteGNN" no artigo) e FNO_SplitGNN
(`run_0002` de cada, as mesmas dos resultados principais). Todos os números são **medianas
entre amostras** das medianas por amostra; o IQR é entre amostras; média e desvio padrão
ficam só no JSON.

### 5.1 O `mi_analyze` remalha (verificado em 2026-11-12, `timing remesh`)

`check_analyze_remesh` (amostra 5408, 3 rep.; o `.fem` salvo sem malha e reaberto) mede os
filhos do `femm.exe` em cada chamada:

| Chamada | ms (mediana) | Processos |
|---|---|---|
| `mi_createmesh` | 397 | `triangle.exe` ×2 |
| `mi_analyze` sem malha antes | 3858 | `triangle.exe` ×2 + `fkn.exe` |
| `mi_analyze` após `mi_createmesh` | 3870 | `triangle.exe` ×2 (263) + `fkn.exe` (3524) |

O `mi_analyze` **sempre** roda o `triangle.exe` de novo, com ou sem malha prévia. A malha das
duas soluções é idêntica entre si e à do raw, com $|\Delta A| = 0$. Logo, o tempo do
`mi_analyze` inteiro inclui a malha e não serve como $t_{FEMM}$. Nada é subtraído: mede-se o
solver diretamente.

### 5.2 Definições

**$t_{FEMM}$** (`femm_solve_timed_procs`): duração do processo **`fkn.exe`** (o solver)
disparado pelo `mi_analyze`.
- Medida por polling dos filhos do `femm.exe` (`ChildWatch`: `psutil.pids()` a cada leitura,
  com nome e pai consultados só nos PIDs novos). A espera alvo é `POLL_S` = 4 ms; o
  intervalo real foi de **5,0 ms** de mediana, p99 ≤ 7,7 ms e máximo de 19,2 ms (0,47% das
  leituras > 10 ms). A duração medida (última − 1ª leitura) subestima a real em até cerca
  de 1 intervalo em cada ponta.
- **O `fkn.exe` inclui a leitura da malha (`.node/.ele/.pbc`) e a escrita do `.ans` pelo
  próprio solver**, que não dá para separar de fora.
- Fora: `openfemm`, desenho da geometria, `mi_createmesh`, os 2 `triangle.exe` dentro do
  `mi_analyze`, `closefemm` e a leitura do `.ans` (todos gravados no JSON).
- Conferência por amostra: `mi_analyze` = triangle + fkn + resto, com o resto de **85 ms**
  de mediana (IQR 6,3; 71–96 ms), que são o intervalo entre processos e o overhead do FEMM.
  Em todas as 50 amostras houve exatamente 2 `triangle.exe` + 1 `fkn.exe`.

**$t_{pre}$** (`preprocess_timed` + `encode_timed`): do `.ans` resolvido até a entrada
codificada na GPU, pelo mesmo caminho da construção do dataset (`check_preprocess` =
igualdade exata com `parse_smooth` e `parse_split_smooth`, todas as chaves dos 4 layouts, 50/50).
- Etapas:
  - `read_mesh`: linhas do `.ans` + x, y dos nós + conectividade/rótulos (`usecols`, sem
    converter a coluna A).
  - `materials`, `x_hw` (malha → grade: trifinder + valor do elemento em cada célula) e
    `node_pos`.
  - `graph_topology` (arestas + pares do wrap), `v1_graph` (vértices), `elem_feats` (o
    `elem_x` do bipartido, comum a bip/Elem/Split), `bip_graph`, `elem_graph` e
    `split_graph`.
  - `encode`: `_prepare` da inferência (numpy → tensor → GPU + z-score), com
    `cuda.synchronize`.
- $t_{pre}$ da arch = soma das etapas de `ARCH_STAGES[arch]` + encode, por repetição;
  1 aquecimento + 5 rep. por amostra.
- Fora: geração da malha e `read_solution` (conversão da coluna A, gravada à parte: 8,9 ms).
  Ressalva: no `.ans`, x, y e A estão na mesma linha, então os bytes de A são lidos do disco
  no `read_mesh`. Lê o `.ans` não comprimido que o FEMM acabou de gravar.
- FNO2d: inclui `node_pos` e o envio de r/c dos nós (0,32 ms), porque entrega o campo nos
  nós como o FEMM.

**$t_{inf}$** (`time_per_sample`): `best.pth`, `torch.inference_mode`, lote 1, forward +
desnormalização, com `cuda.synchronize` antes e depois.
- FNO2d: + interpolação grade → nós (parcelas `grid` e `interp` à parte no JSON).
- Split: saídas N_aux e N_base desnormalizadas.
- 3 aquecimentos no início de cada run + 10 rep. por amostra, com a entrada já na GPU.
- Entradas: `preprocess_timed` sobre o `.ans.gz` do raw (malha e A idênticos aos do FEMM).
- Fora: $t_{pre}$ e qualquer pós-processamento.

### 5.3 Condições de medição

- **Afinidade de CPU** (`--cpu-affinity pcores`, padrão): o processo Python e o `femm.exe`
  ficam só nos núcleos P (lógicos 0–15 do i9-13900KF, detectados por `EfficiencyClass`); o
  `triangle.exe` e o `fkn.exe` herdam a afinidade (conferido nas 50 amostras).
  - Motivo: a inferência de lote 1 é limitada pelo lançamento de kernels (CPU). Sem
    afinidade, o Windows alterna a thread entre núcleos P e E e os tempos ficam bimodais
    (FNO2d 2,1 × 3,4 ms; Elem 3,8 × 6,1 ms). Fixado dos dois lados por igualdade.
- **Spin-up da GPU** (`--gpu-spinup-s 2`): antes dos aquecimentos de cada run, 2 s de matmul
  genérico, sem o modelo. Sem isso, a 1ª amostra do 1º modelo saía cerca de 3× mais lenta
  (GPU em repouso; os 3 aquecimentos somam cerca de 10 ms).
- Limite de 95% da VRAM (WDDM: OOM real em vez de transbordar para a RAM).
- Hardware: i9-13900KF (24 núcleos, 32 lógicos), 31,8 GiB de RAM, RTX 4080 16 GB (driver
  610.88), Windows 11 Pro, Python 3.12.2, torch 2.3.1+cu121, FEMM 4.2 (`femm.exe` de
  2019-04-21).

### 5.4 Resultados (2026-11-12)

$t_{FEMM}$ = **3184,8 ms** (IQR 435,4; 2491–4445 ms). O `mi_analyze` inteiro dá 3549,6 ms;
o `triangle.exe` ×2, 262,8 ms.

| Arquitetura | $t_{inf}$ ms (IQR) | % do FEMM | $t_{pre}$ ms (IQR) | encode ms |
|---|---|---|---|---|
| FNO2d | 2,25 (0,11) | 0,07 | 374,9 (51,1) | 3,6 |
| FNO_GNN | 3,33 (0,07) | 0,10 | 472,1 (52,1) | 3,5 |
| FNO_BipartiteGNN_Elem | 3,91 (0,11) | 0,12 | 498,6 (55,9) | 5,8 |
| FNO_SplitGNN | 5,68 (0,18) | 0,18 | 447,3 (61,3) | 8,4 |

- FNO2d: grade 1,96 + interpolação 0,21 ms.
- $t_{pre}$ por etapa (mediana, ms): `read_mesh` 39,9, `materials` 2,2, **`x_hw` 327,8**
  (IQR 50,5), `node_pos` 0,5, `graph_topology` 80,0, `v1_graph` 10,4, `elem_feats` 1,6,
  `bip_graph` 12,6, `elem_graph` 21,4, `split_graph` 63,4. A interpolação malha → grade é
  66% (Elem) a 87% (FNO2d) do $t_{pre}$ e domina o custo: o pré-processamento é de 79×
  (Split) a 167× (FNO2d) a inferência.
- Lote máximo (fora da tabela): FNO2d 512 (0,80 ms/amostra), FNO_GNN 128 (2,28), Elem 128
  (3,55), Split 64 (6,21 — não melhora em relação ao lote 1).
- A/B do refatoramento: com o `preprocess_timed` antigo e o novo intercalados (10 amostras,
  as duas ordens), a soma das etapas sem a leitura fica igual dentro de 1% em todas as
  archs antigas; a divisão de `bip_graph` em `elem_feats` + `bip_graph` custa cerca de
  0,2 ms. A única mudança de $t_{pre}$ é a leitura: `read_ans` (59–60 ms, com A) →
  `read_mesh` (41–42 ms).

Saída: `dados/timing.json` com `remesh_check`, `cpu_affinity`, `femm_preproc` (por amostra,
etapas, conferências, polling), `inference` (repetições por amostra, varredura de lote),
`hardware` e `table`, e `dados/timing_table.tex` (gerado do JSON: ms com 1 casa, fração com
1 casa ou 2 se < 1%, vírgula decimal, `--decimal` para trocar).

## 6. Outros subcomandos

| Subcomando | O que faz | Saída |
|---|---|---|
| `figures [--data] [--out]` | curvas de treino (`mae_graph`; FNO2d `mae_hw`); barras de $L1_A$/$L2_A$ com o piso de cada representação; $|\mathbf B|$ nos nós dos 5 modelos mae nas amostras P50/P95 do $L1_A$ de `FNO_BipartiteGNN_mae` (setor e zoom no entreferro) | `plots/1_curvas_treino`, `2_barras_erro`, `3_campo_*`; `dados/figures_smooth.json` |
| `surface IDX [LOSS]` / `surface --pct P [LOSS] [--ref-run FNO2d_mae] [--vmax T] [--cmap viridis] [--fig-article DIR]` | erro no domínio em 1 amostra (ou na do percentil P do L1_A da `--ref-run` no eval): archs do artigo (FNO2d, FNO_GNN, Elem, Split N_base), cada uma na superfície nativa; layout, Split e Ω do regions; L1_A/L2_A só em Ω. Mapa $\sum e\,w/\sum w$ por triângulo (`surface_error_per_triangle`), fora de Ω em cinza, contornos de material; escala comum com vmax = p99 dos 4 mapas em Ω arredondado para cima em 0,01 T (ou `--vmax`). JSON com conferências (média de e_tri × L1_A; caminho do regions) | `docs/figures/erro_dominio.{pdf,png}` (as 4 archs em 2×2, 18 cm, nome da arch centralizado acima de cada painel; uma colorbar vertical à direita, sem seta, ticks a cada 0,1 T; `--cmap` default `viridis`; sem contorno dos materiais; fontes por `primitives.apply_article_style()`: Times New Roman 10 pt no surface (`apply_article_style(size)`, padrão 8), mathtext stix, pdf.fonttype 42, falha se a fonte não existir) + `docs/figures/campo_referencia.{pdf,png}` (|B| do FEMM suavizado, média por triângulo nos pontos de quadratura, 1 painel de 8,8 cm, mesmo padrão, 0–2,5 T com ticks de 0,5); diagnóstico em `dados/surface_error/sample_*.{json,npz}` (com `in_omega`, `e_tri_FNO_SplitGNN`) e `plots/surface_error/*.png` |
| `python -m scripts.fig_erro_zoom [--sample 2184] [--loss mae] [--mode both\|coarse\|fine] [--band] [--d MM] [--percentile 99] [--sem-figura]` (script à parte, não é subcomando) | figura "Distribuição espacial do erro": $\Delta|\mathbf B|$ = erro do módulo (a grandeza das métricas) em 3 janelas fixas (Quina e Interface com 4,5 mm, Entreferro com 8 mm em θ = 90°; centro x, y e lado em mm em `DEFAULT_WINDOWS`, impressos no terminal) × 4 archs mae, numa amostra só, sem recalcular métricas. Mesmos pontos de avaliação do regions (`_regions_sample`, `_surface_points`/`_split_points`). Padrão: 1 valor por triângulo (Σe·w/Σw); `fine`: pedaços célula ∩ triângulo (FNO2d) e os 16 subtriângulos da quadratura (P1 e Bipartite). Cada linha pinta só a sua região (`ROW_REGION`: Quina = quinas, Interface = fronteira sem quinas, classificação do regions por ponto no d; Entreferro = anel entre o raio externo do estator e o interno do rotor ∩ ar; Interior também disponível); cada polígono é recortado pela forma exata da região (`region_zone`: discos de raio d nas quinas, faixa de largura d em Γ menos os discos, resto, ou o anel do entreferro; ∩ Ω) e só a parte de dentro é pintada, com a média dos pontos da região no polígono; borda igual nas 4 archs; a área recortada é conferida contra a da região (JSON `zone_area_rel_diff`). No fino a Bipartite também é subdividida (16 subtriângulos, os mesmos 48 pontos da métrica). Escala por linha: vmin 0, vmax = p99 dos polígonos coloridos (4 archs), colorbar só retângulo (acima de vmax = cor do topo); coluna à esquerda com o domínio completo e o retângulo da janela; viridis; malha cinza 0,2, interfaces pretas 0,6; `--band` desenha o contorno de dist(Γ) ≤ d tracejado; 7,16 in, 8 pt | `docs/figures/erro_zoom_regioes{,_fino}.{pdf,png}`, `docs/figures/erro_zoom_regioes_localizacao.{pdf,png}` (setor + janelas), `dados/fig_erro_zoom/sample_XXXXXX_<loss>.json` (janelas, vmax, contagens) e `dados/fig_erro_zoom/sample_XXXXXX_<loss>_dados.json` (`zoom_data`: por região → arch → modo normal/fino, os polígonos pintados com val [T], área, centróide e resumo: média ponderada pela área, máx, p50/p95/p99; `--sem-figura` grava só esse) e `dados/fig_erro_zoom/sample_XXXXXX_<loss>_resumo.json` (`zoom_summary`, ~50 kB, para um agente ler as figuras: janela, vmax, materiais da zona; por arch × modo, estatísticas, frações de área acima de vmax e vmax/2, 5 pontos quentes com r, θ e material, mapa 6×6 da janela) |
| `python -m scripts.fig_representacoes [--sample 2184] [--center X Y] [--side MM] [--layout 2x2\|1x4]` (script à parte) | figura "Representações de entrada": a mesma janela (padrão: trecho reto da interface ferro/cobre no flanco do dente dentro da ranhura, longe das quinas, com malha grossa; centro (18,07; 28,94) mm, r = 34,1 mm, θ = 58,0°, lado 2,5 mm: 14 vértices, 29 elementos, 185 células) nas 4 entradas: (a) grade `x_hw` (células de lados retos, `_cell_corners`), (b) grafo de vértices (`node_x`, `edge_index`), (c) Elem (centróides com `node_x`, arestas duais, vértices auxiliares sem material, arestas cruzadas), (d) Split (N_aux, Edge_dual, cópias N_base por classe (material, M) e Edge_aux até a cópia do material do elemento). Material de cada marcador = classe de μ_r e M das features do próprio layout; conferido contra `elem_mat`/`node_mat` e o elemento que contém o centro do pixel (0 divergências na #2184). No (d) as cópias de vértices com 2+ cópias são deslocadas 20% da distância vértice → centróide médio dos elementos do seu material (`COPY_SHIFT`). Todos os nós circulares: elemento = disco cheio, vértice / cópia = anel na cor do material, vértice sem material = anel cinza fino; fundo dos 4 painéis com a região de cada material em tom muito fraco (`FAINT`); na (a) cada célula só com o contorno fino na cor forte do seu material (`x_hw`), recuado 10% (`CELL_INSET`), para a célula que atravessa a interface aparecer sobre o fundo do outro material; arestas: malha cinza fina, grafo quase preto sólido, vértice–elemento (cruzada / Edge_aux) cinza médio tracejada, sem linha de interface (o fundo mostra a divisa); legenda em 3 colunas (materiais | célula, vértice, elemento | arestas); malha cinza, interfaces pretas, Okabe–Ito; 18 cm, 8 pt, com e sem títulos | `docs/figures/representacoes_entrada.{pdf,png}` (sem título nos painéis; rótulos (a) Grade polar, (b) Malha como grafo, (c) Grafo bipartido, (d) Grafo por material), `dados/fig_representacoes/sample_XXXXXX.json` (janela, materiais, contagens, deslocamento, conferência) |
| `python -m scripts.fig_arquiteturas [--only fno2d fno_gnn fno_bipartite fno_splitgnn] [--estilo quadrado\|largo]` (script à parte) | diagramas de blocos das 4 archs do artigo, sem hiperparâmetros. Padrão (quadrado, 3,5 in, 7 pt): fluxo de cima para baixo à esquerda (FNO como bloco único) e a camada de mensagens aberta à direita; FNO2d só com lifting → 4 camadas de Fourier (bloco empilhado) → projeção, sem detalhe. `--estilo largo` (7,16 in): linha principal (entrada → FNO → projeção grade → grafo → concatenação com as features do grafo → GNN → soma residual → saída; no FNO2d, lifting → 4 camadas de Fourier → projeção) e a camada repetida aberta numa moldura tracejada (Fourier: FFT → R → FFT⁻¹ em paralelo com W, soma, GELU; GNN: gate sig(MLP(e)) × W h, Σ, concatenação, MLP do nó; Bipartite com 2 mensagens, Split com 3 mensagens e 2 atualizações). Cores por tipo de bloco (dados, FNO, projeção, GNN, gate); 7,16 in, 8 pt | `docs/figures/arq_{fno2d,fno_gnn,fno_bipartite,fno_splitgnn}.{pdf,png}` |
| `train-table` | tabela das 10 runs só a partir dos logs (`config.json`, `run_summary.json`, `epochs.csv`): parâmetros, pico de GPU, medianas de tempo por época, melhor/última época, motivo da parada, hardware (`device`, `gpu_name`, versões de torch/CUDA/cuDNN, `hostname`) | `dados/train_table.json` |

## 7. Execução com memória controlada (`scripts/run_postprocess.py`)

Script de partida para as long runs na máquina local (32 GB de RAM). Ele só orquestra: chama
as mesmas funções do `postprocess.py` (`regions_setup/state/track/add_floors/add_run/
finalize`, que o subcomando `regions` também usa), sem cálculo próprio. Por ora, a única etapa
é `--steps regions`. As demais entram no dicionário `STEPS`, chamando os `run_*` existentes.

```
python -m scripts.run_postprocess --steps regions --n 10 --workers 2      # placeholder
python -m scripts.run_postprocess --steps regions --all --workers 2 --max-ram-gb 24 --resume --log-dir resultados/BLDC_138x276/logs
python -m scripts.run_postprocess --steps regions --all --workers 3 [--max-ram-gb 26]
python -m scripts.run_postprocess --steps regions --all --resume          # continua
```

**Opções novas (2026-11-10)**
- A saturação por ponto de operação (§4.1) é acumulada na mesma passagem; `--no-saturation`
  desliga.
- `--log-dir DIR` grava uma cópia do log, com horário em cada linha.
- Parada limpa (`--max-ram-gb`/`--stop-after`): o script sai com código 3 e o checkpoint fica
  para `--resume`.
- As conferências (soma das disjuntas, área de Ω) vão para o log; o `regions.json` traz
  `execution.omega_area_per_sample`.
- Execução noturna (orquestrador avulso, fora do git): `tests/run_night.py` (etapas 1 e 2,
  relança com 1 worker, valida, faz backup e copia) e `tests/night_report.py`.

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
