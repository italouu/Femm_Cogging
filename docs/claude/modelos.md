# Modelos e treino

As 5 archs estão registradas em `ARCH_REGISTRY` (`src/neural_op/archs/__init__.py`). Cada
`ArchEntry` tem a classe, a config, o `loader_mode` (`grid` / `vertex` / `bipartite`), os
kwargs do construtor, o `step_fn` (loss) e a `metric_fn` (monitoramento). A arch e a loss
de uma run vêm do **tipo** de `NnCfg.arch_cfg` e `NnCfg.loss_cfg`. Os canais de
entrada/saída são lidos do 1º chunk em `NnCfg.__post_init__` (`_detect_chunk_dims`).

Todas as entradas e saídas dos modelos estão no espaço normalizado de `dados.md` §5.
$\mathrm{enc}_k$ / $\mathrm{dec}_k$ = z-score / inverso com as stats da chave $k$.

---

## 1. FNO2d (`archs/fno.py`, `archs/_blocks.py`)

Entrada $x \in \mathbb R^{B\times 3\times H\times W}$ (`x_hw`), saída
$\hat y \in \mathbb R^{B\times 2\times H\times W}$ (`y_hw`):

$$ \hat y = Q\big(\mathcal L_4\circ\cdots\circ\mathcal L_1(P(x))\big),\qquad
\mathcal L_\ell(v) = \sigma\big(\mathcal K_\ell v + W_\ell v\big),\quad \sigma = \mathrm{GELU} $$

- $P$ (lift) e $Q$ (projeção): MLPs **pontuais** (`MLP`, aplicadas em cada pixel) com 3
  camadas: $3\to 64\to 64\to w$ e $w\to 64\to 64\to 2$, com GELU entre as camadas e saída
  linear.
- $W_\ell$: `Linear(w, w)` pontual (com bias, sem ativação própria).
- $\mathcal K_\ell$ (`SpectralConv`), com $\hat v = \mathrm{rfft2}(v)$ sobre os eixos $(r, \theta)$:
  $$ (\mathcal K v)^\wedge_{o}(k_r, k_\theta) = \sum_i \hat v_i(k_r,k_\theta)\,R^{(1|2)}_{i o}(k_r,k_\theta) $$
  - $R^{(1)}$ age nas linhas $k_r\in[0, m_1)$ e $R^{(2)}$ nas linhas $[H-m_1, H)$, nas
    colunas $k_\theta\in[0, m_2)$; o resto é zerado; depois vem o `irfft2`.
  - Pesos complexos $w\times w\times m_1\times m_2$, inicializados com `torch.rand` (partes
    real e imaginária em $U[0,1)$).
- Na bateria, $m_1 = 69 = \lceil H/2\rceil$ e $m_2 = 139 = W/2 + 1$ cobrem **todo** o
  espectro da grade 138×276 (`full_spectrum_modes`). Não há truncamento de modos.
- A DFT impõe **periodicidade implícita nos dois eixos**. Em $\theta$, ela coincide com a
  periodicidade de 120° do problema (a grade cobre exatamente um período). Em $r$, liga
  $R_{in}$ a $R_{out}$, o que não corresponde à física.
- Larguras: $w = 8$ (FNO2d) e $w = 6$ (estágio FNO das archs híbridas), sempre 4 blocos.

## 2. Interpolação grade → pontos (`archs/interp.py::interpolate_grid_to_nodes`)

Bilinear com eixo angular **circular** e eixo radial preso à borda. Em índices de pixel:

$$ u = c_{base}\,W - \tfrac12,\qquad v = r_{base}\,H - \tfrac12 $$

- $\theta$: o alvo polar $(B_r, B_\theta)$ é periódico no corte 0°/120° (setor com nº par de
  polos). A grade ganha uma coluna de cada lado (coluna $W-1$ antes da 0, coluna 0 depois da
  $W-1$), e um ponto a menos de meio pixel do corte interpola entre as colunas 0 e $W-1$.
- $r$: `padding_mode='border'`; um ponto a menos de meio pixel de $R_{in}$/$R_{out}$ recebe a
  linha de borda.

Implementação: `F.grid_sample(grid_padded, mode='bilinear', align_corners=False,
padding_mode='border')` sobre a grade $H\times(W+2)$, com

$$ x_n = \frac{2\,c_{base}W + 2}{W + 2} - 1,\qquad y_n = 2r_{base} - 1 $$

É uma operação linear, então comuta com a (des)normalização afim por canal. É a mesma
função no treino (FNO_GNN, GNN_PostBase, FNO_BipartiteGNN, Elem) e no pós-processamento
(`inference.predict`, `floor_nodes`, `timing`).

## 3. Blocos de GNN (`archs/_blocks.py`)

`NodeMLP(in, out, hidden, n)`: lineares com GELU entre elas, saída linear.

**`EdgeConvLayer`** (FNO_GNN, GNN_PostBase), aresta $i\to j$ com atributo $e_{ij}$:

$$ g_{ij} = \mathrm{sigmoid}\big(\mathrm{NodeMLP}_{E\to 4E\to 1}(e_{ij})\big),\qquad
m_{ij} = g_{ij}\, W h_i\ (W\ \text{sem bias}),\qquad a_j = \sum_{i\to j} m_{ij} $$
$$ h_j' = \mathrm{NodeMLP}_{2w\to w\to w}\big([h_j \,\|\, a_j]\big) $$

**`GNN`**: $h^0 = \sigma(\mathrm{Lin}(x))$; $h^{l} = \mathrm{EdgeConv}_l(h^{l-1})$;
saída $\mathrm{Lin}(h^{L})$. Não há resíduo nem ativação depois da atualização.

**`BipartiteEdgeConvLayer`** (FNO_BipartiteGNN, Elem): duas mensagens com pesos próprios
por camada (`EdgeMessageBlock`). O grafo auxiliar não é atualizado; cada camada lê as
features cruas (normalizadas) `elem_x`:

$$ F^1_j = \sum_{i\to j} \mathrm{sig}\big(\mathrm{MLP}(e_{ij})\big)\,W_1 h_i,\qquad
F^2_j = \sum_{a\to j} \mathrm{sig}\big(\mathrm{MLP}(d_{aj})\big)\,W_2\, x^{aux}_a $$
$$ h_j' = \mathrm{NodeMLP}_{3w\to w\to w}\big([h_j \,\|\, F^1_j \,\|\, F^2_j]\big) $$

com as arestas cruzadas $a\to j$ (linha 0 = auxiliar, linha 1 = principal) e
$d_{aj}$ = `cross_edge_attr`. **`BipartiteGNN`**: lift + $L$ camadas + projeção, como o `GNN`.

**`SplitGNNLayer`** (FNO_SplitGNN). Aqui os dois tipos de nó têm estado, e cada um tem a
sua rede: $a$ = N_aux (elementos), $b$ = N_base (vértices por material). São três mensagens
`EdgeMessageBlock` com pesos próprios. A Edge_aux é usada nos dois sentidos com os mesmos
atributos estáticos. As duas atualizações leem só a camada $k$ (atualização simultânea):

$$ F^{dual}_i = \sum_{j\to i} \mathrm{sig}\big(\mathrm{MLP}(e^{dual}_{ji})\big)\,W_1 h^a_j,\quad
F^{b\to a}_i = \sum_{c\to i} \mathrm{sig}\big(\mathrm{MLP}(e^{aux}_{ci})\big)\,W_2 h^b_c,\quad
F^{a\to b}_c = \sum_{i\to c} \mathrm{sig}\big(\mathrm{MLP}(e^{aux}_{ci})\big)\,W_3 h^a_i $$
$$ h^{a\prime}_i = \mathrm{NodeMLP}^a_{3w\to w\to w}\big([h^a_i \,\|\, F^{dual}_i \,\|\, F^{b\to a}_i]\big),\qquad
h^{b\prime}_c = \mathrm{NodeMLP}^b_{2w\to w\to w}\big([h^b_c \,\|\, F^{a\to b}_c]\big) $$

**`SplitGNN`**:
- entrada: lift e GELU próprios de cada tipo de nó;
- miolo: $L$ camadas `SplitGNNLayer`;
- saída: uma projeção linear por tipo de nó.

## 4. Archs híbridas

Notação: $u_k = I(\hat y_{hw})(r_k, c_k)$ é a saída do FNO interpolada no ponto $k$, e
$R(u) = \mathrm{enc}_{node\_y}(\mathrm{dec}_{y\_hw}(u))$ é a **recodificação** das stats da
grade para as do grafo (`fno_gnn._rescale_fno_to_node_space`).

**FNO_GNN** (`archs/fno_gnn.py`): o grafo é o de vértices, e a interpolação lê `node_x[:,3:5]`.

$$ \tilde u_k = R(u_k),\qquad \hat y_k = \tilde u_k + \mathrm{GNN}\big([\,x^{node}_k \,\|\, \tilde u_k\,]\big) $$

**FNO_BipartiteGNN** (`archs/fno_bipartite_gnn.py`): o grafo principal é o de vértices
(`node_x` = posição, interpolada em `node_x[:,0:2]`), e o auxiliar são os elementos
(`elem_x`, 6 canais), injetados por arestas elemento → vértice:

$$ \hat y_k = \tilde u_k + \mathrm{BipartiteGNN}\big([\,x^{node}_k \,\|\, \tilde u_k\,],\ x^{elem}\big) $$

**FNO_BipartiteGNN_Elem** (`archs/fno_bipartite_gnn_elem.py`): é a **mesma rede** da
FNO_BipartiteGNN (subclasse, mesma config). A diferença está toda no layout de dados
(`dados.md` §3.4): o grafo principal são os elementos, a interpolação é nos centróides
(`node_x[:,0:2]`), o auxiliar são os vértices (`elem_x` = posição) e as arestas cruzadas vão
de vértice para elemento. A saída é $(B_r, B_\theta)$ **por elemento**.

**FNO_SplitGNN** (`archs/fno_split_gnn.py`): o layout está em `dados.md` §3.5. O FNO é
interpolado nos centróides (`node_x[:,0:2]`) e nas cópias de vértice (`elem_x[:,0:2]`), e
cada valor é recodificado para as stats de `node_y` e `elem_y`, respectivamente. A coluna
$c_{base}$ não entra na GNN: o ângulo entra pelo par $(\sin, \cos)$. A rede tem duas saídas:

$$ \hat y^a = \tilde u^a + \Delta^a,\quad \hat y^b = \tilde u^b + \Delta^b,\qquad
(\Delta^a, \Delta^b) = \mathrm{SplitGNN}\big([x^a_{\setminus c} \,\|\, \tilde u^a],\ [x^b_{\setminus c} \,\|\, \tilde u^b]\big) $$

A perda é $\mathcal L(\hat y^a, y^a) + \mathcal L(\hat y^b, y^b)$, cada termo com a média nos
seus próprios pontos. O `mae_graph` é a média dos dois MAEs.

O treino é **conjunto** (FNO + GNN): a perda é aplicada só na saída final $\hat y_k$, e o
gradiente chega ao FNO pela interpolação. A grade $\hat y_{hw}$ não é supervisionada.

**GNN_PostBase** (`archs/gnn_post_base.py`): FNO2d **congelado** (o `best.pth` da run FNO2d
da mesma bateria e mesma loss; `requires_grad=False`, `eval`, `no_grad`) mais um `GNN`
treinável. O round-trip entre as normalizações da run e da base é:

$$ x^{base} = \mathrm{enc}^{base}_{x\_hw}\big(\mathrm{dec}^{run}_{x\_hw}(x_{hw})\big),\quad
u_k = I\big(\mathrm{FNO}_{base}(x^{base})\big)(r_k,c_k),\quad
\tilde u_k = \mathrm{enc}^{run}_{node\_y}\big(\mathrm{dec}^{base}_{y\_hw}(u_k)\big) $$
$$ \hat y_k = \tilde u_k + \mathrm{GNN}\big([\,x^{node}_k \,\|\, \tilde u_k\,]\big) $$

### Hiperparâmetros (bateria `bldc`) e nº de parâmetros reais (complexo conta ×2)

| Arch | FNO ($w$, blocos, modos, lift/proj) | GNN (largura, camadas, entrada, aresta, aux, cruzada) | Parâmetros |
|---|---|---|---|
| FNO2d | 8, 4, 69×139, 64×3 | — | 9 831 274 |
| FNO_GNN | 6, 4, 69×139, 64×3 | 32, 3, 6+2, 4, —, — | 5 547 253 |
| FNO_BipartiteGNN | 6, 4, 69×139, 64×3 | 32, 3, 2+2, 3, 6, 1 | 5 550 704 |
| FNO_BipartiteGNN_Elem | 6, 4, 69×139, 64×3 | 32, 3, 6+2, 3, 2, 1 | 5 550 448 |
| GNN_PostBase | FNO2d da base (congelado) | 64, 6, 6+2, 4, —, — | 9 931 634 (100 360 treináveis) |
| FNO_SplitGNN (bateria `split`) | 6, 4, 69×139, 64×3 | 32, 3, 6+2 (cada tipo de nó), 7, 7, 7 | 5 568 217 |

---

## 5. Treino (`scripts/train.py::run`, `src/neural_op/training_utils.py::fit`)

### 5.1 Perda (`src/neural_op/losses.py`, `LOSS_REGISTRY`)

Sobre a **saída final** e no espaço **normalizado**, componente a componente, sem peso de
área. Para o lote com $N$ escalares (todos os pontos de todas as amostras, × 2 componentes):

$$ \mathcal L_{\mathrm{mse}} = \frac1N\sum (\hat y - y)^2,\qquad \mathcal L_{\mathrm{mae}} = \frac1N\sum |\hat y - y| $$

Os pontos são as células da grade (FNO2d, `y_hw`), os nós (FNO_GNN, GNN_PostBase,
FNO_BipartiteGNN, `node_y`) ou os elementos (Elem, `node_y`).
- `train_loss` da época = média simples das perdas por lote.
- `val_loss` = a mesma perda nos chunks de validação, também média por lote. Os chunks
  de teste não entram no treino.

### 5.2 Otimização

`torch.optim.Adam(lr)` com os defaults do PyTorch, mais `StepLR(step_size=100, gamma)`, que
avança uma vez por época. 500 épocas, lote 32.

| Run | lr | $\gamma$ |
|---|---|---|
| FNO2d | 0,005 | 0,7 |
| FNO_GNN | 0,003 | 0,8 |
| FNO_BipartiteGNN, Elem, FNO_SplitGNN | 0,01 | 0,6 |
| GNN_PostBase | 0,001 | 0,6 |

### 5.3 Monitoramento e parada (`src/neural_op/monitor.py::TrainingMonitor`, `MonitorCfg`)

O **heartbeat** acontece a cada `checkpoint_every = 10` épocas (época $t$ com
$(t+1) \bmod 10 = 0$). Em cada heartbeat:
- grava `latest.pth`;
- grava `best.pth` se $L_{val} < L^*_{val} - 10^{-6}$;
- atualiza os contadores:
  - paciência: heartbeats consecutivos sem melhora;
  - GL: heartbeats consecutivos com $\mathrm{GL} > 5$, onde
    $$ \mathrm{GL}(t) = 100\Big(\frac{L_{val}(t)}{L^*_{val}} - 1\Big) $$

A parada ocorre na primeira das condições abaixo, e só se $t + 1 \ge$ `min_epochs = 100`:
- paciência $\ge 10$ heartbeats (100 épocas sem melhora);
- GL $> 5\%$ por 3 heartbeats consecutivos;
- fim das 500 épocas.

O `best.pth` sempre corresponde a uma época de heartbeat.

**Métricas de monitoramento** (`metric_fn`, a cada época com `metrics_every_epoch=True`, um
2º forward sobre a validação): MAE em T, depois de desnormalizar, média por lote.
- `mae_hw`: só FNO2d (grade).
- `mae_graph`: as demais (nós; elementos na Elem).

Nenhuma decisão de treino usa essas métricas.

### 5.4 Logs da run (`src/training/model_manager.py::ModelManager`)

Diretório: `data/logs/<problem>/<arch>/run_XXXX/`.

| Arquivo | Conteúdo |
|---|---|
| `config.json` | `asdict(NnCfg)` (inclui `norm_stats`), contagens de parâmetros, git, base do PostBase |
| `split.json` | nomes dos chunks de treino / validação / teste (`train`, `val`, `test`) |
| `metrics.jsonl` | 1 linha por heartbeat (`train_loss`, `val_loss`, lr, tempos, MAE) |
| `epochs.csv` | 1 linha por época (perdas, lr, tempos de treino/avaliação/métricas, MAE) |
| `run_summary.json` | status, motivo da parada, melhor época e `best_val_loss`, tempo de parede, pico de memória da GPU, hardware (`device`, `gpu_name`, `torch_version`, `cuda_version`, `cudnn_version`, `hostname`) |
| `status.txt`, `notes.txt` | estado (`running`/`done`/`stopped`/`failed`) e notas manuais |
| `checkpoints/{best,latest}.pth`, `model_final.pth` | `model_state_dict`, `optimizer`, `scheduler`, `epoch` |

### 5.5 Bateria e runner

`src/configs/batteries/bldc.py::BATTERY`: 10 runs (5 archs × mse/mae), cada uma uma função
que devolve um `NnCfg` escrito por inteiro (`src/configs/battery.py`).
- Ordem: Elem → Bipartite → FNO_GNN → FNO2d → GNN_PostBase.
- `postbase_*` declaram `needs=('fno2d_*',)` e recebem o `run_dir` da base.
- Dataset `BLDC_138x276/<arch>`, problem `BLDC_138x276`, split 60/30/10 (`train_split=0.60`, `val_split=0.30`), `split_seed=12`.

`src/configs/batteries/split.py::BATTERY`: 2 runs do FNO_SplitGNN (`split_mse`,
`split_mae`), com os mesmos hiperparâmetros das `elem_*`, exceto `metrics_every_epoch=False`.
Com isso o MAE só é calculado no heartbeat, a cada 10 épocas, o que economiza cerca de 11% da
época.
- Dataset `BLDC_138x276/FNO_SplitGNN`.
- Pico de memória estimado em cerca de 31 GiB no lote 32 para amostras medianas, e cerca de
  34 GiB no pior caso (lote 8 medido × 4), contra 18,8 GiB da Elem.
- As mensagens usam `SplitEdgeMessageBlock`, que aplica W nos nós antes de levar às arestas:
  é a mesma conta e economiza cerca de 15% de memória. Com o `EdgeMessageBlock` original, o
  pico chegava a cerca de 40 GiB e deu OOM na L40S.
- Na VM, use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

`scripts/run_battery.py`:
- pré-checa as runs sem dependência;
- treina com `scripts.train.run`;
- registra em `data/logs/_batteries/<nome>/manifest.json` o status, o `run_dir` e
  $\mathrm{hash} = \mathrm{sha256}(\mathrm{json}(\mathrm{asdict}(NnCfg)))[{:}16]$.

Ao rodar de novo:
- runs `done`/`stopped` com o mesmo hash são puladas;
- hash diferente aborta a bateria (mostra a diferença);
- runs `failed`/`interrupted` são refeitas.

O hash inclui as stats de normalização.
