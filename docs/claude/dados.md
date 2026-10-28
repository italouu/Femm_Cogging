# Dados: problema físico, geração, gabaritos, layouts e datasets

Notação: $r,\theta$ coordenadas polares no plano da seção (mm, rad); $M$ nº de elementos e
$S$ nº de vértices da malha de uma amostra; $H\times W = 138\times 276$ a grade polar.

---

## 1. Problema físico (FEMM)

### 1.1 Equação

Magnetostática planar 2D em termos do potencial vetor $\mathbf A = A_z(x,y)\,\hat z$
[Wb/m], resolvida pelo FEMM 4.2 (`mi_probdef(0, 'millimeters', 'planar', 1e-8, L, 200)`,
`src/data_gen/femm_solve.py::save_ans_gzip_sample`):

$$
\nabla\times\big(\nu(|\mathbf B|)\,\nabla\times\mathbf A\big) = J_z\,\hat z + \nabla\times\mathbf H_c ,
\qquad \mathbf B = \nabla\times\mathbf A = \Big(\frac{\partial A_z}{\partial y},\,-\frac{\partial A_z}{\partial x}\Big)
$$

Na forma escalar:
$-\nabla\cdot(\nu\nabla A_z) = J_z + \partial_x H_{c,y} - \partial_y H_{c,x}$, com $\nu = 1/\mu$.

- Frequência 0 (magnetostático), precisão do solver não linear $10^{-8}$.
- Unidades: coordenadas em mm no arquivo; $A$ em Wb/m; $B$ em T; $J$ em A/m² na EDP
  (os layouts guardam J em A/mm²).

### 1.2 Materiais

Definidos em `src/data_gen/motor_constants.py` (`FEMM_MATERIAL_DEFS`) ou lidos da biblioteca
do FEMM (`motor_model.py::BLDC_FEMM_Model._get_material`, que exige igualdade com a
definição quando o material existe nas duas fontes).

| Material | Uso | No FEMM | Constante usada como entrada dos modelos (`PERMEABILITY`) |
|---|---|---|---|
| `iron_1008` | ferro do estator e do rotor | não linear: curva B-H de 19 pontos (0 a 2,5851 T / 397 887 A/m), `Mu_x = 1` | $\mu_r = 5000$ |
| `N35` | ímãs (biblioteca padrão do FEMM) | $\mu_r = 1{,}05$, $H_c = 905\,659$ A/m (valores lidos do `.ans`) | $\mu_r = 1{,}05$ |
| `copper` | bobinas | $\mu_r = 0{,}999991$ | $\mu_r = 0{,}999$ |
| `vacuum` | entreferro, furo, margem externa | $\mu_r = 1$ | $\mu_r = 1$ |

O $\mu_r$ dos modelos é só um rótulo de material por canal. A não linearidade do ferro existe
só no FEMM.

**Magnetização dos ímãs**: cada pedaço de polo é um bloco com direção constante
$\alpha$ (`magdir`) = ângulo do centro do **polo inteiro** (+180° para polos ímpares),
corrigido no pedaço partido pelo corte periódico
(`BLDC_FEMM_Model_Sym120.draw_motor`, `direction + (center − (a1+a2)/2)`): magnetização
paralela dentro do bloco, radial no centro do polo, polaridade alternada.

### 1.3 Domínio e geometria

Motor de **rotor externo**, 42 polos / 36 ranhuras, modelado num setor de 120° (período
mínimo periódico: 14 polos e 12 ranhuras) — `BLDC_FEMM_Model_Sym120_Annular`.

Faixas radiais (mm; valores fixos ou sorteados, ver 1.5):

| Região | Raio |
|---|---|
| contorno interno $A=0$ | $R_{in} = \texttt{inner\_diameter}/2 = 28{,}5$ |
| furo (vácuo) | $28{,}5$ a $\texttt{stator\_inner\_diameter}/2 = 30$ |
| estator (ferro), ranhuras e bobinas | $30$ a $SOD/2$ |
| entreferro (vácuo) | $SOD/2$ a $RID/2$ |
| ímãs | $RID/2$ a $RID/2 + h_p$ |
| coroa do rotor (ferro) | $RID/2 + h_p$ a $\texttt{rotor\_outer\_diameter}/2 = 45$ |
| margem (vácuo) | $45$ a $46{,}5$ |
| contorno externo $A=0$ | $R_{out} = \texttt{outer\_diameter}/2 = 46{,}5$ |

$SOD$ = `stator_outer_diameter`, $RID$ = `rotor_inner_diameter`, $h_p$ = `pole_thickness`.

**Condições de contorno** (`_set_boundary_sector`, `_draw_periodic_cuts`):
- $A = 0$ (propriedade `"A=0"`, `BdryFormat = 0`) nos arcos $r = R_{in}$ e $r = R_{out}$;
- **periódica** (`BdryFormat = 4`) entre os cortes $\theta = 0°$ e $\theta = 120°$:
  $A(r, 0°) = A(r, 120°)$. Há uma propriedade periódica por faixa radial (o FEMM só aceita
  2 segmentos por propriedade): `per_bore_hole`, `per_core`, `per_slot_bottom`,
  `per_slot_throat`, `per_airgap`, `per_pole_ring`, `per_back_iron`, `per_outer_margin`.

**Ranhura e bobina** (`_coil_coords`, `_create_stator_sector`): abertura da ranhura
$\phi_s = \arccos\!\big((2R^2 - B_{s0}^2)/(2R^2)\big)$, $R = SOD/2 - H_{s0}$ (corda $B_{s0}$
no raio $R$). Cada lado de bobina (lóbulo) é o polígono de $r_{bs0} = R\cos(\phi_s/2)$ até
$r_{bs0} - H_{s1} - H_{s2}$, com larguras $B_{s0}/2$, $B_{s1}/2$, $B_{s2}/2$ a partir do eixo
da ranhura. O corte em 0°/120° passa pelo centro de uma ranhura: a ranhura 0 tem só o lóbulo
de $\theta$ maior e a ranhura 12 só o de $\theta$ menor.

**Polos** (`_pole_pieces`): polo $k=0..13$ em $\theta_k = (k\cdot 360/42 + \phi_{rot}) \bmod 120°$,
largura $(360/42)\cdot\texttt{pole\_embrance}$. O polo que cruza 120° vira dois pedaços (um
em cada ponta do setor). Pedaços com menos de 0,01° são colados ao corte.

**Profundidade**: $L$ = `stack_length` = 15 mm (`STACK_LENGTH_MM`), passada ao
`mi_probdef`. Em problema planar o campo $\mathbf B$ não depende de $L$; $L$ só multiplica
integrais (torque, energia, fluxo concatenado). Os `.ans` do raw atual registram
`[Depth] = 1` (gerados com profundidade 1 mm); isso não afeta o campo nem o treino, e o
torque usa $L$ do CSV.

**Malha**: todo bloco é criado com `mi_setblockprop(..., automesh = 0, meshsize = 1, ...)`:
`automesh = 0` (sem malha automática) e área máxima de elemento de 1 mm² por rótulo
(coluna `MaxArea` = 1 nos rótulos do `.ans`). Ângulo mínimo efetivo dos triângulos: 30°,
registrado em `[MinAngle]` do `.ans` (o código passa 200 como `minangle` ao `mi_probdef`, e o
FEMM aplica 30). A malha é gerada pelo FEMM (`mi_createmesh`, seguido de `mi_analyze`).
Mediana por amostra (512 amostras): $S = 32\,270$ vértices (27 891–36 979) e
$M = 63\,825$ triângulos (55 165–73 243).

### 1.4 Correntes

Sistema trifásico equilibrado com amplitude $I$ [A] e ângulo elétrico $\theta_e$
(`BLDC_Process.phase_currents`):

$$ I_a = I\cos\theta_e,\quad I_b = I\cos(\theta_e - 120°),\quad I_c = I\cos(\theta_e + 120°) $$

Enrolamento do setor (`motor_constants.WINDING_SECTOR`): concentrado, camada dupla, uma
bobina por dente $k=0..11$, com fase e sinal $s_k$ da tabela
`A+ A− C− C+ B+ B− A− A+ C+ C− B− B+`. A bobina do dente $k$ tem dois rótulos de cobre:
`coil_k_0` (lóbulo no lado de $\theta$ maior da ranhura $k$), com corrente $+s_k I_{fase}$,
e `coil_k_1` (lóbulo no lado de $\theta$ menor da ranhura $k+1$), com $-s_k I_{fase}$. São
circuitos em série com $N = 18$ espiras por rótulo (`N_TURNS`).

Densidade de corrente uniforme no rótulo (como o FEMM a aplica na magnetostática;
`ans_parsing._parse_label_currents`):

$$ J_\ell = \frac{N_\ell\, I_{circ(\ell)}}{S_\ell}, \qquad S_\ell = \sum_{e\in\ell} A_e \;[\mathrm{mm}^2] $$

Rótulos fora de circuito têm $J = 0$.

### 1.5 Espaço de projeto (`BLDC_Process.generate_samples`, `add_phase_currents`)

`DatagenConfig` (`src/configs/datagen.py`): `dataset='BLDC_138x276'`, `n_samples=6000`,
`datagen_seed=12`, `distribution='uniform'` (o `mode` calculado só valeria para a
distribuição triangular; com `uniform` todos os sorteios são uniformes), `current_peak=10`.

| Parâmetro | Valor / sorteio |
|---|---|
| polos / ranhuras | 42 / 36 |
| `outer_diameter`, `inner_diameter` | 93 / 57 mm (janela do domínio) |
| `rotor_outer_diameter`, `stator_inner_diameter` | 90 / 60 mm |
| `stack_length` | 15 mm fixo. Um valor $U[5,10]$ é sorteado e descartado para manter a sequência do gerador. |
| `stator_outer_diameter` $SOD$ | $U[70{,}5;\,81]$ mm ($81 = 90 - 2(4+0{,}5)$; $70{,}5 = (60+81)/2$) |
| entreferro $g$ | $U[0{,}5;\,2]$ mm; $RID = SOD + 2g$ |
| `pole_thickness` $h_p$ | $U[1{,}5;\; 45 - RID/2 - 1]$ mm |
| `pole_embrance` | $U[0{,}6;\,0{,}9]$ |
| coroa do estator $y_s$ | $U[0{,}1h;\,0{,}3h]$, $h = (SOD-60)/2$ |
| `slot_Hs0`, `slot_Hs1` | $U[0{,}1h';\,0{,}2h']$ cada, $h' = h - y_s$ |
| `slot_Hs2` | $h' - H_{s0} - H_{s1}$ |
| `slot_Bs0` | $U[1;\;0{,}2\cdot 2\pi(SOD/2)/36]$ mm |
| `slot_Bs1` | $U[0{,}3;\,0{,}5]\cdot 2\pi(SOD/2 - H_{s0} - H_{s1})/36$ |
| `slot_Bs2` | $U[0{,}40;\,0{,}65]\cdot 2\pi(SOD/2 - H_{s0} - H_{s1} - H_{s2})/36$ |
| `slot_Rs` | 0 |
| `rotor_phase` $\phi_{rot}$ | $U[0,\; 2\cdot360/42) = U[0;\,17{,}14°)$ (um período elétrico) |
| `current_amplitude` $I$ | $U[0, 10]$ A (gerador próprio `default_rng([seed, 0x1A])`, depois da geometria) |
| `current_angle` $\theta_e$ | $U[0, 360)$°, independente de $\phi_{rot}$ |

Os sorteios garantem $|I_a|,|I_b|,|I_c| \le 10$ A. Os parâmetros ficam em
`data/raw/BLDC_138x276/valid_designs.csv` (uma linha por amostra, colunas `nome [unidade]`).

### 1.6 Geração e conteúdo do `.ans`

`scripts/generate_raw.py` → para cada amostra `femm_solve.save_ans_gzip_sample`:
`openfemm` → `mi_probdef` → `draw_motor` → `mi_saveas` → `mi_createmesh` → `mi_analyze` →
`closefemm`, e o `.ans` inteiro é copiado em gzip para
`data/raw/BLDC_138x276/sample_XXXXXX.ans.gz` (escrita atômica). Sem `mi_loadsolution`.
Os workers são processos com `max_tasks_per_child=1`, há 1 nova tentativa por falha, e a
geração é retomável. `_check_existing_raw` recusa completar um raw cujas colunas ou valores
sorteados divergem do CSV existente.

O `mi_analyze` resolve o problema não linear sobre a malha. [VERIFICAR] se o FEMM 4.2
regenera a malha dentro do `mi_analyze` (teste em `tests/check_analyze_remesh.py`; ver
timing, `avaliacao.md`).

Seções do `.ans` usadas pelo código:

| Seção | Conteúdo | Leitor |
|---|---|---|
| `[LengthUnits]`, `[ProblemType]`, `[Frequency]`, `[Depth]` | cabeçalho | `ans_b_smoothing.load_ans_lines` |
| `[BdryProps]` | contornos | — |
| `[BlockProps]` | uma entrada por rótulo: `Mu_x`, `Mu_y`, `H_c`, curva B-H | `_parse_block_materials`, `load_ans_lines`, `saturation.parse_bh_curves` |
| `[CircuitProps]` | 24 circuitos, `TotalAmps_re` | `_parse_label_currents` |
| `[NumBlockLabels]` | `x y BlockType MaxArea InCircuit MagDir InGroup Turns ...` | `_parse_label_materials`, `_parse_label_currents`, `load_ans_lines` |
| `[Solution]` | $S$ linhas `x y A marcador`; $M$ linhas `n1 n2 n3 rótulo ...` | `_parse_solution_lines`, `load_ans_lines` |

**Leitura única** (`ans_parsing.read_ans`): descomprime em memória e devolve
`AnsData(lines, nodes [S,3] = (x, y, A), elems [M,4] = (n1, n2, n3, rótulo), mesh: AnsMesh)`.
Todos os parsers aceitam `ans=`.

**Material por rótulo** (`_parse_label_materials`): rótulo → `BlockType` → bloco. A
classificação usa `Mu_x` bruto: com curva B-H ou $\mu > 10$ → ferro (id 0);
$1{,}01 < \mu < 10$ → ímã (2); $\mu < 0{,}999995$ → cobre (3); resto → ar (1).
**Polaridade** $M = +1$ se $\cos(\alpha - \theta_{rótulo}) > 0$ (magnetização para fora), senão
$-1$; $M = 0$ fora dos ímãs.

---

## 2. Campos derivados e gabaritos

### 2.1 $\mathbf B$ exato por elemento (P0)

Com $A$ linear (P1) no triângulo de vértices $(x_i, y_i)$, $i = 0,1,2$:

$$
b_0 = y_1-y_2,\; b_1 = y_2-y_0,\; b_2 = y_0-y_1;\quad
c_0 = x_2-x_1,\; c_1 = x_0-x_2,\; c_2 = x_1-x_0;\quad \Delta = b_0c_1 - b_1c_0
$$
$$
B_x = \frac{1}{\Delta\,\ell}\sum_i A_i c_i, \qquad B_y = -\frac{1}{\Delta\,\ell}\sum_i A_i b_i,
\qquad \ell = 10^{-3}\ \mathrm{m/mm}
$$

Há duas implementações:
- `ans_b_smoothing.element_b` (float64): é a base da suavização;
- `ans_parsing._element_b_from_A`: é o gabarito do torque, calculado com $A$ convertido
  para float32 antes da derivada (`elem_structure.parse_elem_structure(target_field='B')`).

### 2.2 Suavização do FEMM (porte do xfemm)

`ans_b_smoothing.nodal_b` → `b1, b2 [M,3]`: valor de $\mathbf B$ em cada vértice **de cada
elemento** (P1 descontínuo), como o `mo_smooth('on')` do FEMM. Para o vértice $k$ do elemento
$e_0$, $\mathrm{star}(k)$ = elementos incidentes em $k$.

**Mesmo material** (`_same_material`), entre os elementos $e$ e $n$: mesmo rótulo; ou mesmos
$\mu_x$, $\mu_y$, $H_c$ e mesma `magdir`; ou mesmo bloco e mesma `magdir`.

1. **Caso normal** (todos os elementos de $\mathrm{star}(k)$ do mesmo material): média por
   inverso da distância ao centróide, igual para todos os elementos do nó:
   $$\mathbf b(k) = \frac{\sum_{e\in\mathrm{star}(k)} w_e \mathbf B_e}{\sum_e w_e},\qquad w_e = \frac{1}{|\mathbf x_k - \mathbf c_e|}$$
2. **Interface**: a partir de $e_0$, varre $\mathrm{star}(k)$ nos sentidos anti-horário e
   horário pelos vizinhos que compartilham aresta até achar um elemento de **outro rótulo**.
   Para a aresta de interface $(k, p)$, de comprimento $\lambda$ e tangente unitária
   $\hat t$, com $e$ = último elemento do próprio lado:
   $$ z = \frac{1}{2\lambda},\quad b_t = \mathbf B_e\cdot\hat t,\quad b_n = \frac{A_p - A_k}{\lambda\,\ell},\quad
   \mathbf b \mathrel{+}= z\big(\hat t\, b_t + (t_y, -t_x)\, b_n\big) $$
   e $\mathbf b(k) = \sum z(\ldots)/\sum z$ sobre as interfaces achadas.
3. **Borda do domínio** (a varredura sai da malha): $\mathbf b(k) = \mathbf B_e$.
4. **Canto**: se as duas interfaces foram achadas e não são colineares
   ($-\hat t_1\cdot\hat t_2 \le 0{,}985$), então
   $\mathbf b(k) = \big(\max_{e\in\mathrm{star}(k),\,\mathrm{rót}(e)=\mathrm{rót}(e_0)} |\mathbf B_e|\big)\,\mathbf B_{e_0}/|\mathbf B_{e_0}|$.
5. Pontos com corrente pontual: $\mathbf B_e$ (não ocorre neste raw).

**Avaliação num ponto** (`point_b`, `smooth=True`): interpolação linear dos valores do
elemento que contém o ponto:
$$\mathbf B(\mathbf x) = \sum_{i=0}^2 \mathbf b_i\,\lambda_i(\mathbf x),\qquad
\lambda_i = \frac{a_i + b_i x + c_i y}{\Delta},\quad a_0 = x_1y_2 - x_2y_1 \ (\text{cíclico})$$

### 2.3 Material do nó e gabarito nodal

`ans_parsing._node_material_priority`: o material do nó é o de **maior prioridade** entre os
elementos incidentes (ferro > ímã > cobre > ar).
`smooth_layouts._node_b_by_material`: o valor nodal é a média dos valores do vértice $k$ nos
elementos incidentes **do material escolhido**:

$$\mathbf y_k = \frac{1}{|E_k|}\sum_{e\in E_k}\mathbf b_e(k),\qquad
E_k = \{e\in\mathrm{star}(k) : \mathrm{mat}(e) = \mathrm{mat}(k)\}$$

### 2.4 Referencial polar

`ans_parsing._to_polar`, com $\theta = \operatorname{atan2}(y, x)$ do ponto onde o valor está
(nó, centróide ou centro do pixel), calculado em float64 e gravado em float32:

$$ B_r = B_x\cos\theta + B_y\sin\theta,\qquad B_\theta = -B_x\sin\theta + B_y\cos\theta $$

### 2.5 Gabarito por arch e do torque

| Uso | Valor | Onde |
|---|---|---|
| `y_hw` (todas as archs) | $\mathbf B$ suavizado (`point_b`) no centro do pixel, polar | `smooth_layouts._grid_smooth_b` |
| `node_y`, FNO_GNN / GNN_PostBase / FNO_BipartiteGNN | média nodal por material (2.3), polar no nó | `smooth_layouts._parse_smooth` |
| `node_y`, FNO_BipartiteGNN_Elem | $\mathbf B$ suavizado no centróide ($\lambda_i = 1/3$: média dos 3 valores do elemento), polar | `elem_smooth._fill_elem_smooth` |
| torque `gt` | curl(A) exato P0 (2.1), cartesiano | `elem_structure.parse_elem_structure(target_field='B')` |
| torque `smooth` | o `node_y` da Elem levado de volta a $(B_x, B_y)$ | `torque.parse_torque_sample` |

Para pixels fora da triangulação, usa-se o elemento de centróide mais próximo (cKDTree), e o
campo linear dele é avaliado no pixel.

---

## 3. Layouts por amostra

Janela: $R_{in} = 28{,}5$, $R_{out} = 46{,}5$ mm (lidos do CSV), $\theta \in [0°, 120°]$.
Posição normalizada:

$$ r_{base} = \frac{r - R_{in}}{R_{out} - R_{in}},\qquad c_{base} = \frac{\theta - \theta_1}{\theta_2 - \theta_1} $$

### 3.1 Grade polar (todas as archs)

Grade **centrada em células** (`ans_parsing._grid_polar_xy`):
$r_i = R_{in} + (i+\tfrac12)\Delta r$, $\theta_j = \theta_1 + (j+\tfrac12)\Delta\theta$, com
$\Delta r = 18/138 = 0{,}1304$ mm e $\Delta\theta = 120°/276 = 0{,}4348°$. O pixel
$(i,j)$ tem $r_{base} = (i+\frac12)/H$ e $c_{base} = (j+\frac12)/W$.

| Chave | Shape | Conteúdo |
|---|---|---|
| `x_hw` | [3,H,W] | $\mu_r$, $M$, $J$ [A/mm²] do elemento que contém o centro do pixel (`trifinder`; fora da malha, elemento de centróide mais próximo) — `bipartite_structure._grid_const_per_element` |
| `y_hw` | [2,H,W] | $(B_r, B_\theta)$ [T], 2.5 |

### 3.2 Grafo de vértices — FNO_GNN, GNN_PostBase (`smooth_layouts._parse_smooth`)

| Chave | Shape | Conteúdo |
|---|---|---|
| `node_x` | [S,6] | $\mu_r$ do material do nó, $M$ do nó, área dual [mm²], $r_{base}$, $c_{base}$, $J$ do nó [A/mm²] |
| `node_y` | [S,2] | $(B_r, B_\theta)$, 2.3–2.4 |
| `edge_index` | [2,E] | linha 0 = origem, linha 1 = destino; os dois sentidos |
| `edge_attr` | [E,4] | $\Delta r$, $\Delta c$, $d$ [mm], $\Delta\mu$ |
| `x_hw`, `y_hw` | | 3.1 |
| `L` | [1] | $S$ |

Features de nó:
- $M_k$: só para nó de ímã, sinal de $\sum A_e$ dos ímãs $+1$ menos $\sum A_e$ dos $-1$
  incidentes (empate → $+1$); $0$ nos demais (`_node_magnet_polarity`).
- $J_k$: só para nó de cobre, $\sum_{e\in\mathrm{cu}} A_e J_e/\sum A_e$; $0$ nos demais
  (`_node_current_density`).
- Área dual: $A^{dual}_k = \sum_{e\ni k} A_e/3$ (`_node_dual_area`).

Arestas (`_build_bidirectional_edge_attrs`): arestas únicas da malha mais os pares de **wrap**
(`_wrap_edge_pairs`: nós em $\theta = 0°$ e $120°$ com tolerância de $10^{-3}$°, pareados por
$r$ ordenado; exige a mesma contagem nos dois cortes). Para a aresta $i\to j$:

$$ \Delta r = (r_{base,j} - r_{base,i})\,H,\quad \Delta c = (c_{base,j} - c_{base,i})\,W,\quad d = |\mathbf x_j - \mathbf x_i|,\quad \Delta\mu = \mu_i - \mu_j $$

$\Delta r$ e $\Delta c$ ficam em unidades de célula da grade. No wrap, $\Delta r = \Delta c = d = 0$
(os dois nós são o mesmo grau de liberdade do FEMM). O sentido inverso recebe
$(-\Delta r, -\Delta c, d, -\Delta\mu)$. Ordem das colunas de `edge_index`: todos os $i\to j$,
depois todos os $j\to i$.

### 3.3 Grafo bipartido — FNO_BipartiteGNN (`bipartite_structure.parse_bipartite_structure`)

Este parser monta só a estrutura; `node_y` e `y_hw` vêm dos parsers smooth.

| Chave | Shape | Conteúdo |
|---|---|---|
| `node_x` | [S,2] | $r_{base}$, $c_{base}$ dos vértices |
| `node_y` | [S,2] | como 3.2 |
| `edge_index`, `edge_attr` | [2,E], [E,3] | vértice–vértice: $\Delta r$, $\Delta c$, $d$ (sem $\Delta\mu$; wrap = 0) |
| `elem_x` | [M,6] | $\mu_r$, $M$, área [mm²], $r_{base}$ e $c_{base}$ do centróide, $J$ |
| `cross_edge_index` | [2,3M] | linha 0 = elemento, linha 1 = vértice (cada elemento → seus 3 vértices) |
| `cross_edge_attr` | [3M,1] | $|\mathbf x_v - \mathbf c_e|$ [mm] |
| `L`, `elem_L`, `E_L`, `C_L` | [1] | $S$, $M$, $E$, $3M$ |

### 3.4 Layout Elem — FNO_BipartiteGNN_Elem (`elem_structure.parse_elem_structure`)

Usa as mesmas chaves do bipartido, com **papéis trocados**: `node_*` = elementos (grafo
principal), `elem_*` = vértices (auxiliar). O contrato completo está na docstring do
módulo, e `check_elem_layout` confere que há 1 linha de `node_x` por elemento e
$C = 3M$.

| Chave | Shape | Conteúdo |
|---|---|---|
| `node_x` | [M,6] | $r_{base}$, $c_{base}$ do centróide, $\mu_r$, $M$, área, $J$ |
| `node_y` | [M,2] | 2.5 (centróide) |
| `edge_index`, `edge_attr` | [2,F], [F,3] | grafo dual: elementos que compartilham aresta, mais o wrap |
| `elem_x` | [S,2] | $r_{base}$, $c_{base}$ dos vértices |
| `cross_edge_index` | [2,3M] | linha 0 = vértice, linha 1 = elemento |
| `cross_edge_attr` | [3M,1] | distância centróide–vértice |
| `L`, `elem_L` | | $M$, $S$ |

Arestas duais (`_build_dual_edges`): aresta interna, com $\Delta r = (r^c_j - r^c_i)H$,
$\Delta c = (c^c_j - c^c_i)W$ e $d = |\mathbf c_j - \mathbf c_i|$. No wrap, o elemento $i$
(corte 0°) se liga ao $j$ (corte 120°) com $\Delta c = (c^c_j - 1 - c^c_i)W$ e $d$ medido com
$\mathbf c_j$ girado de $-120°$: distância real através do corte, não zero.

Com `target_field='B'` (torque): `node_y` = curl(A) exato $(B_x, B_y)$ por elemento e
`y_hw = None`.

### 3.5 Layout Split — FNO_SplitGNN (`split_layout.parse_split_smooth`)

Grafo dual de elementos mais os vértices **divididos por material**. Os dois tipos de nó
têm estado e saída. O contrato completo está na docstring do módulo, e `check_split_layout`
confere as contagens. As chaves são as do bipartido, mais `elem_y`: `node_*` = **N_aux**
(elementos), `elem_*` = **N_base** (cópias de vértice), `edge_*` = **Edge_dual**,
`cross_*` = **Edge_aux**.

**Construção:**
- **N_aux:** 1 nó por elemento, no centróide.
- **Edge_dual:** `elem_structure._build_dual_edges` (elementos que compartilham uma aresta,
  mais o wrap), nos dois sentidos, com a mesma ordem da Elem.
- **Fusão do corte:** os pares de vértices em 0° e 120° (`_wrap_edge_pairs`) viram um único
  vértice, o de 0°.
- **N_base:** 1 cópia por (vértice, classe), com classe = (material_id, $M$). Um vértice
  interno gera 1 cópia, uma fronteira de 2 materiais gera 2 e os raros vértices triplos
  geram 3. Por amostra: cerca de 31,7 mil vértices fundidos, 5,2 mil duplicados e 52
  triplicados, o que dá $K \approx 37$ mil. Separar ou não o ímã pela polaridade não muda
  a contagem, porque ímãs opostos não se tocam.
- **Edge_aux:** cada elemento liga aos seus 3 vértices, sempre na cópia do **seu** material
  ($C = 3M$). Não há aresta vértice–vértice: os materiais só se comunicam pelas Edge_dual.

**Coordenadas:** $r_{base}$ e o ângulo do setor levado a 0–360°, $\varphi = 2\pi c_{base}$,
como $(\sin\varphi, \cos\varphi)$, que é contínuo no corte. $c_{base}$ fica no chunk só para
a interpolação (coluna crua).
- Os deltas são calculados com essas coordenadas finais ($\Delta r_{base}$, $\Delta\sin$,
  $\Delta\cos$), então atravessam o corte pelo caminho curto sem regra especial.
- O comprimento [mm] usa as posições reais:
  - no wrap da Edge_dual, o centróide do outro lado é girado de $-120°$, como na Elem;
  - na Edge_aux, usa-se o vértice **original** do elemento, antes da fusão.

| Chave | Shape | Conteúdo |
|---|---|---|
| `node_x` | [M,7] | $r_{base}$, $c_{base}$ (crus), $\sin\varphi$, $\cos\varphi$, $\mu_r$, $M$, $J$ do elemento |
| `node_y` | [M,2] | $(B_r, B_\theta)$ suavizado no centróide (igual ao da Elem, 2.5) |
| `edge_index`, `edge_attr` | [2,F], [F,7] | Edge_dual, destino − origem: $\Delta r$, $\Delta\sin$, $\Delta\cos$, comprimento, $\Delta\mu$, $\Delta J$, $\Delta M$ |
| `elem_x` | [K,7] | $r_{base}$, $c_{base}$ (crus; no fundido, o lado de 0°), $\sin\varphi$, $\cos\varphi$, $\mu_r$, $M$, $J$ (média por área dos elementos da cópia) |
| `elem_y` | [K,2] | $(B_r, B_\theta)$ da cópia (abaixo) |
| `cross_edge_index`, `cross_edge_attr` | [2,C], [C,7] | linha 0 = cópia, linha 1 = elemento; vértice − centróide: $\Delta r$, $\Delta\sin$, $\Delta\cos$, comprimento; e $\mu_r$, $J$, $M$ do elemento |
| `x_hw`, `y_hw` | | 3.1 |
| `L`, `elem_L`, `E_L`, `C_L` | [1] | $M$, $K$, $F$, $C$ |

**Gabarito da cópia** (montado do raw na geração do dataset; o raw não muda):

$$\mathbf y_{(k,c)} = \frac{1}{|E_{k,c}|}\sum_{e\in E_{k,c}} \mathrm{polar}_{\theta_k^e}\big(\mathbf b_e(k)\big),\qquad
E_{k,c} = \{e \ni k : \mathrm{classe}(e) = c\}$$

Aqui $\mathbf b_e(k)$ é o valor nodal suavizado do FEMM (2.2), e $\theta_k^e$ é o ângulo do
vértice original. Fora do corte, isto é a regra 2.3 aplicada a cada material, e coincide com
o `node_y` do layout de vértices na cópia do material escolhido pela prioridade.

A média também resolve os dois casos em que o FEMM dá valores diferentes por elemento no
mesmo nó e mesmo material, um erro numérico conhecido do solver (10 amostras):
- **cantos:** cerca de 0,5% das cópias, com $\Delta$ médio de cerca de 0,5 T em ferro/ar;
- **vértices fundidos do corte:** cada lado foi suavizado separadamente; $\Delta$ médio
  abaixo de 0,02 T.

---

## 4. Datasets e chunks

### 4.1 Geração

`python -m scripts.build_datasets` → `elem_smooth.parse_all_layouts` por amostra: o `.ans`
é lido 1×, o bipartido e o $\mathbf B$ suavizado são calculados 1×, e a Elem é derivada do
bipartido. Os 5 layouts saem de uma passada.
- Saída: `data/torch/data_chunks/BLDC_138x276/<arch>/data_chunk_XXXX.pt`.
- 188 chunks por arch: 187 × 32 + 1 × 16 (`chunk_size = 32`, `MAX_SAMPLES = None`).
- A amostra $s$ vai para o chunk $\lfloor s/32\rfloor$, na mesma posição nos 5 datasets.
- Pool de 6 processos por chunk (`chunk_max_workers`); a retomada é por chunk e grava só
  as archs que faltam.
- `scripts.build_chunks_main` (4 archs) e `scripts.build_chunks_elem` (só a Elem, com o
  precheck da amostra 0) geram os mesmos chunks separadamente.
- O FNO_SplitGNN fica **fora** do `build_datasets` e é gerado por
  `scripts.build_chunks_split`:
  - parse de cerca de 1,4 s por amostra; cerca de 0,66 GB por chunk de 32, ou cerca de
    124 GB para os 188 chunks;
  - 20,3 MB por amostra, dos quais 79% são arestas: atributos de Edge_aux e Edge_dual com
    5,1 MB cada, e índices int64 com 2,9 MB cada;
  - uma época lê cerca de 107 GiB (treino + validação), contra cerca de 62 GiB da Elem. Isso
    passa um pouco do cache de disco da VM (cerca de 100 GiB), então parte dos chunks é relida
    do disco a cada época (ver as pendências no `CLAUDE.md`);
  - precheck da amostra 0 contra a Elem e o layout de vértices (3.5).

### 4.2 Composição do chunk (`src/data_gen/chunks.py::write_chunk`, `merge`)

| Layout (`LAYOUT_BY_ARCH`) | Arch | Regra |
|---|---|---|
| `grid` | FNO2d | `x_hw`, `y_hw` empilhados → [B,C,H,W] |
| `vertex` | FNO_GNN, GNN_PostBase | `edge_index` + offset acumulado de `L`; acrescenta `E_L` |
| `bipartite` | FNO_BipartiteGNN, Elem, FNO_SplitGNN | `edge_index` + offset de `L`; `cross_edge_index` linha 0 + offset de `elem_L`, linha 1 + offset de `L` |

As demais chaves são concatenadas no eixo 0, e `dim = (H, W)`. A escrita é atômica:
o arquivo vai para `.tmp/` e é renomeado no fim.

Os chunks atuais de FNO_GNN e GNN_PostBase em disco têm também `node_A` e `a_hw`, que o
código atual não gera e o loader ignora (não afetam o treino).

### 4.3 Split (`loaders.split_chunk_paths`)

Os caminhos são ordenados e embaralhados com `random.Random(split_seed=12)`; com
`train_split = 0,60` e `val_split = 0,30`:

$$ n_{train} = \mathrm{round}(188\cdot 0{,}60) = 113,\qquad n_{val} = \mathrm{round}(188\cdot 0{,}30) = 56,\qquad n_{test} = 188 - 113 - 56 = 19 $$

| Conjunto | Chunks | Uso |
|---|---|---|
| treino | 113 | ajuste dos pesos e stats da normalização |
| validação (`val`) | 56 | parada antecipada (GL/paciência) e escolha do `best.pth` |
| teste (`test`) | 19 (608 amostras) | só avaliação final no pós-processamento |

Os três são disjuntos e cobrem os 188 chunks. Como os nomes de arquivo são iguais nas
5 archs, o split também é. A run grava `split.json` com as três listas; o loader de treino
só lê treino e validação.

### 4.4 Leitura (`src/neural_op/dataloaders/loaders.py`)

`ChunkStreamDataset` (grid), `ChunkVertexDataset` (`vertex_collate`) e
`ChunkBipartiteDataset` (`bipartite_collate`):
- a ordem dos chunks é embaralhada a cada época e as amostras de cada chunk são permutadas;
- um buffer de reservatório de 64 amostras alimenta lotes de 32;
- uma thread pré-carrega 2 chunks;
- o modo `bipartite` repassa `elem_y` (fatiado por `elem_L`) quando o chunk tem essa chave
  (FNO_SplitGNN);
- o collate refaz os offsets de índice no lote, como em 4.2;
- `num_workers` é forçado a 0 no Windows.

---

## 5. Normalização (`src/neural_op/normalization.py`)

z-score por canal, com média e desvio **populacionais** calculados só sobre os chunks de
**treino** do dataset da run (`Normalizer.fit`). O cálculo é por streaming, com a fórmula de
combinação de Chan/Welford, e $\sigma = \sqrt{\max(\mathrm{var}, 10^{-12})}$:

$$ \tilde x_c = \frac{x_c - \mu_c}{\sigma_c} $$

- Chaves normalizadas: `x_hw`, `y_hw`, `node_x`, `node_y`, `elem_x`, `elem_y`, `edge_attr`,
  `cross_edge_attr` (as presentes no chunk), incluindo as arestas de wrap. `elem_y` só existe
  no FNO_SplitGNN.
- **Colunas cruas** ($\mu = 0$, $\sigma = 1$; tabela `_LAYOUTS`): as posições lidas pela
  interpolação grade → pontos, que exige $r_{base}, c_{base}\in[0,1]$:

  | Arch | Colunas cruas |
  |---|---|
  | FNO_GNN, GNN_PostBase | `node_x` 3, 4 |
  | FNO_BipartiteGNN | `node_x` 0, 1 |
  | FNO_BipartiteGNN_Elem | `node_x` 0, 1 e `elem_x` 0, 1 |
  | FNO_SplitGNN | `node_x` 0, 1 e `elem_x` 0, 1 (`elem_y` normalizado como `node_y`) |

  As posições dos centróides em `elem_x` do bipartido (colunas 3, 4) são normalizadas.
- `Normalizer.fit` falha se o nº de colunas divergir de `_LAYOUTS`.
- Cache: `data/torch/data_chunks/<dataset>/norm_stats_train_v2_<sha1 dos nomes dos chunks de treino>.json`.
  As stats vão para o `config.json` da run (`norm_stats`).
- Aplicação: `CUDAPrefetcher` → `Normalizer.encode_batch` na GPU, antes do `step_fn`. A
  inferência usa `inference._enc`, que falha se o nº de canais não bater com as stats.
