# Histórico — simetria 120° e malha real do FEMM como grafo (jul/2026)

> Movido de `CLAUDE.md` em 2026-10-11 (limpeza do CLAUDE.md), texto sem alteração. Referências "acima"/"abaixo" podem apontar para seções que agora estão em outro arquivo de `docs/claude/` — índice no `CLAUDE.md`. Cópia integral do original: `docs/CLAUDE_backup_2026-10-11.md`.

## Simetria 120° e malha real do FEMM como grafo (2026-07-22)

Investigação em andamento: comparar o custo de usar o **grafo qtree atual** vs usar a
**triangulação real do solver FEMM** como grafo (nós/arestas da própria malha, em vez de
folhas de quadtree). Protótipos em `tests/proto_femm_*.py` (não tocam `motor_model.py`).

### Simetria de 120° (`BLDC_FEMM_Model_Sym120` — `proto_femm_sym120.py`)

Motor fixo em 42 polos / 36 ranhuras. Matemática da simetria:
- passo de polo = 360/42 = 8,571°; passo de ranhura = 360/36 = 10°
- período combinado = 360/gcd(42,36) = 60°, mas 60° contém 7 polos (ímpar) → **anti-periódico**
- **120° contém 14 polos (par) e 12 ranhuras (12×10=120 exato) → período periódico
  (não anti-periódico) mínimo** — corta exatamente no centro da ranhura 0/12; polos não
  precisam ser cortados (polo 0 já começa em θ=0, `pole_offset=0`)

Correntes das bobinas são sempre 0 (`mi_addcircprop(..., 0, 1)` — cogging torque, sem
excitação), então a única fonte é o array de ímãs — simplifica a simetria.

**Resultado validado** (seed=42, malha completa vs setor): nós 110k→32,7k (**0,297x**),
`mi_analyze` 14,7s→3,5s (**~4,2x mais rápido**), erro máx. em A ~3,5%, em B ~8,5-9,5%
(erro concentrado em poucos pontos — malhas independentes geram ruído de discretização
normal ao comparar por interpolação; erro médio é <<1%).

**Truncamento radial** (`BLDC_FEMM_Model_Sym120_Annular` — `proto_femm_sym120_annular.py`):
tentativa de também restringir o domínio radialmente a `[inner_diameter/2, outer_diameter/2]`
(a mesma janela já usada pra amostrar dados) em vez de ir até r=0. **Testado e não compensa**:
`inner_diameter/2` (28,5mm) fica só 1,5mm dentro do furo do estator (30mm) — o corte remove
pouco, e o FEMM parece malhar mais denso perto da faixa anelar fina nova, aumentando nós/elementos
em vez de reduzir (32,7k→35k). **Mantido mesmo assim** (decisão do usuário) — arcos radiais
interno e externo usam contorno `A=0` (Dirichlet).

### Erros do FEMM encontrados ao montar o corte periódico (úteis pra próxima vez)

- `mi_addboundprop` com `BdryFormat=4/5` (periódico/anti-periódico) só aceita **exatamente 2**
  segmentos/arcos por nome de propriedade — erro `"assigned to more than two segments"` se
  reusar o mesmo nome em várias faixas radiais. Precisa de um nome único por faixa.
- `mi_addsegment(x1,y1,x2,y2)` **só conecta pontos que já existem como nó** — se nenhum dos
  dois extremos foi tocado por geometria anterior (ex: a origem), a chamada **falha em
  silêncio, sem erro, sem criar nada**. Corrigir sempre com `mi_addnode(x,y)` antes (no-op
  seguro se o ponto já existir).
- Região fechada sem `mi_addblocklabel`/`mi_setblockprop` **não** vira buraco automaticamente
  — dá erro `"Material properties have not been defined for all regions"`. Não existe
  "buraco" nativo por omissão; sempre precisa de um label (mesmo que vácuo).

### Extração de dados direto do arquivo `.ans` (`proto_femm_mesh_graph.py`, `proto_export_fem_mesh.py`)

`mo_getnode`/`mo_getelement` em loop custam ~30s pra malha de 110k nós (RPC/COM por
chamada, sem função de extração em lote). O arquivo `.ans` (escrito por `mi_analyze`,
texto puro) já contém a seção `[Solution]`: nós (`x y A`) seguidos de elementos
(`n1 n2 n3 label nbr1 nbr2 nbr3 circuito`, `n1/n2/n3` 0-based indexando a própria lista
de nós). Parse via `numpy.loadtxt` nessas linhas: **~0,2s** para 35-110k nós+elementos
(~130x mais rápido que o loop COM). `A` bate com `mo_geta` a `1e-13` de diferença.

- **Material por elemento**: o campo `label` de cada elemento é índice **0-based direto**
  em `[BlockProps]` (lista de materiais na ordem de criação, com duplicatas — cada
  `mi_getmaterial()` cria uma entrada nova mesmo pro mesmo nome). Classificar por
  `<BlockName>` + limiar de `<Mu_x>`/`<BHPoints>` (mesmo critério de `_material_ids_from_mu`:
  tem curva BH ou mu>10 → ferro, 1,01<mu<10 → ímã, mu<0,9995 → cobre, senão → ar), depois
  **substituir pelo valor constante da tabela antiga** (`PERMEABILITY`/`MATERIAL_ID`) em vez
  do mu local real (que pro ferro satura e varia por região, ex. 2160 vs 1775 no mesmo motor).
- **B (Bx,By) NÃO dá pra extrair só do arquivo de forma confiável.** `A` é solução nodal de
  elementos lineares (1ª ordem), então `B=curl(A)` é constante por triângulo e dá pra
  calcular por álgebra linear dos 3 vértices (fórmula padrão de gradiente de elemento
  linear) — **mas testado contra `mo_getb` em 60 elementos, ~47% divergem de forma
  não-trivial** (o FEMM parece suavizar/interpolar nodalmente o B reportado, não é o valor
  bruto por triângulo). B continua precisando de `mo_getb` ao vivo, por nó — é o único
  custo residual de COM por-elemento que sobra.

  **ATUALIZAÇÃO 2026-08-07 — causa raiz encontrada, afirmação acima estava incompleta**:
  os ~47% de divergência tinham duas causas identificáveis, nenhuma delas uma dificuldade
  real de diferenciação:
    1. **Unidade**: o gradiente foi avaliado com coordenadas em milímetros (convenção
       `mi_probdef` do projeto) sem converter pra metros — erra por um fator de ~1000
       (`dA/dx` "por mm" é 1000x menor que "por m", a unidade correta pro Tesla).
    2. **Suavização**: `mo_getb` por padrão devolve um campo suavizado/interpolado entre
       elementos vizinhos (`mo_smooth`, default aparentemente `'on'`, nunca setado
       explicitamente em nenhum script do projeto) — não o valor bruto por elemento.

    Testado em `tests/proto_femm_element_b_check.py` (2026-08-07): corrigindo as duas coisas
    — coordenadas em metros + `femm.mo_smooth('off')` antes da consulta — `mo_getb` bate com
    a fórmula fechada de gradiente de elemento P1 **a nível de erro de ponto flutuante**
    (~1e-16), em todos os pontos testados dentro de um mesmo elemento. **B *dá* pra
    reconstruir de A com precisão exata, 100% em numpy, sem nenhuma chamada COM** — desde que
    se use metros e `mo_smooth('off')`. Isso elimina, em teoria, o loop de `mo_getb` por nó
    (~3,5s/amostra) como custo residual de COM — ainda não migrado pra produção
    (`femm_mesh.py`), fica como possibilidade de otimização futura (ver "Pendências
    conhecidas").

    Efeito colateral a notar: o `node_y`/`y_hw` (Bx,By) já gravado nos 4000 chunks de
    `mesh_138x276_FEMM_MESH*` foi extraído via `mo_getb` sem chamar `mo_smooth`
    explicitamente — ou seja, é o campo **suavizado** (default do FEMM), não o valor bruto
    por elemento. Os dois são auto-consistentes entre si (groundtruth e comparações feitas
    até agora usam o mesmo método default), mas não são idênticos ao valor de elemento
    constante — importante saber qual dos dois um código novo está comparando.

    **O que é exatamente o B suavizado do nó?** Testado em 6 nós internos (grau 6-7,
    materiais variados): comparando `mo_getb(smooth='on')` no nó contra 3 candidatos de
    recuperação nodal calculados a partir do B exato (por elemento, Verificação 1 acima) dos
    elementos incidentes — **média simples (não-ponderada)** é consistentemente o candidato
    mais próximo (4-5 dos 6 nós testados; divergência típica 1e-4 a 1e-6 T), bem melhor que
    média ponderada por área ou por ângulo interno (divergência 1e-3 a 1e-2 T no mesmo teste).
    Não bate a nível de ponto flutuante como a Verificação 1 (FEMM é software fechado, sem
    acesso ao algoritmo real), mas caracteriza empiricamente o processo: o B do nó é
    aproximadamente a média simples do B constante dos elementos que o tocam — não ponderada
    por área nem por ângulo.
- **Material por nó (interfaces)**: um nó é compartilhado por vários elementos, que podem
  ter materiais diferentes (nó bem na fronteira ferro/ar). Resolvido por **votação
  ponderada por área** — material do nó = o de maior área total somada entre os elementos
  incidentes (mesmo princípio do `frac_dom` do quadtree pra células mistas). Vetorizado
  (`np.add.at`), custo desprezível.
- **Projeção grafo↔grade H×W**: como a malha do FEMM não é alinhada à grade regular (ao
  contrário do grafo qtree, que é construído por refinamento da própria grade), cada nó
  ganha `r_base`/`c_base` (mesma convenção normalizada [0,1] de `node_x[:,4]/[:,5]` do
  parser qtree), calculado direto de `(x,y)` contra os limites da janela de amostragem —
  numpy puro, sem custo. **Em aberto**: isso localiza o nó na grade, mas não elimina a
  necessidade de interpolar a saída do FNO (que vive nos pixels H×W) pra posição exata do
  nó — ainda não testado se essa interpolação é barata o suficiente.

### Benchmark completo (seed=42, `n_r=40 n_a=120`, `proto_export_fem_mesh.py`)

Comparação honesta — ambos os lados geram grafo **e** grade H×W (a versão anterior
esquecia a grade do lado malha-FEMM, favorecendo-a artificialmente):

| Etapa | Malha FEMM (setor+anelar) | Qtree (produção, motor 360° completo) |
|---|---|---|
| solve (mesh+analyze) | 4,24s | incluso |
| A+material+r_base/c_base (só arquivo) | 0,18s | — |
| B no grafo (`mo_getb` por nó, COM) | 4,42s | — |
| Bx/By/A na grade H×W (COM) | 1,24s | — |
| Mu_r/M na grade H×W (Shapely) | 1,80s | — |
| **TOTAL** | **11,9s** | **44,8s** |
| tamanho do arquivo | 2957 KB (.pt) | 1608 KB (12 CSVs) |

~3,8x mais rápido no total. Escala linear por amostra (cada design precisa de
`draw_motor`+`mi_createmesh`+`mi_analyze` do zero — sem amortização entre amostras).

### Promovido para produção (2026-07-23)

A investigação acima virou pipeline de produção: `BLDC_FEMM_Model_Sym120`/
`BLDC_FEMM_Model_Sym120_Annular` foram promovidas para `motor_model.py` (mesmos nomes,
mesma geometria); `src/data_gen/femm_mesh.py` reúne a extração do grafo
(`generate_mesh_sample`); `scripts/generate_data_femm_mesh.py` +
`scripts/build_data_chunks_femm_mesh.py` fazem a geração paralela (mesmo padrão
`ProcessPoolExecutor` janela+retry de `generate_data.py`) e o agrupamento em chunks. `tests/`
não foi tocado — os protótipos continuam lá como estavam. Ver contrato de dados completo em
"Formatos de dados" → "Chunks — malha real do FEMM" acima.

Três decisões que mudaram em relação ao protótipo original (`proto_femm_mesh_gendata.py`):

1. **Polaridade do ímã 100% FEMM, sem Shapely.** O protótipo usava point-query Shapely
   (`_magnet_sign_at_points`) porque `mu` sozinho (via `[BlockProps]`) não distingue N35p de
   N35n (mesma `mu_r=1.05`). Descoberta desta sessão, confirmada empiricamente num motor real
   (seed=42): cada um dos 14 polos do setor gera uma entrada **própria** em `[BlockProps]`
   (mesmo nome "N35", duplicada — o FEMM cria uma entrada nova a cada `mi_getmaterial()`,
   mesmo pro mesmo nome), na MESMA ordem em que os polos são desenhados — confirmado contando
   os elementos de cada polo por `label` (14 valores distintos, um por polo, na ordem de
   criação). Combinado com a paridade já conhecida do código
   (`direction = 0 if pole%2==0 else 180` em `BLDC_FEMM_Model_Sym120.draw_motor`), dá pra
   recuperar a polaridade de cada bloco só pela posição na lista ordenada de blocos
   classificados como ímã (`_block_magnet_polarity` em `femm_mesh.py`) — sem nenhuma consulta
   de geometria. **Fragilidade documentada**: depende da ordem de criação dos blocos
   permanecer sequencial; `_block_magnet_polarity` falha alto (`ValueError`) se a contagem de
   blocos de ímã não bater com `N_POLES_SECTOR`, em vez de silenciosamente dar polaridade errada.
2. **Wrap periódico θ=0↔120 resolvido.** Confirmado empiricamente que o FEMM garante a MESMA
   discretização radial nos dois cortes (mesmos `r`, diff ~1e-14) e resolve `A` como o mesmo
   grau de liberdade nos dois lados (diff `|A|` entre pares casados por `r` ~1e-11, ruído de
   ponto flutuante) — o pareamento por raio ordenado (`_wrap_edge_pairs`) é exato, sem
   heurística de tolerância espacial. As arestas de wrap entram no `edge_index`/`edge_attr`
   com `delta_r=delta_c=center_dist=0` (identificação do mesmo grau de liberdade, não um
   salto espacial real).
3. **`shared_length` removido de `edge_attr`** (não tem análogo real numa malha triangular —
   o protótipo usava placeholder 0; decisão do usuário 2026-07-23: remover em vez de manter
   zerado). `edge_attr` da malha fica com 4 colunas (`delta_r,delta_c,center_dist,delta_mu`),
   não intercambiável com os parsers qtree existentes (`FNO_GNN_PARSER`/`FNO_GNN_V2_PARSER`)
   — precisa de parser/arch próprios antes de treinar (ver "Pendências conhecidas").

**Bug encontrado e corrigido nesta sessão**: o limiar de classificação de cobre herdado dos
protótipos (`mu_x < 0.9995` sobre o `Mu_x` bruto do `[BlockProps]`) nunca disparava — o
material "copper" da biblioteca do FEMM tem `Mu_x=0.999991` (bem mais perto de vácuo do que
o `0.999` aproximado que o projeto usa depois de classificado), então todo nó de cobre caía
silenciosamente em ar/vácuo. Corrigido para `mu_x < 0.999995` em `_parse_block_materials`
(`femm_mesh.py`) — confirmado com contagem de material não-zero pra cobre num motor real.
Esse limiar antigo (`0.9995`) só existia nos protótipos de `tests/` (não afeta o pipeline
qtree/grid, que classifica material via Shapely direto, não via `[BlockProps]` do FEMM).

Staging BRUTO: `.npz` por amostra (`data/raw/<dataset>/sample_XXXXXX.npz`, escrita atômica
`.tmp.npz`→rename — mesma convenção de `sample_processor.py` do pipeline qtree), sem
`motor_params` (dict não é nativo de `.npz`; proveniência fica em `valid_designs.csv` via
`BLDC_Process.export_params`, igual pipeline qtree, mesmo diretório). **Corrigido 2026-07-27**:
antes ficava em `data/temp/samples_mesh/` (gitignored, "pode ser limpo a qualquer momento") —
incoerente com o papel de fonte única cara de gerar (1 solve FEM completo por amostra); agora
em `data/raw/`, junto do CSV do pipeline qtree/grid, permanente (4000 `.npz` já existentes
migrados, conteúdo idêntico). `data/temp/samples_npz/<dataset>_<npz_parser>/` (staging FINAL,
pós-parser, gerado pelo branch `mode='femm_mesh'` de `gen_npz_structures.py`) é
gitignored/descartável — pode ser regenerado a qualquer momento a partir do staging bruto de
`data/raw/`, sem custo de FEMM.

### Vazamento de memória no `generate_data_femm_mesh.py` (2026-07-23)

`scripts/generate_data_femm_mesh.py::multi_process` criava o `ProcessPoolExecutor` sem
`max_tasks_per_child=1` — diferente de `scripts/generate_data.py` (pipeline qtree), que já
usava esse parâmetro. Sem ele, o laço interno (`while fut2meta: ...`) realimenta a MESMA
pool enquanto houver amostras pendentes, então um único processo Python/FEMM roda centenas
ou milhares de ciclos `femm.openfemm()` → `mi_analyze()` → `femm.closefemm()` sem nunca ser
reiniciado pelo SO. Automação COM (pywin32) tem um vazamento residual conhecido por ciclo
que não é 100% liberado por `closefemm()` dentro do mesmo processo hospedeiro — RAM subia
indefinidamente até o processo crashar (encontrados `tmp_femm_mesh_*` vazios em `data/temp/`
de execuções interrompidas assim). Corrigido adicionando `max_tasks_per_child=1` (worker
reciclado a cada amostra), igual ao pipeline qtree.

### Point-query vs malha para a grade H×W — benchmark e limitações da API do FEMM (2026-07-23)

Investigação (a pedido do usuário) sobre se valia a pena substituir a grade H×W
derivada-da-malha por point-query ao vivo no FEMM (`mo_getpointvalues`, a única chamada COM
real por trás de `mo_getb`/`mo_geta`/`mo_getmu` — os três só fatiam o mesmo retorno dela,
então chamar os três separadamente por pixel triplica round-trips à toa).

**Timing** (seed fixo, resolução de produção, ~30k nós): grade via malha (trifinder +
interp. baricêntrica) = **0,39s**; grade via point-query (`mo_getpointvalues` 1x/pixel) =
**4,12s** — point-query é **~10,5x mais lento** nesse passo. Nos dois casos o gargalo real
do pipeline continua sendo o solve (`mi_analyze`, ~3,7s) e o loop de B por nó do grafo
(~3,5s), que são custos de COM obrigatórios independente dessa escolha.

**Fidelidade** (malha vs point-query, mesma amostra): `A` bate **exato** (`MAE≈0`) — faz
sentido, já que a interpolação baricêntrica na malha é literalmente a mesma função de forma
que o FEM usa internamente para avaliar `A` num ponto arbitrário do elemento. `Bx`/`By` têm
~1,5% de erro (ruído de discretização — o FEMM aplica alguma suavização nodal própria em
`mo_getb` que não é idêntica à interpolação baricêntrica simples a partir dos nós).

**`Mu_r` bruto diverge ~90%, mas não é bug**: point-query devolve o `Mu1` REAL/local após a
saturação não-linear do ferro no ponto de operação convergido, enquanto a malha usa o valor
CONSTANTE da tabela `PERMEABILITY` (5000 pro ferro) — comparação de valores brutos não é
justa. Repetindo a comparação por LABEL de material (mesmos limiares de
`_make_material_masks`: ferro `mu>10`, ímã `1,01<mu≤10`, cobre `mu<0,9995`, senão ar):
89,1% de acurácia global — ferro/ar/ímã classificam com **100% de recall**, mas **cobre
tem 0% de recall, sempre confundido com ar**. Causa: `Mu1` do cobre no ponto convergido
resolve **exatamente `1,0`** (idêntico a vácuo), não o `0,999991` da biblioteca do FEMM
usado pra classificar block-a-block antes do solve — o cobre é magneticamente equivalente ao
vácuo nesse problema (sem corrente nas bobinas), então o ponto de operação apaga essa
distinção. Testado também `Jm` (índice 8 de `mo_getpointvalues`, único outro candidato a
fonte de magnetização) — **sempre retorna `0`, inclusive dentro de ímãs**. Confirmado por
comparação direta com Shapely (point-in-polygon puro, sem malha nenhuma) na mesma janela/
resolução: `malha` vs `Shapely` = **100% idênticos** (0 pixels divergentes, mesmas larguras
de ranhura), confirmando que a malha reproduz a geometria contínua exatamente e que a
limitação é da API do FEMM (`mo_getpointvalues`), não do método de acesso.

**Conclusão / decisão** (usuário, 2026-07-23): `material_id`/`mu_r`/`M` continuam vindo
100% da malha (única fonte correta, cobre e magnetização não são recuperáveis via
point-query de jeito nenhum). `A`/`Bx`/`By` da grade **voltaram** a vir de point-query ao
vivo (aceitando o custo ~10x maior nesse passo) — ver `_query_grid_pointvalues` em
`femm_mesh.py` e nota na seção "Chunks — malha real do FEMM" acima.

### Periodicidade de amostragem angular — ranhuras "alternando" (2026-07-23)

Ao inspecionar o label de material renderizado (`imshow`), ranhuras de cobre apareciam com
duas larguras alternando (`9,10,9,10,...` colunas) mesmo sendo **geometricamente idênticas**
na malha real (largura angular medida direto em graus: desvio padrão `~1e-6°` entre as 12
ranhuras do setor). Causa: `N_A=270` colunas sobre 120° com `N_SLOTS_SECTOR=12` dá
`270/12=22,5` colunas por ranhura — fração **exatamente meio-inteira**, que força o
rasterizador (qualquer um — `trifinder` da malha OU `Point.covers()` do Shapely, testado e
confirmado idêntico nos dois) a arredondar ora pra cima ora pra baixo numa alternância
perfeita de período 2. Não é bug de nenhum método, é aliasing de amostragem puro. Corrigido
escolhendo `N_A` múltiplo de 12: **`270→276`** (`23` colunas/ranhura, exato), com `N_R`
escalado proporcionalmente (**`135→138`**) pra manter a mesma densidade de pixel nas duas
direções — dataset renomeado `mesh_135x270` → `mesh_138x276`. Confirmado após a mudança:
larguras de ranhura uniformes (`10,10,10,...`, com a exceção esperada da ranhura cortada
pelo período `θ=0/120°`, que aparece como `5+5` nas duas pontas do array, somando `10`
também). Datasets anteriores (`mesh_135x270`) não foram apagados nem regenerados — convivem
no mesmo `data/torch/data_chunks/`, nomes diferentes evitam mistura.

### Nós além da borda nominal do ferro externo, nos cantos do corte periódico (2026-07-23)

No dataset `mesh_138x276`, ~0,3% dos nós (91/29498 numa amostra) classificados como `ferro`
têm `r_base` um pouco além da linha nominal onde o ferro do rotor deveria terminar,
concentrados nos dois cantos do corte periódico (`θ≈0°` e `θ≈120°`, ~50 dos 91 pontos) com o
resto espalhado em baixíssima densidade pelo resto do setor. **Não é bug**: é o efeito
esperado de triangular um contorno CURVO (o círculo do raio externo do ferro) com segmentos
RETOS — qualquer malha FEM tem nós que caem uma fração além/aquém do raio nominal, mais
pronunciado nos cantos onde o corte radial reto encontra o arco curvo (elementos mais
distorcidos ali). Confirmado que a maioria desses nós tem `frac_dom<1` (nós de interface,
tocando mais de um material) e `mu_r=5000` correto — não é contaminação de valor, só a
posição nominal (`r_base`) desses nós de fronteira que fica ligeiramente além do esperado.

---
