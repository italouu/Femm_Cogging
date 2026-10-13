# Pipelines de dados (femm_mesh v1/v2/v3, unificado, smooth) e arquiteturas de grafo

> Movido de `CLAUDE.md` em 2026-10-11 (limpeza do CLAUDE.md), texto sem alteração. Referências "acima"/"abaixo" podem apontar para seções que agora estão em outro arquivo de `docs/claude/` — índice no `CLAUDE.md`. Cópia integral do original: `docs/CLAUDE_backup_2026-10-11.md`.

## Registro original da seção "Datasets e bateria OFICIAIS (2026-10-03)"

**Únicos datasets oficiais do projeto** — todos derivados do MESMO raw, com o MESMO
gabarito B (curl(A) por elemento P1 + média simples nos nós; grade por interpolação
baricêntrica), mesmo chunking (125 chunks × 32) e mesmo split (`split_seed=12`):

| Item | Caminho |
|---|---|
| Raw (fonte única) | `data/raw/mesh_ans_138x276/` — 4000 `sample_*.ans.gz` + `valid_designs.csv` (`mode='femm_mesh_v2'`, `sample_method='legacy'`, `distribution='uniform'`, `datagen_seed=12`) |
| Chunks (1 pasta por arch) | `data/torch/data_chunks/mesh_ans_138x276_unified/{FNO2d,FNO_GNN,GNN_PostBase,FNO_BipartiteGNN}/` — gerados por `python -m scripts.build_unified_ans_chunks_direct` |
| Bateria de treino oficial | `data/logs/mesh_ans_138x276_unified_best_mse_mae/` — `python -m scripts.run_best_configs` (4 archs × mse/mae) |

**Estado em 2026-10-03**: os chunks unificados **ainda não existem nesta máquina**
(precisam ser gerados a partir do raw ou copiados). Os logs da bateria oficial que
existem aqui foram interrompidos cedo pelo critério GL antigo (paciência 1, ver
`MonitorCfg`) — precisam ser refeitos com `gl_patience`/`min_epochs`/
`early_stop_patience` atuais antes de qualquer tabela de resultados.

**Preparação da bateria definitiva (2026-10-03)**: correções B1–B6 commitadas (chaves no
topo de `scripts/run_best_configs.py`); execução movida para a VM Linux via
`bash scripts/run_vm_pipeline.sh` — ver `docs/bateria_definitiva/ROTEIRO.md`. Fase 0, V3 e
B0 ainda NÃO rodados por completo (nesta máquina: Fase 0 só em 1 chunk, V3 com GNN_PostBase
estourando a RTX 4080 16 GB).

**Fase do rotor (achado 2026-10-05)**: o raw oficial `mesh_ans_138x276` foi gerado com o
rotor SEMPRE na fase 0 — a coluna `rotor_phase [deg]` do `valid_designs.csv` (0–17,14°) foi
sorteada mas nunca chegou ao desenho. Corrigido no código (ver "Fase do rotor no setor de
120°" abaixo); o raw/chunks/bateria ainda NÃO foram regerados com a fase aplicada.

**Removidos em 2026-10-03** (a pedido do usuário, para não haver confusão com os oficiais):
- Todos os datasets derivados antigos — `data/temp/samples_npz/*` e
  `data/torch/data_chunks/*`: `mesh_138x276_FEMM_MESH`, `mesh_138x276_FEMM_MESH_A`,
  `mesh_ans_138x276` (alvo A), `mesh_ans_138x276_B`, `mesh_ans_138x276_B_v3`,
  `mesh_135x270`, `mesh_138x276`, `mesh_138x276old`, `motor_default_v4_135x270`,
  `test_fixed_geometry_v4_135x270`.
- Todos os logs que treinaram nesses datasets — inclusive as baterias
  `mesh_138x276_FEMM_MESH_best_mse_mae` e `mesh_ans_138x276_B_best_mse_mae` (as das tabelas
  antigas em `.claude_scratch/*.json`, que comparavam gabaritos de B diferentes — ver
  "Datasets unificados a partir do raw mesh_ans_138x276").
- Os raws antigos (`data/raw/mesh_138x276/`, `mesh_135x270/`, `*_135x270/`) foram
  **mantidos**, mas não são oficiais.

Qualquer menção, no restante deste arquivo, a datasets/runs da lista acima é **registro
histórico** — esses dados não existem mais. Defaults de código que ainda apontam pra eles
(ex: `NnCfg.dataset='mesh_ans_138x276_B'` em `src/configs/training.py`) estão
desatualizados — sempre passar `dataset='mesh_ans_138x276_unified/<arch>'` explicitamente.

---

### Filosofia raw → npz → chunks (pipelines de geração de dados, 2026-07-27, unificado)

Princípio comum aos pipelines qtree e femm_mesh — cada etapa tem UMA responsabilidade,
sem sobreposição. Desde a unificação (2026-07-27), **os 3 comandos e as 3 pastas-base são
os mesmos para os dois modos** — só o conteúdo interno de cada etapa difere:

| Etapa | Contém | Onde (ambos os modos) |
|---|---|---|
| **raw** | TODOS os dados extraídos da simulação, sem filtro — fonte única, cara de gerar (CSV = 1 solve FEM; staging mesh = 1 solve FEM completo por amostra). Nunca reprocessada só porque um parser mudou. | `data/raw/<dataset>/` — CSVs (qtree/grid) ou `sample_*.npz` de staging bruto (femm_mesh) |
| **npz** | SÓ o dado final: parser já aplicado (seleção de colunas), metadados de rastreabilidade do raw já descartados, chaves já com nome definitivo. **Todo processamento/decisão de conteúdo acontece aqui.** | `data/temp/samples_npz/<dataset>/` (qtree/grid) ou `data/temp/samples_npz/<dataset>_<npz_parser>/` (femm_mesh — sufixo mantido pra não misturar parsers) |
| **chunks** | os `.npz` da etapa anterior concatenados/empilhados, **sem alterar chaves nem descartar dados** — concatenação pura. Só reconstrói `dim` (tupla) a partir de `dim_H`/`dim_W`, mecânica de serialização do `.npz` (que não tem tupla nativa), não decisão de conteúdo. | `data/torch/data_chunks/<dataset>/` (qtree/grid) ou `data/torch/data_chunks/<dataset>_<npz_parser>/` (femm_mesh) |

Comandos — **idênticos para os três modos** (`grid`/`qtree`/`femm_mesh`), cada script
despacha internamente por `DatagenConfig.mode`:

```
python -m scripts.generate_data        # raw
python -m scripts.gen_npz_structures   # npz (parser aplicado)
python -m scripts.build_data_chunks    # chunks (chama gen_npz_structures.run() + concatena)
```

`gen_npz_structures.py` ganhou um branch `if MODE == 'femm_mesh': return _run_femm_mesh(...)`
que aplica `PARSER_REGISTRY[npz_parser]` sobre o staging bruto (sem `ProcessPoolExecutor` —
é só filtro de colunas, não simulação). `build_data_chunks_femm_mesh.py` (chamado por
`build_data_chunks.py` quando `mode='femm_mesh'`) não gera mais o raw sozinho dentro de
`main()` — só chama `gen_npz_structures.run()` + `build()`, mesma simetria do
`build_data_chunks.py` do pipeline qtree/grid (que também não chama `generate_data.py`,
espera o raw já existir; geração do raw é comando explícito e separado —
`python -m scripts.generate_data`).

**Antes de 2026-07-27** o staging bruto do femm_mesh ficava em `data/temp/samples_mesh/`
(gitignored, "pode ser limpo a qualquer momento") — incoerente com seu papel de fonte única
cara de gerar (mesmo raciocínio que já valia pro CSV do qtree, em `data/raw/`). Corrigido:
staging bruto agora em `data/raw/<dataset>/`, junto de `valid_designs.csv` (que já ficava
lá). O script `apply_parser_femm_mesh.py` (etapa intermediária própria, introduzida horas
antes na mesma sessão) foi absorvido pra dentro de `gen_npz_structures.py` e comentado
(`# [REMOVIDO]`, não apagado) — existia só pra fazer o femm_mesh ter comandos/pastas
diferentes do resto do projeto sem necessidade real.

**Migração dos dados existentes (2026-07-27)**: os 4000 `.npz` de staging de
`data/temp/samples_mesh/mesh_138x276/` foram movidos (conteúdo idêntico, só mudança de
pasta) para `data/raw/mesh_138x276/`.

### Raw — malha real do FEMM v2 (`.ans` bruto, `mode='femm_mesh_v2'`, 2026-08-07)

Motivação: `mode='femm_mesh'` deriva `node_x`/`edge_index`/etc. na hora (`generate_mesh_sample`)
e descarta a malha original (`elems` — conectividade de elementos, nunca persistida em lugar
nenhum, nem no staging bruto) — sem ela, não dá pra recalcular nada depois (ex: `B` exato por
elemento, validado nesta sessão — bate com `mo_getb(smooth='off')` a `1e-14 T`) sem resimular
no FEMM. Contraria a própria filosofia "raw = tudo extraído da simulação, sem filtro" (ver
"Filosofia raw → npz → chunks" acima). `mode='femm_mesh_v2'` corrige isso guardando o `.ans`
**inteiro**, sem nenhuma extração — qualquer derivação futura parte só do arquivo.

**Saída**: `data/raw/<dataset>/sample_XXXXXX.ans.gz` — um `.ans` completo por amostra
(`save_ans_gzip_sample` em `femm_mesh.py`), sem CSV nem `.npz` intermediário. `valid_designs.csv`
continua gravado do mesmo jeito (`BLDC_Process.export_params`, mesmo diretório). **Sem `.fem`**
salvo (decisão do usuário 2026-08-07) — a geometria é 100% reconstruível a partir de
`motor_params` (já em `valid_designs.csv`) + `BLDC_FEMM_Model_Sym120_Annular.draw_motor()`
(determinístico), redundante manter os dois.

**Sem `mi_loadsolution()`/nenhuma chamada COM de pós-processamento** — só
`mi_createmesh()`+`mi_analyze()`, depois `femm.closefemm()` e cópia do arquivo que o solver já
escreveu em disco. Mais rápido que `mode='femm_mesh'` (que ainda faz o loop de `mo_getb` por nó,
~3,5s/amostra) — esse custo não existe mais aqui.

**Conteúdo do `.ans`** (medido em `tests/proto_femm_element_b_check.py`/investigação
2026-08-07, amostra típica ~35k nós / ~69k elementos, ~4,46 MB descomprimido):

| Seção | % do arquivo | Conteúdo |
|---|---|---|
| `[Format]` .. `[NumBlockLabels]` | ~0,8% | mesma definição de problema/geometria do `.fem` (materiais em `[BlockProps]`, boundary props, pontos/segmentos/arcos do desenho) — pequeno, mas necessário (classificação de material depende de `[BlockProps]`) |
| `[Solution]` — nós | ~51% | `x, y, A, <-1 constante>` por nó — a 4ª coluna é sempre `-1` em todas as amostras testadas, sem informação |
| `[Solution]` — elementos | ~48% | `n1, n2, n3, label, nbr1, nbr2, nbr3, circuito` por elemento — **`nbr1-3` (vizinhos) são `-1` em 99,7% dos casos** (achado 2026-08-07, corrige suposição anterior de que viriam "de graça": o FEMM não popula adjacência real pra problemas magnéticos), e `circuito` é sempre `0` (nenhuma bobina tem corrente neste projeto — cogging torque, sem excitação) |

**Compressão**: gzip padrão da stdlib, cópia em bytes crus (`'rb'`/`'wb'`, sem tradução de
encoding/newline) — formato portável, decomprime idêntico em qualquer SO. Testado remover as
colunas sem informação (4ª do nó, `nbr1-3`+`circuito` do elemento) antes de comprimir: só
~3% de economia adicional sobre o gzip puro (`1352 KB` vs `1396 KB` por amostra) — não compensa
a complexidade de reescrever linha a linha, então a compressão é sobre o `.ans` **completo**,
sem remover nada. Razão de compressão ~3,18x (`4,46 MB → ~1,4 MB`/amostra); projeção pras 4000
amostras: **~17 GB sem compressão, ~5,45 GB com gzip**. Quem for descomprimir/parsear depois
(Linux, ao que tudo indica) deve abrir com `encoding='utf-8'` **explícito** — nunca depender do
encoding padrão do SO (nesta sessão, `print()` no console Windows já quebrou por causa disso,
`cp1252` não é UTF-8).

**Status** (atualizado 2026-08-10): as 3 etapas existem e foram validadas ponta a ponta —
geração bruta (`generate_data_femm_mesh_v2.py`), parser/staging (`gen_npz_structures.py`,
branch `_run_femm_mesh_v2`) e chunks (`build_data_chunks_femm_mesh_v2.py`), mesmos 3 comandos
de sempre (`generate_data`/`gen_npz_structures`/`build_data_chunks`). Ver seção "Grafo duplo
vértices+elementos" abaixo para o parser e o design da GNN sobre esse formato. Testado só em
8/4000 amostras (smoke test) — rodar o dataset completo é a próxima pendência, não a
implementação em si (ver "Pendências conhecidas").

### Grafo duplo vértices+elementos (`mode='femm_mesh_v2'`, 2026-08-10)

Pipeline completo (parser + arquitetura) construído sobre o raw `.ans.gz` da seção anterior,
decidido em sessão de design (conversa 2026-08-10, ver histórico). Resolve um problema do
grafo único do `mode='femm_mesh'` (v1): um nó de vértice bem numa fronteira ferro/ar tinha que
**votar** por UM material só (`_node_material_stats`, argmax por área) — ambíguo, descarta a
informação do lado perdedor. A ideia: separar "posição/campo" (vértices) de "material"
(elementos) em dois grafos, e deixar a rede **aprender** a combinar os elementos exatos ao
redor de cada vértice, em vez de uma fórmula fixa decidir isso de antemão.

#### Parser — `src/data_gen/parsers/femm_mesh_v2.py::parse_ans_gzip_sample`

Deriva tudo do `.ans.gz` bruto, sem FEMM aberto, sem chamada COM — reaproveita as funções de
parsing de arquivo já existentes em `src/data_gen/parsers/ans_parsing.py` (`_parse_solution`,
`_parse_block_materials`, `_block_magnet_polarity`, `_build_edges`, `_element_areas`,
`_wrap_edge_pairs`, `_grid_polar_xy`), sem duplicar lógica. Relocado de
`src/data_gen/femm_mesh_v2.py` pra `src/data_gen/parsers/femm_mesh_v2.py` em 2026-08-13 — ver
"Serviço de parsing sem `import femm`" logo abaixo pro motivo completo (não é sobre
organização de pasta, é sobre remover uma dependência real de `femm`/pywin32 que esse caminho
100% numpy não deveria ter).

- **Grafo 1 (vértices/campo)** — nós da malha real, topologia idêntica à v1 (arestas da
  triangulação + wrap periódico θ=0/120°): `node_x=[r_base,c_base]` (só posição — sem
  material, sem votação nenhuma), `node_y=[A]` (alvo, valor nodal exato do `.ans`),
  `edge_attr=[delta_r,delta_c,center_dist]` (**sem `delta_mu`** — não tem `mu_r` em `node_x`
  pra subtrair).
- **Grafo 2 (elementos/centróides)** — 1 nó por triângulo, **sem arestas internas, nunca
  atualizado** (não há mensagem elemento→elemento nem vértice→elemento):
  `elem_x=[mu_r,M,area,r_base_centróide,c_base_centróide]` — tudo exato por elemento, zero
  ambiguidade (ao contrário do voto por área da v1).
- **Arestas cruzadas (elemento→vértice, unidirecional)** — conectividade **imediata**, sem
  raio/k-NN: cada elemento liga só aos seus 3 vértices (decisão do usuário 2026-08-10) —
  reaproveita a mesma incidência já usada em `_node_material_stats`, sem busca espacial nova.
  `cross_edge_attr=[center_dist]` (distância centróide↔vértice, mm).
- **Grade H×W** — `x_hw=[Mu_r,M]` constante por elemento (trifinder, mesma técnica de
  `_assign_mesh_to_grid` da v1, sem live point-query); `y_hw=[A]` via interpolação
  baricêntrica nos 3 nós do elemento que contém o pixel (mesma função de forma do FEM linear
  — já validado `MAE≈0` contra point-query em 2026-07-23; é o "regime 2" que existia
  comentado em `femm_mesh.py`, revivido aqui só pra `A`, já que não há sessão FEMM viva pra
  fazer point-query no v2).

**Achado desta sessão**: testado se `H_cAngle` (`[BlockProps]`) distingue polaridade do ímã
(N35p/N35n) direto no `.ans`, evitando a regra frágil de ordem de criação de
`_block_magnet_polarity`. **Não distingue** — todos os blocos de ímã têm `H_cAngle=0` no
arquivo bruto (a orientação é só geometria de desenho, não fica em propriedade de bloco). A
fragilidade documentada de `_block_magnet_polarity` (depende da ordem sequencial de criação
dos blocos) continua valendo também no v2.

Timing: ~1,2-5,7s/amostra (staging, puro CPU/numpy — sem COM; a primeira execução sequencial
completa das 4000 amostras, 2026-08-10/11, levou ~6h18min — mais que a estimativa original de
~80min, provavelmente por contenção de I/O/CPU da máquina usada). Paralelizado em 2026-08-13
(`ProcessPoolExecutor` persistente em `gen_npz_structures.py::_run_femm_mesh_v2`, ver entrada
de `scripts/gen_npz_structures.py` na árvore de arquivos acima) — validado byte-a-byte
idêntico ao resultado sequencial, mas não re-executado em escala (as 4000 amostras já
processadas sequencialmente não foram regeneradas, só testado num subconjunto pequeno).

#### Serviço de parsing sem `import femm` (2026-08-13)

Achado numa sessão de manutenção: `gen_npz_structures.py` (branch `_run_femm_mesh_v2`) importava
`parse_ans_gzip_sample` de `src/data_gen/femm_mesh_v2.py`, que por sua vez reaproveitava
funções puras (`_parse_solution` etc.) de `src/data_gen/femm_mesh.py` — mas `femm_mesh.py` tem
`import femm` (pywin32/COM) no topo, necessário só pras funções que abrem o FEMM de verdade
(`generate_mesh_sample`, `save_ans_gzip_sample`, `_query_grid_pointvalues`). Importar qualquer
coisa desse arquivo arrastava `femm` junto — contradizendo a própria promessa documentada do
pipeline `femm_mesh_v2` de rodar "SEM FEMM aberto, possivelmente em Linux depois" (ver seção
"Raw — malha real do FEMM v2" acima). Na prática não travava nada (o `pyfemm` do pip captura o
`import win32com.client` num try/except, falha graciosamente sem Windows) — mas é uma
dependência desnecessária (pacotes `femm`+`six`) pro caminho 100% numpy.

Corrigido com dois módulos novos, sem nenhum `import femm`/COM na cadeia inteira (verificado:
`import scripts.gen_npz_structures` não bota `'femm'` em `sys.modules`):
- `src/data_gen/parsers/ans_parsing.py` — as funções puras de parsing/geometria do `.ans`
  (`_parse_solution` .. `_assign_mesh_to_grid`), extraídas de `femm_mesh.py`.
- `src/data_gen/motor_constants.py` — `MATERIAL_ID`/`PERMEABILITY`/`MAGNETIZATION`/
  `N_POLES_SECTOR`, extraídos de `motor_model.py` (que também tem `import femm` no topo —
  `ans_parsing.py`/`parsers/femm_mesh_v2.py` usavam `BLDC_Process`/`BLDC_FEMM_Model_Sym120_Annular`
  só por essas 4 constantes, arrastando `femm` de novo por um nível acima).

`femm_mesh.py` e `motor_model.py` (que legitimamente precisam de `femm` pras suas próprias
funções COM) reimportam essas funções/constantes de volta — mesmos valores, sem duplicação,
sem quebrar nenhum consumidor existente (inclusive os vários `tests/proto_femm_*.py` que fazem
`from src.data_gen.femm_mesh import _parse_solution` etc. — continuam funcionando, porque o
reimport em `femm_mesh.py` reexpõe o nome no namespace do módulo).

#### Chunk final — `data/torch/data_chunks/<dataset>/data_chunk_XXXX.pt`

Sem sufixo de parser (não há `PARSER_REGISTRY` nesse pipeline — `parse_ans_gzip_sample` já
produz o formato final, não há seleção de colunas a fazer).

```
node_x:            [S_tot, 2]   float32  r_base, c_base
node_y:            [S_tot, 1]   float32  A
edge_index:        [2, E_tot]   int64    bidirecional — malha + wrap periódico
edge_attr:         [E_tot, 3]   float32  delta_r, delta_c, center_dist
elem_x:            [M_tot, 5]   float32  mu_r, M, area, r_base_centróide, c_base_centróide
cross_edge_index:  [2, C_tot]   int64    linha 0 = índice do elemento, linha 1 = índice do vértice
cross_edge_attr:   [C_tot, 1]   float32  center_dist (centróide↔vértice)
x_hw:              [B, 2, H, W] float32  Mu_r, M
y_hw:              [B, 1, H, W] float32  A
L / elem_L / E_L / C_L: [B] int64        nós / elementos / arestas / arestas-cruzadas por amostra
dim:               (H, W)
```

`build_data_chunks_femm_mesh_v2.py` cuida de **4 espaços de índice** ao concatenar (diferente
da v1, que só tem nó+aresta): `edge_index` offsetado por nó; `cross_edge_index` com as DUAS
linhas offsetadas por índices DIFERENTES (linha 0 por elemento, linha 1 por nó — mesmo offset
de nó que `edge_index` usa). Mesma mecânica no dataloader (`ChunkFemmMeshV2Dataset` +
`femm_mesh_v2_collate`, `src/neural_op/dataloaders/grid_loader.py`) — fatiamento por amostra
via `L`/`elem_L`/`E_L`/`C_L`, re-offset ao montar o batch.

`Normalizer` (`src/neural_op/normalization.py`) precisou de ajuste: `node_x` desse formato é
só posição (`[r_base,c_base]`, colunas 0,1) — os índices fixos `{0,2,3,4}` que a v1 excluía do
z-score (pensados pro layout de 5-9 colunas do grafo qtree/femm_mesh) estourariam `IndexError`
ou normalizariam `r_base`/`c_base` por engano (`grid_sample` precisa deles crus, em `[0,1]`).
Corrigido com `_NODE_X_FULLY_STRUCTURAL_ARCHS = {'FNO_BipartiteGNN'}` — exclui `node_x`
por inteiro do z-score pra esse arch, em vez dos índices fixos. `elem_x` passou a ser
normalizado por inteiro (`mu_r`/`M`/`area`/`r`/`c`, sem exclusão) — sem isso, `mu_r=5000`
(ferro) dominaria a escala de entrada da rede.

#### Arquitetura da GNN — `src/neural_op/archs/_blocks.py` / `femm_mesh_v2_gnn.py`

Decisão construída incrementalmente em conversa (2026-08-10, ver histórico) partindo da
mecânica já implementada em `EdgeConvLayer`/`GNN` (grafo único, ver seção "Formatos de
dados" → batch). Fórmula final, camada `l` do stack de vértices:

```
h_j^(l) = H^(l)( h_j^(l-1),  F1^(l)(vizinhos vértice),  F2^(l)(vizinhos elemento) )

F1^(l) = Σ_{i∈viz_vértice(j)} G1^(l)(e_ij)   · W1^(l)(h_i^(l-1))
F2^(l) = Σ_{e∈viz_elem(j)}    G2^(l)(dist_je) · W2^(l)(elem_x_e)
```

`G` (gate escalar, sigmoid) e `W` (transform linear, sem bias) são funções **separadas**,
multiplicadas por aresta, depois **somadas** (agregação fixa, sem peso) sobre toda a
vizinhança — só então `H` (MLP de atualização) entra, recebendo o estado próprio do nó mais os
dois agregados já prontos (framework padrão de Message Passing Neural Network — mensagem +
update, Gilmer et al.).

**F1 muda a cada camada por necessidade estrutural**: depende de `h_i^(l-1)`, que já foi
atualizado pela camada anterior — é o próprio mecanismo que propaga informação além do vizinho
imediato (camada `l` enxerga até `l` saltos de distância na malha). Pesos `G1`/`W1`/`H`
independentes por camada (não compartilhados).

**F2 — decisão: pesos próprios por camada, mesmo elem_x estático** (não há necessidade
estrutural como em F1, já que `elem_x` nunca muda — mas o usuário decidiu por isso mesmo
assim: "não estamos elaborando G e F suficientemente bem para esperar que modelem", ou seja,
dar mais um grau de liberdade compensa a simplicidade do gate/transform escalar). Custo
verificado desprezível antes da decisão: ~+346 parâmetros (~2% do total da GNN, ~13-16k) e
~+61M multiply-adds/amostra (~10% em cima do custo de F1, pequeno frente ao ramo FNO) —
decisão tomada por design, não por peso computacional.

Implementação: `EdgeMessageBlock` (`_blocks.py`) isola só a parte de mensagem (gate×transform
+soma) do `EdgeConvLayer` original — sem MLP de atualização — reaproveitada com pesos
independentes tanto pra F1 (vértice→vértice) quanto F2 (elemento→vértice, dimensões de entrada
diferentes: `elem_in_ch=5` em vez de `node_width`, `cross_edge_dim=1` em vez de `edge_dim=3`).
`BipartiteEdgeConvLayer` combina as duas mensagens + `upd_mlp` (`H`, entrada
`3*node_width` — próprio + F1 + F2). `BipartiteGNN` empilha `n_layers` dessas.
`EdgeConvLayer`/`GNN` originais (usados por `FNO_GNN`/`GNN_PostBase`) **não foram tocados**.

`FNO_BipartiteGNN` (`femm_mesh_v2_gnn.py`) segue o mesmo pipeline de `FNO_GNN`: FNO no `x_hw`
→ interpola no nó (`_interpolate_fno_to_nodes_v2` — mesma lógica de `_interpolate_fno_to_nodes`,
mas lendo `r_base`/`c_base` de `node_x[:,0]`/`[:,1]`, não `[:,3]`/`[:,4]` como no grafo
qtree/femm_mesh v1) → concatena com `node_x` → `BipartiteGNN` (recebendo também `elem_x` +
arestas cruzadas) → resíduo somado à predição do FNO. `FNO_BipartiteGNNConfig`
(`src/configs/training.py`) com `edge_dim`/`grid_in_ch`/`grid_out_ch`/`node_in_ch`/
`elem_in_ch`/`cross_edge_dim` todos auto-detectados (`_detect_chunk_dims` estendido). Registrado
em `ARCH_REGISTRY['FNO_BipartiteGNN']`, `loader_mode='femm_mesh_v2'`.

**Validado** (`tests/proto_femm_mesh_v2_arch_smoke.py`, 2026-08-10): pipeline completo
`NnCfg`→`build_loaders`→modelo→`step_fn`→`loss.backward()` rodando em GPU sobre chunks reais
(2 chunks pequenos, 8 amostras) — gradiente **finito e não-nulo confirmado separadamente em
F1 e F2** da primeira camada (`layer0.msg_vertex.W.weight.grad` e
`layer0.msg_elem.W.weight.grad`), prova de que a injeção elemento→vértice está de fato
conectada ao grafo computacional, não é um branch morto. `metric_fn` também validado
(mae_hw/mae_graph finitos, unidades físicas corretas via `Normalizer.decode`).

**Não implementado**: rodar as 4000 amostras completas end-to-end (staging já rodou 1x
sequencial pras 4000, chunks nunca foram construídos — ver "Pendências conhecidas").

#### Eval — `femm_mesh_v2_eval_fn` (2026-08-13)

Implementado (antes era stub `NotImplementedError`) em `src/neural_op/archs/eval.py` — mesmo
arquivo/padrão dos demais `*_eval_fn` do projeto (`femm_mesh_v2_gnn.py` só reimporta de lá,
mesma técnica de reexport usada em `femm_mesh.py`/`motor_model.py` acima).

`femm_mesh_v2_eval_fn` fatia a amostra pelos **4 espaços de índice** (`L`/`elem_L`/`E_L`/`C_L`
— nó/elemento/aresta/aresta-cruzada, cuidando do offset assimétrico do `cross_edge_index`:
linha 0 por elemento, linha 1 por nó), roda o modelo com `return_components=True` (separa
`fno_at_nodes` — FNO puro interpolado no vértice — de `delta`, a correção da GNN) e despacha
pra `_plot_femm_mesh_v2` (`eval_cfg.show_b=False`, padrão): scatter 3×4, GT/FNO@nós/GNN de `A`
nos vértices + painel `Δ = GNN − FNO@nós` (mostra o que a GNN de fato corrigiu) + erro nó-a-nó.

**`EvalCfg.show_b`** (`src/configs/eval.py`, padrão `False`): quando `True`, deriva
`B=curl(A)` por elemento como PÓS-PROCESSAMENTO do `A` (GT/FNO@nós/GNN) — `B` não é o alvo de
treino deste arch (que é sempre `A`), é só visualização. Despacha pra `_plot_femm_mesh_v2_b`.

**Achado não-óbvio ao implementar isso**: o chunk não tem coordenadas absolutas (`node_x` só
tem `r_base`/`c_base` normalizados) — diferente dos testes que já validavam `B=curl(A)`
(`tests/proto_femm_element_b_check.py`, bate com `mo_getb` a ~1e-16), que liam `x,y` reais
direto do `.ans`. Sem coordenadas reais, `_triangle_xy_from_edges` reconstrói a forma de cada
triângulo via lei dos cossenos, só a partir das distâncias reais (mm) já salvas em
`edge_attr[:,2]` (`center_dist`) + a conectividade elemento→3-vértices (`cross_edge_index`) —
resolvendo a ambiguidade de reflexão comparando com a orientação de `(r_base,c_base)` (que
preserva orientação em relação a `(x,y)` real, por ser composição de escalas positivas com a
mudança polar→cartesiana, jacobiano `r>0`).

**Isso só dá a FORMA do triângulo, não a rotação global** — 3 distâncias não determinam
orientação absoluta. `Bx,By` calculados nesse referencial local por elemento (`_element_curl_A`)
saem corretos EM MÓDULO, mas cada triângulo cai rotacionado por um ângulo arbitrário e diferente
dos outros (validado numericamente: `Bx,By` batem exato, a ~1e-9, com o `B` real rotacionado
pelo ângulo verdadeiro do triângulo — mas esse ângulo não é conhecido sem coordenadas reais).
Primeira tentativa (reportar `Bx,By` cartesiano direto) dava valores sem sentido físico —
descoberto testando contra um campo `A` afim conhecido (`B` deveria ser constante em toda a
malha, saiu espalhado). Corrigido decompondo em **`Br` (radial) / `Bθ` (tangencial)**
(`_element_radial_tangential`) — projeção via produto escalar com a direção de `∇r_base`
(calculada no mesmo referencial arbitrário via `_element_gradient`), invariante à rotação
desconhecida de cada elemento. `|B|=hypot(Br,Bθ)` é **exato** (invariante de rotação, não
depende da direção); `Br`/`Bθ` têm erro pequeno, decrescente com o refino da malha (validado
contra o campo afim conhecido: ~2,1% numa malha de teste grosseira ~250 elementos, ~0,13% numa
malha do tamanho de produção ~68k elementos — vem da aproximação P1 linear do gradiente de
`r_base`, que não é afim em `x,y` pois `r_base ∝ sqrt(x²+y²)`).

#### `node_cell_count` + projeção de 8 vizinhos (`FNO_BipartiteGNN_v3`, 2026-08-20)

Pedido do usuário: acrescentar ao grafo de vértices (1) quantos nós da malha caem na mesma
célula da grade base H×W que aquele nó, e (2) uma projeção das 8 células vizinhas (3×3 ao
redor, sem o centro) vindas da saída do FNO, com wrap angular e zero radial fora da grade.

Dividido em duas partes por necessidade técnica — a saída do FNO só existe em tempo de
*forward* do modelo (depende de pesos treinados), não em tempo de *parsing* (só o `.ans.gz`
bruto disponível):

- **Parser** (`src/data_gen/parsers/femm_mesh_v3.py::parse_ans_gzip_sample_v3`) — wrapper
  fino sobre `parse_ans_gzip_sample` (v2, reaproveitado sem duplicação): acrescenta só
  `node_cell_count` a `node_x`, que passa de `[S,2]` (r_base,c_base) para `[S,3]`
  (r_base,c_base,node_cell_count). Índice de célula: `floor(r_base·n_r)` clipado em `[0,n_r)`
  (SEM wrap radial — não é periódico); `floor(c_base·n_a) mod n_a` (COM wrap angular, mesma
  periodicidade θ=0/120° de `_wrap_edge_pairs`). Contagem via `np.bincount`, broadcastada de
  volta pra cada nó que compartilha a célula. Validado (2026-08-20, amostra real
  `mesh_ans_138x276/sample_000000`): 29498 nós em 11384 células ocupadas (de 38088 totais),
  contagem por célula 1–79 (mediana 1) — distribuição bem não-uniforme (nós de malha se
  concentram perto de feições geométricas), não um bug.
- **Arquitetura** (`src/neural_op/archs/femm_mesh_v3_gnn.py::_interpolate_fno_to_nodes_v3`) —
  além do valor bilinear do FNO no ponto exato do nó (`fno_at_nodes`, idêntico ao v2), calcula
  a projeção das 8 células vizinhas por INDEXAÇÃO DIRETA (nearest, não bilinear — decisão do
  usuário: célula é unidade discreta da grade), wrap em `c` (coluna/ângulo)/zero em `r`
  (linha/radial) fora de `[0,H)`. `neighbor_proj [S,8·C]` concatenado com `node_x`/
  `fno_at_nodes` como entrada da `BipartiteGNN` (`in_node_features = node_in_ch + grid_out_ch +
  8·grid_out_ch`).

Registrado como arch **separada** (`FNO_BipartiteGNN_v3`, `ARCH_REGISTRY` — nomeada
`FNO_BipartiteGNN_v2` até mais tarde no mesmo dia, renomeada pra `_v3` a pedido do usuário
pra alinhar com o nome do parser `FEMM_MESH_V3`) — decisão do usuário, pra não alterar o
contrato de entrada de `FNO_BipartiteGNN` (runs já treinadas continuam intactas). `loader_mode` continua `'femm_mesh_v2'` (mesmo layout de chunk, só
`node_x` muda de largura — `ChunkFemmMeshV2Dataset`/`femm_mesh_v2_collate` já são agnósticos a
isso, nenhuma mudança necessária ali). `step_fn`/`metric_fn`/`eval_fn` reaproveitados de
`femm_mesh_v2_gnn.py`/`eval.py` sem duplicação — são genéricos, só repassam `node_x` pro
`forward` do modelo, não assumem largura fixa.

Novos parsers no `PARSER_REGISTRY`: `FEMM_MESH_V3`/`FEMM_MESH_V3_A` (mesmo mecanismo de
`FEMM_MESH_V2`/`FEMM_MESH_V2_A` — só carregam `target_field`, repassado a
`parse_ans_gzip_sample_v3` por `scripts/gen_npz_structures.py::_run_femm_mesh_v2`, que agora
seleciona a função de parse por um dict `npz_parser -> parse_fn` local ao módulo). Continua
usando `mode='femm_mesh_v2'` (mesmo raw `.ans.gz`, mesmos scripts de geração/chunking — só o
`npz_parser` muda) — `DatagenConfig.femm_mesh_v2_dataset_name` ganha sufixo `_v3` pra esses
dois parsers, pra não misturar pasta com os chunks v2 (largura de `node_x` diferente).

`Normalizer` (`src/neural_op/normalization.py`): `_NODE_X_PARTIAL_STRUCTURAL_ARCHS =
{'FNO_BipartiteGNN_v3': {0,1}}` — exclui só as colunas 0,1 (r_base,c_base, cruas, usadas por
`_interpolate_fno_to_nodes_v3` pra indexação de célula) do z-score; a coluna 2
(`node_cell_count`) é normalizada normalmente (diferente de `FNO_BipartiteGNN`, onde `node_x`
é 100% estrutural/excluído).

**Validado** (smoke test ad-hoc, 4 amostras reais de `mesh_ans_138x276/`, 2026-08-20):
`node_in_ch` auto-detectado como 3; `Normalizer.exclude['node_x'] == {0,1}`; forward+backward
com gradiente finito e não-nulo em F1, F2 **e no `lift` da GNN** (confirma que
`node_cell_count`/`neighbor_proj` participam do grafo computacional, não são branch morto);
`metric_fn` roda sem erro. Pipeline completo (staging + chunks em escala) ainda não rodado —
mesma pendência que `FNO_BipartiteGNN`/v2 antes dele (ver "Pendências conhecidas").

### Datasets unificados a partir do raw `mesh_ans_138x276` (2026-10-01)

**Raw de origem único: `data/raw/mesh_ans_138x276/`** (4000 `sample_*.ans.gz` +
`valid_designs.csv`, `mode='femm_mesh_v2'`). Todos os 4 datasets abaixo saem só dele.

**Motivação.** A tabela de resultados das 4 archs principais (erro ponto a ponto e por
integral de superfície, `.claude_scratch/surface_integral_*.json`/
`all_runs_two_metrics_results.json`) comparava FNO2d/FNO_GNN/GNN_PostBase treinados em
`mesh_138x276_FEMM_MESH` (raw v1, `data/raw/mesh_138x276/`) contra FNO_BipartiteGNN treinado
em `mesh_ans_138x276_B` (raw v2). Rastreado nesta sessão: os dois raws são as **mesmas 4000
simulações** (`valid_designs.csv` idênticos byte a byte; mesma malha, mesmo nº de nós; `A`
nodal igual a ~1e-10; `x_hw` idêntico; mesmo chunking 125×32 e `split_seed=12` → mesmo
split de teste) — **mas o gabarito B é diferente**:

| | v1 (`mesh_138x276_FEMM_MESH`) | v2 (`mesh_ans_138x276_B`) |
|---|---|---|
| B no nó | `mo_getb` ao vivo, suavização padrão do FEMM | curl(A) exato por elemento P1 + média simples dos elementos incidentes |
| B na grade | point-query `mo_getpointvalues` | interpolação baricêntrica do B nodal |

Diferença medida (chunk 0, nós): mediana |Δ|B|| ≈ 0,0012 T, média 0,047 T, p95 0,40 T, máx
1,3 T; correlação 0,97 por componente (mesma convenção cartesiana) — concentrada nas
interfaces, justamente onde está a maior parte do erro dos modelos. B_ref também difere
(0,7749 vs 0,7507). Ou seja, a comparação antiga não usava o mesmo gabarito.

**Processamento** (`scripts/build_unified_ans_datasets.py` +
`src/data_gen/parsers/femm_mesh_unified.py::parse_ans_gzip_sample_unified`): cada `.ans.gz`
é parseado uma vez, sem FEMM/COM, e gera os 4 layouts com **o gabarito B do v2** (única
definição de B reconstruível a partir do `.ans` — o B suavizado do FEMM não é reproduzível
fora dele):

| Pasta (`<arch>`) | Layout | Origem |
|---|---|---|
| `FNO2d/` | só grade: `x_hw` [2,H,W] (Mu_r,M), `y_hw` [2,H,W] (Bx,By) | `x_hw`/`y_hw` do layout bipartite |
| `FNO_GNN/` | layout v1 já filtrado pelo `FEMM_MESH_PARSER`: `node_x` [S,5] (mu_r,M,node_dual_area,r_base,c_base), `node_y` [S,2], `node_A`, `edge_index`, `edge_attr` [E,4], `x_hw`, `y_hw`, `a_hw`, `L`, `E_L` | votação de material por área (`_node_material_stats`/`_node_magnet_polarity`/`_build_bidirectional_edge_attrs`, mesmo código de `generate_mesh_sample`); `node_y`/`y_hw` do layout bipartite; `a_hw` baricêntrico |
| `GNN_PostBase/` | idêntico a `FNO_GNN/` (cópia — decisão do usuário: uma pasta por arch) | idem |
| `FNO_BipartiteGNN/` | `parse_ans_gzip_sample(target_field='B')`, sem mudança | idem `mesh_ans_138x276_B` |

Caminhos (filosofia raw → npz → chunks mantida):
- npz: `data/temp/samples_npz/mesh_ans_138x276_unified/<arch>/sample_XXXXXX.npz`
- chunks: `data/torch/data_chunks/mesh_ans_138x276_unified/<arch>/data_chunk_XXXX.pt`
  (concatenação pura reaproveitando os `build()` de `build_data_chunks.py` [FNO2d],
  `build_data_chunks_femm_mesh.py` [FNO_GNN/GNN_PostBase] e
  `build_data_chunks_femm_mesh_v2.py` [FNO_BipartiteGNN] — só os diretórios globais de
  entrada/saída são trocados temporariamente durante a chamada).
- `chunk_size=32`, mesma ordem de amostras nas 4 pastas → mesmo split treino/teste.
- Treino: `NnCfg(dataset='mesh_ans_138x276_unified/<arch>', ...)` (glob de
  `_detect_chunk_dims`/`Normalizer` aceita subpasta). Disco: ~105 GB de npz + ~105 GB de
  chunks.

```
python -m scripts.build_unified_ans_datasets       # etapa 1 (npz) + etapa 2 (chunks), retomável
python -m scripts.build_unified_ans_chunks_direct  # alternativa: raw -> chunks direto, sem npz
```

**Variante direta** (`build_unified_ans_chunks_direct.py`, 2026-10-01): pula o staging
`.npz` (desvio consciente da filosofia raw → npz → chunks, a pedido do usuário, pra
economizar ~105 GB de disco). Por chunk: parse paralelo das 32 amostras (pool novo por
chunk — reciclagem de worker), concatenação com offsets e gravação via os `_flush()` dos
builders existentes, escrita atômica em `<arch>/.tmp/` → rename; retomável (chunk presente
nas 4 pastas é pulado). Validado: 2 chunks × 4 archs **idênticos** (`torch.equal` em todas
as chaves) aos gerados pelo caminho via npz; ~10,5 s/chunk (6 workers) → ~22 min
estimados pros 125 chunks. Disco só dos chunks (~105 GB).

**Validado (2026-10-01)**: amostra 0 — layout v1 bit a bit idêntico ao staging antigo
`mesh_138x276_FEMM_MESH` em `node_x`/`edge_index`/`edge_attr`/`x_hw`/`node_A`/`L` (só
`node_y`/`y_hw` mudam, por construção; `a_hw` baricêntrico vs point-query antigo: 2e-10);
layout bipartite 100% idêntico ao staging `mesh_ans_138x276_B`. Smoke test de 64 amostras
(2 chunks/arch, em pasta temporária): `y_hw`/`node_y` das 4 pastas idênticos entre si e ao
`mesh_ans_138x276_B`; chunks bipartite diferem do `mesh_ans_138x276_B` antigo só por ruído
de ponto flutuante (≤2e-16 em `r_base`, ≤5e-14 em `delta_r`, <0,05% dos valores —
provavelmente chunks antigos gerados em outra máquina/libm). ~1 s/amostra no parse.

### Chunks — malha real do FEMM (`mode='femm_mesh'`, 2026-07-23)

`data/torch/data_chunks/<dataset>_<npz_parser>/data_chunk_XXXX.pt` (nome combinado
`DatagenConfig.parsed_dataset_name`, pra não misturar chunks de parsers diferentes) — mesmo
diretório-base do pipeline qtree/grid, mas **layout diferente**: gerado por
`src/data_gen/femm_mesh.py` + branch `mode='femm_mesh'` de `scripts/gen_npz_structures.py` +
`scripts/build_data_chunks_femm_mesh.py`, sem CSV nem quadtree. Já tem parser
(`PARSER_REGISTRY['FEMM_MESH']`, ou `FEMM_MESH_A` pra alvo escalar A) e já treina em
`FNO_GNN`/`GNN_PostBase` **sem precisar de arch nova** — `edge_dim`/`grid_in_ch`/`grid_out_ch`/
`node_in_ch` do GNN interno são auto-detectados a partir dos chunks reais (`_detect_chunk_dims`,
ver `NnCfg.__post_init__` em `training.py`, seção "Canais/dimensões auto-detectados" acima), não
mais hardcoded; `node_x[:,3]/[:,4]` (r_base/c_base) já bate com a convenção usada por
`_interpolate_fno_to_nodes`. Ver "Pendências conhecidas" pro que ainda falta (validação
end-to-end com o dataset completo).

Layout do chunk final (já filtrado pelo `FEMM_MESH_PARSER` — ver tabela raw/npz/chunk abaixo
para o que existe em cada etapa anterior):

```
node_x:     [S_tot, 5]     float32  mu_r, M, node_dual_area, r_base, c_base
node_y:     [S_tot, 2]     float32  Bx, By
node_A:     [S_tot]        float32  A (potencial vetor, valor nodal exato do .ans)
edge_index: [2, E_tot]     int64    bidirecional — malha real + wrap periódico θ=0↔120
edge_attr:  [E_tot, 4]     float32  delta_r, delta_c, center_dist, delta_mu (SEM shared_length)
x_hw:       [B, 2, H, W]   float32  Mu_r, M — derivado da malha (trifinder, valor
                                     constante do elemento; ver _assign_mesh_to_grid).
                                     NÃO vem de point-query — ver nota 2026-07-23 abaixo.
y_hw:       [B, 2, H, W]   float32  Bx, By  — point-query AO VIVO no FEMM
                                     (mo_getpointvalues; ver _query_grid_pointvalues,
                                     reversão de 2026-07-23, mesmo dia — ver nota abaixo)
a_hw:       [B, H, W]      float32  A       — idem Bx/By, point-query ao vivo
L:          [B]            int64    nós por amostra
E_L:        [B]            int64    arestas (bidirecionais) por amostra
dim:        (H, W)
```

**`node_y`/`y_hw` com parser `target_field='A'`** (`FEMM_MESH_A_PARSER`, 2026-08-05): a tabela
acima é o layout com `target_field='B'` (padrão, `FEMM_MESH_PARSER`). Com `FEMM_MESH_A_PARSER`,
`node_y` vira `[S_tot, 1]` (A, não Bx/By) e `y_hw` vira `[B, 1, H, W]` (idem) — construídos a
partir de `node_A`/`a_hw` em `_apply_parser_femm_mesh` (`scripts/gen_npz_structures.py`). `node_A`/
`a_hw` continuam no chunk mesmo assim (não removidos, redundantes com o novo `node_y`/`y_hw`) —
necessário porque `eval.py::fno_gnn_eval_fn` usa `'node_A' in d` pra decidir se despacha pro plot
de malha (`_plot_fno_gnn_mesh`); ver essa função para o suporte a alvo escalar.

`x_hw`/`y_hw`/`a_hw` (sem sufixo `_grid`, 2026-07-27): `generate_mesh_sample` já grava com
esse nome na fonte — femm_mesh não tem variante "suavizada" separada (ao contrário do qtree,
onde `x_hw`/`x_hw_grid` coexistem com significados diferentes — ver seção "Modo qtree"
acima), então o sufixo `_grid` só causava confusão e exigia rename tardio no agrupamento em
chunks. Os 4000 `.npz` de staging do dataset `mesh_138x276` já existentes foram migrados
in-place (chaves renomeadas, dados idênticos, escrita atômica) nessa mesma sessão.

Tabela completa por etapa (raw bruto tem TODAS as colunas; npz/chunk já filtrados pelo parser):

| Chave | Staging bruto (`data/raw/<dataset>/`) | `.npz` parseado (`data/temp/samples_npz/<dataset>_<npz_parser>/`) / chunk |
|---|---|---|
| `node_x` | `[S,9]` — material_id,mu_r,M,node_dual_area,r_base,c_base,frac_dom,0,0 | `[S,5]` (`FEMM_MESH_PARSER`: mu_r,M,node_dual_area,r_base,c_base) |
| `node_y` | `[S,2]` | `[S,2]` (target_field='B') **ou** `[S,1]` a partir de `node_A` (target_field='A', `FEMM_MESH_A_PARSER`) |
| `node_A` | `[S]` | `[S]` — passa intacto |
| `edge_index` | `[2,E]` | `[2,E]` |
| `edge_attr` | `[E,4]` | `[E,4]` — todas as colunas (mesh não tem `shared_length` a descartar) |
| `x_hw`/`y_hw`/`a_hw` | `[2,H,W]`/`[2,H,W]`/`[H,W]` | idem — sem filtro de canal |
| `L` | `[1]` | `[1]` |
| `dim_H`,`dim_W` | escalares | escalares no npz; viram `dim` (tupla) só no chunk |
| `r_in_mm`,`r_ext_mm`,`ang_1_deg`,`ang_2_deg` | escalares | **ausentes** — metadados de rastreabilidade do raw, descartados no branch `mode='femm_mesh'` de `gen_npz_structures.py` |

Diferenças em relação ao contrato qtree (`node_x`/`edge_attr` acima):
- `node_x[:,3]` (staging bruto) é `node_dual_area` (mm², área lumped do nó — soma de 1/3 da
  área dos triângulos incidentes), **não** `cell_area` normalizada por célula base (não
  existe "célula base" numa malha não estruturada).
- `edge_attr` tem 4 colunas, não 4-ou-5 — `shared_length` foi removido (sem análogo real
  numa malha triangular; decisão 2026-07-23).
- Sem `x_hw`/`y_hw` "suavizados" (estilo média-por-área do qtree) — só a variante nítida.
- `material_id`/`mu_r`/`M` do nó vêm de votação por área **100% dados do FEMM** (sem
  Shapely) — ver seção "Simetria 120°..." abaixo para os detalhes de como a polaridade do
  ímã é recuperada sem consulta de geometria.
- `material_id`/`mu_r`/`M` (nó **e** grade) são **100% livres de Shapely e de point-query
  live ao FEMM** — vêm só da malha já extraída do `.ans` (`_assign_mesh_to_grid` em
  `femm_mesh.py`). Isso é definitivo, não uma etapa transitória: testado e confirmado
  (2026-07-23, ver seção "Simetria 120°..." abaixo) que point-query **não tem como**
  recuperar essas colunas — `Mu1` do cobre resolve exatamente `1.0` no ponto convergido
  (indistinguível de vácuo/ar) e não existe campo de magnetização consultável por ponto
  (`Jm`, único candidato em `mo_getpointvalues`, sempre retorna `0` mesmo dentro de ímãs).
- `A`/`Bx`/`By` (só grade — `y_hw`/`a_hw`) **voltaram a vir de point-query ao
  vivo** (`mo_getpointvalues`, `_query_grid_pointvalues` em `femm_mesh.py`) — reversão
  parcial no mesmo dia 2026-07-23, a pedido do usuário, depois de validar por benchmark que
  não perde qualidade (`A` bate exato, `MAE=0`, contra a interpolação baricêntrica da malha;
  `Bx`/`By` têm ~1,5% de ruído de discretização) ao custo de ~10x mais tempo nesse passo
  específico (~0,4s → ~4s por amostra na resolução de produção — pequeno frente ao solve
  completo, ~4s, e ao loop de B por nó do grafo, ~3,5s, que já são chamadas COM obrigatórias
  de qualquer forma). O loop de B por nó do grafo continua sendo o único custo de COM
  estruturalmente necessário independente dessa escolha (B não dá pra reconstruir de A com
  precisão).

---


## Fase do rotor no setor de 120° (2026-10-05)

**Problema**: `BLDC_FEMM_Model_Sym120.draw_motor` desenhava os polos com `phase=0` literal
(ignorando `self.phase`) e `femm_mesh.save_ans_gzip_sample`/`generate_mesh_sample` criavam o
modelo com `phase=0` — herança dos protótipos de 2026-07-22 (`tests/proto_femm_sym120*.py`,
teste isolado setor × 360° com fase 0), promovidos "sem alterações de geometria" no commit
`3bd6afc` (2026-07-23) e copiados para o v2 em `7fdc216` (2026-08-13). `rotor_phase` continuou
sendo sorteado (`generate_samples_constrained`, `PHASE_MAX = 2·360/42`) e gravado no CSV, mas
nunca lido. Todos os raws de malha (`mesh_138x276`, `mesh_135x270`, `mesh_ans_138x276`) têm
fase 0; o pipeline qtree/grid (`generate_one_batch`) aplicava a fase.

**Correção** (só `mode='femm_mesh_v2'`; o gerador v1 `generate_mesh_sample` segue com fase 0):
- `BLDC_FEMM_Model_Sym120._pole_pieces` (`motor_model.py`): polo k em
  `θ = (k·passo + phase) mod 120°`; o polo que cruza 120° vira 2 pedaços (`[θ,120°]` e
  `[0°, θ+largura−120°]`) com a magnetização do polo INTEIRO (magdir = centro do polo, −120°
  no pedaço do início); pedaço < 0,01° é colado no corte. A periodicidade de 120° vale para
  qualquer fase (14 polos, número par). `_draw_periodic_cuts`: faixa dos ímãs em modo
  `"poles"` (reaproveita aresta de polo que caia em 0°/120°, senão desenha) no lugar de
  `"pole0"`. Com `phase=0` o desenho é o mesmo de antes.
- `save_ans_gzip_sample` (`femm_mesh.py`): `phase = motor_params['rotor_phase']`.
- `_parse_label_materials` (`parsers/ans_parsing.py`): material e polaridade por RÓTULO DE
  BLOCO. Achado: o campo `label` do elemento no `.ans` indexa `[BlockLabels]`, não
  `[BlockProps]` — o parser antigo só funcionava porque as duas listas saem com a mesma
  ordem/tamanho; e a polaridade pela ordem de criação (`_block_magnet_polarity`) exigia
  exatamente 14 blocos de ímã (quebra com polo partido = 15). Agora: rótulo → `blockprop` →
  material; polaridade = sinal de `cos(magdir − ângulo do rótulo)` (N35p para fora, N35n
  para dentro). Usado por `femm_mesh_v2.py` e `femm_mesh_unified.py` (chamadas antigas
  comentadas `# [REMOVIDO]`; as funções antigas continuam em `ans_parsing.py`).

**Validação** (`tests/validate_phase_sym120.py`, resultados em
`data/logs/_verificacao_fase/validate_phase.json`; inspeção visual em
`tests/inspect_femm_phase_sym120.py`, que pausa o FEMM antes de analisar):
- T1: 154 amostras do raw (fase 0) — leitura nova == antiga por elemento (material, μ, M).
- T2: `save_ans_gzip_sample` com fase 0 reproduz o raw atual (mesmos nós/elementos, |ΔA| 3e-15;
  só um magdir difere na 14ª casa decimal).
- T3: fase do CSV aplicada (amostras 0, 1, 2 — 14 ou 15 rótulos de ímã): parser unificado
  das 4 archs ok, wrap 0°/120° exato (|ΔA| ≤ 1,5e-11), M ∈ {−1,0,+1}.
- T4: período — φ=3° e φ+17,143° dão x_hw idêntico; A médio 0,009%, p99 0,13% (malhas
  diferentes: ordem de criação dos polos muda); B médio 4e-4 T.
- T5: setor × motor completo 360° com fase 0 / 3° / 5,55°: A médio 0,13–0,15%, B médio
  ~4,5e-3 T nos três casos (mesmo nível; o caso fase 0 é a referência do protótipo).

**Raw regerado (2026-10-05)**: `data/raw/mesh_ans_138x276/` apagado e refeito com o MESMO nome
(`python -m scripts.generate_data_femm_mesh_v2`, ~1 h, 8 workers); `valid_designs.csv`
idêntico ao anterior (mesmo seed/método — mesmas geometrias e fases). Checagem das 4000
(`data/logs/_regen_raw_fase/check_raw.py`): fase na posição esperada, 14/15 rótulos de ímã
(3000 amostras com polo partido), 7+7 polos de cada polaridade, wrap exato (|ΔA| ≤ 1,5e-10),
nós 27–38 mil. **Achado**: polo começando a <0,01° DEPOIS de 0° deixava fresta de vácuo de ~µm
entre o corte e o ímã — sample 943 (fase 8,5732° ≈ 1 passo) saiu com 78 mil nós. Corrigido
em `_pole_pieces` (início < 0,01° colado em 0°) e regeradas as 3 amostras afetadas
(913, 943, 3307). **Pendente**: chunks unificados e bateria.

## Materiais customizados do FEMM — reprodutibilidade (2026-10-07)

`iron_1008` (curva BH de 19 pontos, 0 → 2,5851 T), `vacuum` e `copper` (μ=0,999991, σ=0)
**não existem na biblioteca padrão do FEMM** — foram inseridos à mão no
`D:\femm42\bin\matlib.dat` da máquina original (o padrão tem `"Air"`/`"Copper"`); só `N35` é
da biblioteca padrão. Definições completas copiadas para
`motor_constants.py::FEMM_MATERIAL_DEFS` (conferidas idênticas ao matlib.dat e ao
`[BlockProps]` do raw oficial). As 13 chamadas `femm.mi_getmaterial(...)` de `motor_model.py`
viraram `self._get_material(...)` (antigas comentadas `# [REMOVIDO 2026-10-07]`):
- material na biblioteca local (matlib.dat da instalação registrada no COM `femm.ActiveFEMM`)
  → `mi_getmaterial`, como antes; se também estiver em `FEMM_MATERIAL_DEFS` e as propriedades
  diferirem → `ValueError` (evita reproduzir com outra curva BH sem perceber);
- fora da biblioteca → `mi_addmaterial` + `mi_addbhpoint`, uma vez por documento (o `.ans` sai
  com 1 entrada por material em vez de 1 por rótulo — o parser por rótulo não depende disso).
`BLDC_FEMM_Model.FORCE_EXPLICIT_MATERIALS=True` simula instalação limpa.
Validado (`tests/validate_femm_materials.py`, amostras 0 e 1 do raw oficial): via biblioteca o
`.ans` é idêntico ao raw (cabeçalho e solução); via definição explícita, malha, material/μ/M
por elemento, curva BH e `A` idênticos (ΔA = 0) — só o `[BlockProps]` muda de tamanho.

## Protótipo `FNO_BipartiteGNN_Elem` — B nos elementos (2026-10-07)

Bipartite com os papéis dos grafos TROCADOS: elementos = grafo principal (iterado, F1
elemento↔elemento pelo grafo dual + F2 vértice→elemento; saída Bx,By = curl(A) exato por
elemento P1), vértices = auxiliar estático, só com posição (`[r_base,c_base]`; `aux_fno=True` acrescenta o
FNO interpolado no vértice — desligado por decisão do usuário). Mesmo raw/chunking/split/hiperparâmetros do FNO_BipartiteGNN da
bateria; `x_hw`/`y_hw` idênticos aos da bateria (y_hw continua o B médio nodal baricêntrico);
a média nodal do alvo por elemento reproduz exatamente o `node_y` da bateria.

- Parser `src/data_gen/parsers/femm_mesh_elem.py` — chunk usa as CHAVES do femm_mesh_v2 com
  significado trocado (`node_*` = elementos `[r_c,c_c,mu_r,M,area]`/`[Bx,By]`, `elem_*` =
  vértices, `cross_edge_index` linha 0 = vértice, linha 1 = elemento; `L` = nº de elementos,
  `elem_L` = nº de vértices) → loader/collate/`_detect_chunk_dims`/step_fn/metric_fn da
  bipartite reaproveitados sem mudança. Wrap do dual: elementos com aresta no corte, casados
  via `_wrap_edge_pairs`; `edge_attr` com o centróide do outro lado girado de −120°.
- Arch `src/neural_op/archs/femm_mesh_elem_gnn.py` (`ARCH_REGISTRY['FNO_BipartiteGNN_Elem']`,
  `FNO_BipartiteGNN_ElemConfig`); `Normalizer` exclui `node_x`/`elem_x` cols 0,1 (posição).
  `eval_fn` é stub. `mae_graph` = MAE por elemento (não comparável com a bateria).
- VM, dois executáveis Python (cada um com precheck embutido, `--no-precheck` pula):
  `python -m scripts.build_elem_proto_chunks` (raw/disco + parser da amostra 0 conferido
  contra a bateria → chunks, retomável) e depois `python -m scripts.run_elem_proto` (GPU/125
  chunks + smoke de 1 batch com gradientes → treina mse e depois mae; `--losses` escolhe).
  Os `scripts/run_vm_elem_proto*.sh` estão descontinuados (só imprimem aviso);
  `tests/proto_elem_smoke.py` é opcional (local — `tests/` é gitignored). Chunks em
  `data/torch/data_chunks/mesh_ans_138x276_unified/FNO_BipartiteGNN_Elem/` (~47 GB), logs em
  `data/logs/mesh_ans_138x276_unified_elem_proto/`. Memória medida: ~0,56 GiB/amostra
  (batch 32 ≈ 18 GiB).

**Versão smooth (2026-10-10)** — mesma arch, gabarito do B suavizado do FEMM (coerente com
`mesh_ans_138x276_smooth/<arch>`): `src/data_gen/parsers/femm_mesh_elem_smooth.py` reaproveita
`parse_ans_gzip_sample_elem` e troca só `y_hw` (= `point_b(smooth=True)` no pixel, idêntico
ao da bateria smooth) e `node_y` (= B suavizado NO CENTRÓIDE = média dos 3 valores nodais
`b1,b2` do elemento; decisão do usuário, preferido a curl(A) P0 ou P1 [M,6]).
`python -m scripts.build_elem_smooth_chunks` (precheck da amostra 0 + chunks em
`data/torch/data_chunks/mesh_ans_138x276_smooth/FNO_BipartiteGNN_Elem/`, ~15 s/chunk) e
`python -m scripts.run_elem_proto_smooth` (importa `make_nn_cfg`/`precheck` de
`run_elem_proto.py`; `test_split=0,30` da bateria smooth; logs em
`data/logs/mesh_ans_138x276_smooth_elem_proto/`). Validado: precheck ok, smoke de 64
amostras (pasta `mesh_ans_138x276_smooth/_smoke_elem_smooth`) com forward/backward ok.
Treinado na VM (run_0001 mse: parou por GL na ép. 119, best 49; run_0002 mae: 350 épocas,
best 269; pico 18,8 GiB). Teste: `python -m tests.resultados.elem_smooth_eval [N]` (antes `scripts.pos_bateria_smooth_elem_eval`, hoje em `tests/`) (mesmas
64 amostras de `eval_smooth.json`; Elem → nó = média dos incidentes do material do nó;
saída `data/logs/mesh_ans_138x276_smooth_elem_proto/pos_bateria/eval_elem_smooth.json`).
Resultado 2026-10-10, nós vs gabarito smooth (L1_A/L2_A/pontual, % B_ref): Elem mae
3,41/5,52/6,56 × Bip mae 4,27/6,55/7,81; Elem mse 6,44/9,13/8,94 × Bip mse 7,16/9,93/9,66
(piso da projeção elem→nó 1,25/2,89/0,85). Números da Bip idênticos ao `eval_smooth.json`.

## Suavização de B do FEMM a partir do `.ans` (2026-10-07)

**Módulo**: `src/data_gen/parsers/ans_b_smoothing.py` — isolado (numpy + stdlib, sem
`import femm`, sem dependência do projeto), ainda NÃO commitado nem usado por nenhum pipeline.
Porte fiel de xfemm `34ac6cc` `cfemm/fpproc/fpproc.cpp`: `GetElementB` (PLANAR) →
`element_b`; `GetNodalB` (Frequency==0, regra de interface ccw/cw que compara só `lbl`,
canto agudo, corrente pontual) → `nodal_b`; `GetPointB` → `point_b(smooth=True/False)`;
`Ctr` → `ctr`; NumList/ConList → `build_conlist` (CSR). `load_ans` lê `.ans`/`.ans.gz`.
Saídas: `B1,B2 [M]` (P0) e `b1,b2 [M,3]` (P1 descontínuo — 3 valores nodais por elemento);
`nodal_b(..., return_rule=True)` devolve a regra usada por vértice (0 normal, 1 interface,
2 punt de borda, 3 canto, 4 corrente pontual). ~0,7 s/amostra.

**Validação contra o FEMM** (pyFEMM, `tests/` — gitignored):
- `tests/validate_ans_b_smoothing.py`: 30 000 pontos (amostras 0,1,2,943,3307; interior,
  interface ferro–ar, outras interfaces, canto; 2 pontos no meio + 3 colados nos vértices):
  erro rel. máx. 5e-16 (off) / 2,8e-15 (on).
- `tests/compare_ans_vs_femm_full_mesh.py`: TODOS os elementos das amostras 0 e 943, dois
  caminhos independentes — `path_ans` (só `.ans`) × `path_femm` (só FEMM: laço Lua DENTRO
  do FEMM via `mlab2femm('dofile(...)')`; o Lua do FEMM não tem I/O de arquivo, cada lote
  volta como string `%.17g`). P0: idêntico (erro 0); P1 nodal (recuperado com as
  baricêntricas REAIS dos pontos consultados): 7e-15 rel. Resultado em
  `data/logs/_verificacao_bsmooth/`. Armadilha achada: supor baricêntricas = W exatas dava
  ~1e-8 (cancelamento em `a+b·x+c·y` em elementos minúsculos longe da origem).
- Regras 2 (punt de borda) e 4 (corrente pontual) não ocorrem no raw — portadas, não validadas.
- Estimativa 4000 amostras: `.ans` 47 min (1 proc) / ~7 min (8 proc) / ~5 min (16 proc);
  FEMM laço Lua ~2,4 h (1 instância); FEMM 1 chamada COM por ponto ~30 h. Pool com 30
  processos deu `BrokenProcessPool` (8 e 16 ok).

**Achados que mudam o entendimento anterior** (corrigem "O que é exatamente o B suavizado do
nó?" acima, que dizia "≈ média simples"):
- B suavizado é P1 **descontínuo**: cada elemento tem seus 3 valores nodais. Nós internos a
  um material: valor único (média ponderada por 1/dist ao centróide; difere da média simples
  atual em mediana 0,002 T, máx 0,065 T). **17% dos nós** (interfaces) têm valores diferentes
  por lado — spread mediana 0,75 T, máx 3,2 T (amostra 0).
- **`mo_getb`/`mo_getpointvalues` exatamente num nó depende do HISTÓRICO**: `InTriangle` guarda
  o último elemento achado (`static int k`) e o teste inclui a borda — devolve o valor do
  elemento incidente que a busca testar primeiro. Confirmado no FEMM real (nó 27408, interface
  ferro–cobre, 5 valores diferentes conforme a consulta anterior). O v1 (`mo_getb` nó a nó)
  tinha esse lado arbitrário nos nós de interface. O FEMM NÃO define o B do nó.
- **Voto de material por área é serrilhado**: em ~50% das arestas sobre uma interface as duas
  pontas votam materiais diferentes (o `node_x` do FNO_GNN/GNN_PostBase atual já é assim).
  Figura: `data/temp/bsmooth_cmp/vote_jag.png`.

**Decisão do usuário (2026-10-07)**: novo gabarito B dos datasets da bateria (FNO2d, FNO_GNN,
GNN_PostBase, FNO_BipartiteGNN — Bipartite_Elem fica de fora por ora) virá do smooth do FEMM:
- **grade `y_hw`**: `point_b(smooth=True)` no centro do pixel, elemento via trifinder —
  substitui a interpolação baricêntrica da média nodal;
- **`node_y`**: **prioridade de material** — o nó fica do lado do material de maior
  prioridade entre os elementos incidentes; valor = média dos valores nodais (`b1,b2`) dos
  incidentes desse material. Substitui a média simples. Exemplo (amostra 0) em
  `data/temp/bsmooth_cmp/priority_example.png` (script `priority_example.py` na mesma
  pasta): arestas de interface com lado trocado 48–54% (voto) → 0% (ferro–*) / 3–6% (ar–ímã,
  ar–cobre, só pontos triplos); perfil na ponta dos dentes contínuo, sem o serrilhado.

**Consequências levantadas** (a considerar ao implementar):
1. Gabarito muda para as 4 archs → resultados não comparáveis com a bateria atual; refazer as
   8 runs (inclusive o FNO2d base do GNN_PostBase); norm_stats/B_ref novos (pasta nova).
2. Scripts de avaliação (`pos_bateria_*`, integral de superfície) precisam usar o mesmo
   gabarito — checar onde calculam B por conta própria.
3. `FNO_BipartiteGNN_Elem` deixa de ter `x_hw`/`y_hw` idênticos aos da bateria.
4. `y_hw` mais abrupta (salto ferro–ar inteiro em vez de borrado) — alvo do FNO2d mais difícil.
5. `node_y` na interface = lado de maior prioridade, inclusive picos de canto agudo (~2,9 T
   nas quinas dos dentes) — pesam mais em MSE/`mae_graph`.
6. Valor do lado do ar nos nós de interface só existe na grade; erro do FNO@nós nesses nós
   sobe (bilinear mistura os lados) — correção maior para a GNN.
7. Nós do corte periódico 0°/120°: smoothing do FEMM só vê meio leque → par de wrap com
   valores um pouco diferentes (já era assim; manter fiel ao FEMM).
8. Custo: ~+0,7 s/amostra (~+5–7 min nas 4000 com 8–16 processos).

**Decisões complementares (2026-10-09)**: prioridade também no `node_x` do FNO_GNN/
GNN_PostBase (mu_r e `delta_mu` das arestas); ordem ferro > ímã > cobre > ar; `M` do nó
segue o material escolhido (nó ímã → ±1 pela polaridade dos elementos de ímã incidentes,
qualquer outro → 0 — antes qualquer nó que tocasse ímã tinha M≠0); pasta
`mesh_ans_138x276_smooth/<arch>`; raw = o MESMO da bateria sem smooth (`mesh_ans_138x276`).

**Implementado (2026-10-09)**:
- `ans_parsing.py`: `MATERIAL_PRIORITY` + `_node_material_priority` (novas; nada existente mudou).
- `src/data_gen/parsers/femm_mesh_smooth.py::parse_ans_gzip_sample_smooth` — mesma
  saída `{arch: dict}` de `parse_ans_gzip_sample_unified`; muda só `node_y`, `y_hw`,
  `node_x[:,0:2]` (v1) e `edge_attr[:,3]` (v1). ~2 s/amostra (unified ~1 s).
- `scripts/build_smooth_ans_chunks_direct.py` — raw → chunks direto (reaproveita
  `_builders`/`_bufs_*` de `build_unified_ans_chunks_direct.py`), retomável, escrita
  atômica. `python -m scripts.build_smooth_ans_chunks_direct`. ~17,5 s/chunk (6 workers)
  → ~36 min pros 125 chunks.
- Validado (amostras 0 e 943): `node_y` = referência independente por laço
  (`priority_example.py`) a 1,2e-7 (float32); `y_hw` = `point_b` direto a 6e-8; 0 pixels
  fora da malha; demais campos bit a bit iguais aos unificados; sem NaN. Efeito: 8,5% dos
  nós mudam mu_r, 1,5% mudam M; |Δ| vs unificado mediana ~0,002 T, máx ~0,9 T. Smoke
  (2 chunks, pasta temporária): FNO2d/FNO_GNN/FNO_BipartiteGNN — 1 batch com gradiente
  finito + metric_fn ok; FNO_GNN com mesma semente: loss inicial 4,25 (smooth) × 3,58
  (unified).
- `scripts/run_best_configs_smooth.py` — bateria smooth: importa `run_best_configs.py`
  (mesmas BEST_CONFIGS/B1–B6/N_EPOCHS/critério de parada/pareamento GNN_PostBase) e troca
  só dataset → `mesh_ans_138x276_smooth/<arch>` e problem →
  `mesh_ans_138x276_smooth_battery` (seta `rbc.PROBLEM` pra base do GNN_PostBase
  ser o FNO2d desta bateria). Precheck: 125 chunks por arch, senão para antes de treinar.
  `TEST_SPLIT = 0.30` (`NnCfg.test_split`): teste = 37 chunks (subconjunto dos 88 de teste
  da oficial), treino = os mesmos 37 da oficial (verificado com `split_chunk_paths`,
  seed 12), 51 chunks fora — reservados p/ avaliação final independente (parada/best.pth
  escolhidos no teste de 37). Comparar com a bateria oficial: reavaliar os modelos dela
  nos mesmos 37 (ou nos 51).
  `python -m scripts.run_best_configs_smooth`.
- **Falta**: gerar as 4000 e rodar a bateria (GNN_PostBase estoura a RTX 4080 16 GB — VM).
