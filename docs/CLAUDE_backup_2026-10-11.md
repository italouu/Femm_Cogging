# Instruções para o Claude Code

## Regras obrigatórias
- Sempre, antes de propor código explique a proposta e resuma o que pretende fazer
- NUNCA mova, renomeie ou delete um arquivo sem confirmar comigo antes
- NUNCA delete código — se precisar remover algo, comente com `# [REMOVIDO]` e explique o motivo
- Trabalharemos por métodos
    | - Comentar código antigo
    | - Inserir código novo
    | - Realizar testes
    | - Confirmar atualização e remover código antigo

## Contexto do projeto
Predição de campos eletromagnéticos (B-field) em motores BLDC usando Neural Operators (FNO).
Pipeline: simulação FEM → dados em grade/quadtree → treino de FNO2d ou FNO_GNN.
Dependências principais: `pyfemm`, `shapely`, `torch`, `numpy`, `pandas`, `matplotlib`. Windows-only (pyfemm + pywin32).

### Convenções
- Todos os imports usam prefixo `src.` (ex: `from src.data_gen.motor_model import BLDC_Process`)
- Scripts executados a partir da raiz do projeto
- `motor_model.py` resolve DATA_DIR via `Path(__file__).parents[2] / "data" / "raw"`

### Convenção de erro dos resultados (2026-10-10)

Salvo indicação em contrário, "erro" num resultado significa sempre os **4 números**:
- **L1 e L2 ponto a ponto** — sem peso, média sobre os pontos;
- **L1 e L2 por superfície** — integral de superfície do erro, comparada com a superfície do
  FEMM (a solução de referência), **independente de o modelo sair em grade ou em malha**.

Implementação atual (`Accum` em `scripts/eval_surface_integral_table.py`, usada pelos
`pos_bateria_*`): e_i = | |B|_pred − |B|_FEMM | (erro do MÓDULO); peso a_i = área dual do nó
(malha) / área do elemento / r·dr·dθ (grade); `L1_A` = Σe·a/Σa, `L2_A` = √(Σe²·a/Σa),
`MAE_pt` = média simples de e; acumulado sobre todos os pontos de todas as amostras;
% = relativo a B_ref = √(Σ|B|_FEMM²·a/Σa). Lacuna: o **L2 ponto a ponto** (√(média de e²))
ainda não é calculado pela `Accum` — só o L1 ponto a ponto (`MAE_pt`).

**Grade × malha**: quando a saída do modelo e o FEMM estão em suportes diferentes (uma
superfície é a grade H×W, a outra os triângulos da malha do FEMM), o erro por superfície é
calculado sobre as **interseções** célula da grade ∩ triângulo: cada pedaço de interseção
tem o valor da célula de um lado e o valor do triângulo do outro, e o erro é integrado
pedaço a pedaço (sobreposição/refinamento comum das duas superfícies). Nada de amostrar um
suporte no outro (centro do pixel, interpolação nos nós) para essa métrica.
**Pendente**: não implementado — a métrica "grade" atual dos `pos_bateria_*` compara com o
`y_hw` (FEMM amostrado no centro do pixel) e integra na própria grade; a saída em grade
levada aos nós (FNO2d) usa interpolação. **Decidido (usuário, 2026-10-10), implementar
depois**: para simplificar, a célula da grade é tratada como elemento RETANGULAR/de lados
retos (quadrilátero dos 4 cantos ligados por segmentos em x,y), não como setor polar com
lados em arco — a interseção vira polígono ∩ triângulo (clipping de polígonos convexos).

### Datasets e bateria OFICIAIS (2026-10-03)

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

## Estrutura de arquivos

```
src/configs/      __init__.py  (re-exporta DatagenConfig, NnCfg, FNOConfig, MaskedFNO2dConfig,
                                FNO_GNNConfig, FNO_GNN_v2Config, MaskedFNO_GNNConfig, FNORefConfig,
                                SingleMatFNOConfig, GNN_PostBaseConfig, MonitorCfg, EvalCfg)
                  datagen.py   (DatagenConfig — geometria, simulação, qtree, chunks)
                  training.py  (NnCfg + arch configs; NnCfg valida arch↔arch_cfg em __post_init__)
                  monitor.py   (MonitorCfg)
                  eval.py      (EvalCfg)
src/training/     model_manager.py (ModelManager — run_id, diretórios, config.json, metrics.jsonl, status.txt)
src/data_gen/     motor_model.py (BLDC_Process, BLDC_FEMM_Model, BLDC_Shapely_Model,
                                  BLDC_FEMM_Model_Sym120, BLDC_FEMM_Model_Sym120_Annular
                                  — setor 120°+truncamento radial, promovidas de tests/,
                                  2026-07-23. BLDC_Process.MATERIAL_ID/PERMEABILITY/
                                  MAGNETIZATION e BLDC_FEMM_Model_Sym120.N_POLES_SECTOR
                                  reimportados de motor_constants.py desde 2026-08-13
                                  — mesmos valores, só não duplicados mais)
                  motor_constants.py (MATERIAL_ID, PERMEABILITY, MAGNETIZATION,
                                  N_POLES_SECTOR — literais puros extraídos de
                                  motor_model.py, ZERO dependência de projeto/femm/
                                  shapely; motor_model.py e
                                  parsers/ans_parsing.py/parsers/femm_mesh_v2.py
                                  reimportam daqui; ver "Serviço de parsing sem
                                  `import femm`" abaixo; 2026-08-13)
                  data_utils.py  (QtreeSampleUnifier, GridSampleUnifier, _read_grid_hw,
                                  build_node_x_motor [lazy], build_node_y_motor,
                                  build_graph_edges_motor [lazy], derive_material_id,
                                  generate_one_batch, check_data, combine_grids)
                  femm_mesh.py   (generate_mesh_sample — geração via malha real do FEMM
                                  como grafo, sem quadtree; grava x_hw/y_hw/a_hw já com
                                  nome final desde 2026-07-27; ver seção "Simetria 120°..."
                                  abaixo; 2026-07-23. save_ans_gzip_sample — mode='femm_mesh_v2',
                                  só desenha+malha+mi_analyze() e salva o .ans bruto em gzip,
                                  sem derivar nada; ver "Raw — malha real do FEMM v2" abaixo;
                                  2026-08-07. Funções puras de parsing (_parse_solution,
                                  _parse_block_materials, _block_magnet_polarity, _build_edges,
                                  _element_areas, _node_material_stats, _node_magnet_polarity,
                                  _wrap_edge_pairs, _build_bidirectional_edge_attrs,
                                  _grid_polar_xy, _assign_mesh_to_grid) movidas pra
                                  parsers/ans_parsing.py e reimportadas aqui — este arquivo
                                  mantém só `_query_grid_pointvalues`/`generate_mesh_sample`/
                                  `save_ans_gzip_sample` [COM, `import femm`]; ver "Serviço de
                                  parsing sem `import femm`" abaixo; 2026-08-13)
                  parsers/       sistema de parsers por arquitetura + serviço de parsing
                                 do raw .ans (sem `import femm`, ver seção própria abaixo)
                      __init__.py       (PARSER_REGISTRY dict {nome→parser})
                      _base.py          (MotorQtreeParserConfig, apply_parser_config)
                      ans_parsing.py    (funções puras de parsing/geometria do .ans —
                                         extraídas de femm_mesh.py, ZERO `import femm`;
                                         reaproveitadas por femm_mesh.py [v1, COM] e
                                         femm_mesh_v2.py [v2, sem COM]; não é
                                         MotorQtreeParserConfig, não entra no
                                         PARSER_REGISTRY; ver "Serviço de parsing sem
                                         `import femm`" abaixo; 2026-08-13)
                      femm_mesh_v2.py   (parse_ans_gzip_sample — deriva grafo de vértices +
                                         grafo de elementos [dual, não iterado] + arestas
                                         cruzadas + grade H×W a partir do .ans.gz bruto, SEM
                                         FEMM aberto/COM; relocado de
                                         src/data_gen/femm_mesh_v2.py em 2026-08-13 [mesmo
                                         motivo — não é MotorQtreeParserConfig/
                                         PARSER_REGISTRY, mas é "o serviço de parsing" do
                                         pipeline v2]; ver "Grafo duplo vértices+elementos"
                                         abaixo; 2026-08-10)
                      fno_gnn.py        (FNO_GNN_PARSER)
                      fno_gnn_v2.py     (FNO_GNN_V2_PARSER — FNO_GNN_PARSER + delta_mu
                                         direcional em edge_attr; 2026-07-17)
                      fno2d.py          (FNO2D_PARSER)
                      masked_fno2d.py   (MASKED_FNO2D_PARSER)
                      masked_fno_gnn.py (MASKED_FNO_GNN_PARSER)
                      femm_mesh.py      (FEMM_MESH_PARSER — layout mode='femm_mesh',
                                         node_x [S,5]/edge_attr [E,4]; 2026-07-27)
                      femm_mesh_a.py    (FEMM_MESH_A_PARSER — mesma entrada do
                                         FEMM_MESH_PARSER, alvo A [S,1]/[1,H,W] em vez
                                         de B via target_field='A'; 2026-08-05)
                      ans_b_smoothing.py (porte numpy do B suavizado do FEMM —
                                         GetElementB/GetNodalB/GetPointB, só .ans, sem
                                         femm; ver "Suavização de B do FEMM a partir do
                                         .ans"; 2026-10-07)
                      femm_mesh_smooth.py (parse_ans_gzip_sample_smooth — mesmos
                                         4 layouts do unified, B suavizado do FEMM +
                                         material do nó por prioridade; 2026-10-09)
                      femm_mesh_unified.py (parse_ans_gzip_sample_unified — 1 .ans.gz
                                         → layouts das 4 archs [FNO2d/FNO_GNN/
                                         GNN_PostBase/FNO_BipartiteGNN] com o MESMO
                                         gabarito B; não entra no PARSER_REGISTRY;
                                         ver "Datasets unificados a partir do raw
                                         mesh_ans_138x276" abaixo; 2026-10-01)
src/neural_op/    training_utils.py
                  losses.py    (LOSS_REGISTRY — mse, mae, relative_l2, masked_fno_loss,
                               masked_fno_gnn_loss; helper interno: masked_gnn_node_loss)
                  normalization.py (Normalizer — z-score por canal de x_hw/y_hw/node_x/
                               node_y, aplicado no CUDAPrefetcher [treino] e manualmente
                               em eval.py [chunk bruto, sem loader]; ver seção
                               "Normalização" abaixo; 2026-08-06. elem_x incluído
                               [z-score completo, sem exclusão] e node_x do
                               FNO_BipartiteGNN excluído por inteiro do z-score
                               [_NODE_X_FULLY_STRUCTURAL_ARCHS — layout [r_base,c_base],
                               diferente dos índices fixos {0,2,3,4} do grafo qtree/
                               femm_mesh v1]; 2026-08-10)
                  monitor.py   (TrainingMonitor)
                  archs/
                      _blocks.py        (NodeMLP, MLP, SpectralConv, FNO_Blocks, EdgeConvLayer, GNN;
                                         EdgeMessageBlock, BipartiteEdgeConvLayer, BipartiteGNN —
                                         grafo duplo vértices+elementos, ver "Grafo duplo
                                         vértices+elementos" abaixo; 2026-08-10)
                      fno.py            (FNO2d, fno_step_fn)
                      fno_gnn.py        (FNO_GNN, make_fno_gnn_step; grid_in_ch/grid_out_ch/
                                         node_in_ch auto-detectados do dataset — não mais
                                         hardcoded, ver NnCfg.__post_init__; 2026-08-05)
                      fno_gnn_v2.py     (FNO_GNN_v2 — subclasse vazia de FNO_GNN, hoje
                                         idêntica; requer parser FNO_GNN_V2_PARSER; mantida
                                         só por compatibilidade com runs já treinadas)
                      fno_mat.py        (MaskedFNO2d, masked_fno_step_fn, _make_material_masks)
                      masked_fno_gnn.py (MaskedFNO_GNN, make_masked_fno_gnn_step, _material_ids_from_mu)
                      gnn_post_base.py  (GNN_PostBase, gnn_post_base_step_fn — treino em 2 etapas,
                                         não end-to-end, sobre FNO2d/FNO_GNN congelado;
                                         node_in_ch/base_out_ch auto-detectados — 2026-08-05)
                      femm_mesh_v2_gnn.py (FNO_BipartiteGNN, make_fno_bipartite_gnn_step,
                                         fno_bipartite_gnn_metric_fn — arch pro grafo duplo
                                         de femm_mesh_v2.py; _interpolate_fno_to_nodes_v2
                                         [r_base/c_base em node_x[:,0]/[:,1], não [:,3]/[:,4]
                                         como no grafo qtree/femm_mesh v1]; ver "Grafo duplo
                                         vértices+elementos" abaixo; femm_mesh_v2_eval_fn
                                         reimportado de eval.py desde 2026-08-13 [antes era
                                         stub NotImplementedError]; 2026-08-10)
                      phi_deeponet.py   [REMOVIDO] — código comentado para referência
                      eval.py           (fno_eval_fn, fno_gnn_eval_fn, masked_fno_eval_fn,
                                         masked_fno_gnn_eval_fn; fno_eval_fn/_plot_fno_gnn_mesh
                                         suportam alvo escalar [A] além de vetorial [Bx,By] —
                                         2026-08-05. femm_mesh_v2_eval_fn/_plot_femm_mesh_v2
                                         [grafo duplo, alvo A, scatter em elementos+vértices] +
                                         _plot_femm_mesh_v2_b/_triangle_xy_from_edges/
                                         _element_curl_A/_element_radial_tangential [B=curl(A)
                                         pós-processado via EvalCfg.show_b, ver "Grafo duplo
                                         vértices+elementos" abaixo]; 2026-08-13)
                      __init__.py       (ArchEntry, ARCH_REGISTRY)
                  dataloaders/grid_loader.py (ChunkStreamDataset, ChunkQtreeDataset,
                                              CUDAPrefetcher, qtree_collate, build_loaders;
                                              ChunkFemmMeshV2Dataset, femm_mesh_v2_collate —
                                              grafo duplo, 4 espaços de índice [nó/elemento/
                                              aresta/aresta-cruzada], mode='femm_mesh_v2';
                                              2026-08-10)
src/utils/        visualization.py
src/bench/        datasets/__init__.py (DATASET_REGISTRY, DatasetEntry)
                  datasets/darcy.py
                  metrics.py     [a implementar]
                  eval_manager.py [a implementar]
scripts/          generate_data.py    (dispatcha para generate_data_femm_mesh.py se
                                       mode='femm_mesh', generate_data_femm_mesh_v2.py se
                                       mode='femm_mesh_v2'; senão pipeline CSV — todos gravam
                                       raw em data/raw/<dataset>/)
                  prepare_qtree_data.py
                  gen_npz_structures.py (filtra colunas via PARSER_REGISTRY[npz_parser];
                                       mode='grid'/'qtree' -> a partir das CSVs de
                                       data/raw/<dataset>/; mode='femm_mesh' -> branch
                                       próprio [_run_femm_mesh], a partir do staging bruto
                                       de data/raw/<dataset>/ [sample_*.npz] — sem
                                       ProcessPoolExecutor, só filtro de colunas; ambos
                                       gravam em data/temp/samples_npz/, femm_mesh com
                                       subpasta <dataset>_<npz_parser> [sufixo mantido,
                                       grid/qtree não sufixa]; unificado 2026-07-27.
                                       mode='femm_mesh_v2' -> branch _run_femm_mesh_v2,
                                       chama parse_ans_gzip_sample sobre cada .ans.gz de
                                       data/raw/<dataset>/ + r_in/r_ext de valid_designs.csv
                                       — SEM PARSER_REGISTRY [não há seleção de colunas,
                                       o parser já produz o formato final]; grava em
                                       data/temp/samples_npz/<dataset>/ sem sufixo de
                                       parser; ~1,2-5,7s/amostra; 2026-08-10. Paralelizado
                                       2026-08-13 [ProcessPoolExecutor persistente, sem
                                       max_tasks_per_child — parse_ans_gzip_sample é numpy
                                       puro, sem FEMM/COM aberto, então não tem o vazamento
                                       de memória que exige reciclar workers no resto do
                                       projeto; reaproveita o knob DatagenConfig.npz_max_workers
                                       já usado pelo pipeline grid/qtree]; validado byte-a-byte
                                       idêntico ao resultado sequencial anterior)
                  build_data_chunks.py (dispatcha para build_data_chunks_femm_mesh.py se
                                       mode='femm_mesh', build_data_chunks_femm_mesh_v2.py
                                       se mode='femm_mesh_v2'; senão concatena os .npz de
                                       gen_npz_structures.py em data_chunk_*.pt)
                  generate_data_femm_mesh.py     (geração via malha FEMM — .npz de staging
                                                  BRUTO por amostra em data/raw/<dataset>/ —
                                                  fonte única cara de gerar, não fica em
                                                  data/temp/; 2026-07-23, pasta corrigida
                                                  2026-07-27)
                  build_data_chunks_femm_mesh.py (chama gen_npz_structures.run() [branch
                                                  femm_mesh] + concatenação PURA em
                                                  data_chunk_*.pt, em data/torch/data_chunks/
                                                  <dataset>_<npz_parser>/; 2026-07-23,
                                                  unificado com gen_npz_structures.py
                                                  2026-07-27 — ver "Filosofia raw → npz →
                                                  chunks" abaixo)
                  apply_parser_femm_mesh.py      [REMOVIDO 2026-07-27 — lógica migrada para
                                                  o branch femm_mesh de gen_npz_structures.py;
                                                  arquivo mantido comentado por rastreabilidade]
                  generate_data_femm_mesh_v2.py  (mode='femm_mesh_v2' — só geração bruta: salva
                                                  o .ans INTEIRO comprimido em gzip, direto em
                                                  data/raw/<dataset>/sample_XXXXXX.ans.gz, sem
                                                  derivar node_x/edge_index/etc. — ver seção
                                                  "Raw — malha real do FEMM v2" abaixo; 2026-08-07)
                  build_data_chunks_femm_mesh_v2.py (chama gen_npz_structures.run() [branch
                                                  femm_mesh_v2] + concatenação cuidando de
                                                  4 espaços de índice [nó/elemento/aresta/
                                                  aresta-cruzada] em data_chunk_*.pt, em
                                                  data/torch/data_chunks/<dataset>/ [sem
                                                  sufixo — não há parser a distinguir]; ver
                                                  "Grafo duplo vértices+elementos" abaixo;
                                                  2026-08-10)
                  train.py       (script de treino — configurado por NnCfg; lógica de treino
                                  isolada em run(_nn) desde 2026-09-14, pra ser reaproveitada
                                  por run_best_configs.py sem duplicar o pipeline model/
                                  optimizer/scheduler/fit/ModelManager — `if __name__=='__main__'`
                                  só monta o NnCfg do run manual e chama run())
                  run_best_configs.py (sequência de treinos: reconstrói os hiperparâmetros do
                                  melhor run já encontrado por arquitetura — FNO2d, FNO_GNN,
                                  GNN_PostBase, FNO_BipartiteGNN, ver CLAUDE.md/conversa sobre
                                  levantamento de runs em datasets mesh_* — e roda cada um com
                                  loss='mse' e loss='mae' [8 treinos sequenciais]; cobre a
                                  lacuna de FNO_BipartiteGNN nunca ter sido comparado contra as
                                  losses padrão [até então só graph_div_b_loss no dataset
                                  mesh_ans_138x276_B]; v3 e graph_div_b_loss ficam de fora por
                                  decisão do usuário; 2026-09-14. Adaptado 2026-10-01 pros
                                  datasets unificados: dataset=mesh_ans_138x276_unified/<arch>,
                                  problem único mesh_ans_138x276_unified_best_mse_mae;
                                  GNN_PostBase usa como base o FNO2d treinado NA PRÓPRIA
                                  bateria com a MESMA loss [GNN_PostBase mse -> FNO2d mse,
                                  mae -> FNO2d mae; decisão do usuário], localizado em
                                  runtime [_find_base_run_dir(loss)] — arch_cfg vira fábrica
                                  (recebe a loss), construída só na hora do treino; datasets
                                  antigos comentados # [REMOVIDO]. Atenção: base antiga era
                                  outro run [mesh_138x276_FEMM_MESH/FNO2d/run_0001, width 6,
                                  lr 0,001, 5,3M params], a nova é o FNO2d "best" [width 8,
                                  lr 0,005, 9,4M params]. GNN_PostBase [width 64 × 6 camadas,
                                  batch 32] estoura a RTX 4080 16 GB — pico 16,2 GiB mesmo
                                  com batch 16; ver "Pendências conhecidas")
                  build_unified_ans_datasets.py (raw mesh_ans_138x276 → npz → chunks dos
                                  4 datasets unificados [uma pasta por arch], mesmo gabarito
                                  B pra todas; ver "Datasets unificados a partir do raw
                                  mesh_ans_138x276" abaixo; 2026-10-01)
                  build_unified_ans_chunks_direct.py (mesma saída de
                                  build_unified_ans_datasets.py, mas raw → chunks DIRETO,
                                  sem staging .npz [~105 GB a menos]; escrita atômica via
                                  <arch>/.tmp/, retomável por chunk; 2026-10-01)
                  build_smooth_ans_chunks_direct.py (raw mesh_ans_138x276 → chunks
                                  mesh_ans_138x276_smooth/<arch>, B suavizado do FEMM;
                                  mesma mecânica do _direct unificado; 2026-10-09)
                  run_best_configs_smooth.py (bateria smooth — reaproveita
                                  run_best_configs.py, troca só dataset/problem; 2026-10-09)
                  eval.py        (único script de avaliação — configurado por EvalCfg)
                  parsers/darcy.py  (HDF5 → data_chunks)
                  bench_eval.py  [a implementar]
tests/            gen_test.py, test_chunk_plot.py, test_graph_plot.py (manuais, abrem plot)
                  smoke_test_refactor.py (rodar da raiz: python -m tests.smoke_test_refactor;
                                         5/7 OK desde 2026-08-05 — CHUNK_PATHS aponta pra
                                         dataset/path obsoletos, ver "Pendências conhecidas")
                  open_femm_manual.py         (motor aleatório, FEMM aberto p/ inspeção manual — sem mesh/analyze)
                  simulate_and_plot_A.py      (motor aleatório, mesh+analyze, plota A numa grade polar)
                  proto_femm_mesh_graph.py    (benchmark: extrai nós/elementos da malha real do
                                              FEMM via mo_getnode/mo_getelement em loop — mede custo)
                  proto_femm_sym120.py        (BLDC_FEMM_Model_Sym120 — protótipo original;
                                              promovida para motor_model.py, ver seção "Simetria 120°" abaixo)
                  proto_femm_sym120_annular.py (BLDC_FEMM_Model_Sym120_Annular — idem, promovida)
                  open_femm_manual_sym120.py  (abre o modelo Sym120_Annular no FEMM p/ inspeção manual)
                  proto_export_fem_mesh.py    (benchmark: export via malha real do FEMM vs qtree —
                                              tempo total e tamanho de arquivo)
                  proto_femm_mesh_gendata.py  (protótipo original de generate_mesh_sample —
                                              promovido para src/data_gen/femm_mesh.py, com
                                              correções — ver seção "Simetria 120°..." abaixo)
                  proto_femm_mesh_plot.py     (plota/valida campos da malha real vs point-query;
                                              atribuição malha→grade H×W via trifinder)
                  proto_femm_element_b_check.py (Verificação 1/2: B=curl(A) por elemento via
                                              fórmula fechada P1 bate com mo_getb(smooth='off')
                                              a ~1e-16 [em metros]; B suavizado do nó ≈ média
                                              simples dos elementos incidentes; 2026-08-07)
                  proto_femm_mesh_v2_element_b_plot.py (plota Bx/By/|B| calculado via curl(A)
                                              a partir do .ans.gz bruto, por elemento
                                              [tripcolor shading='flat'] e amostrado na
                                              grade H×W [trifinder, mesma técnica de
                                              _assign_mesh_to_grid] — sem FEMM aberto;
                                              2026-08-10)
                  proto_femm_mesh_v2_parser_check.py (sanidade de
                                              parsers/femm_mesh_v2.py::parse_ans_gzip_sample
                                              num sample real — shapes, sem NaN, C==3*M,
                                              mu_r com os 4 valores canônicos; 2026-08-10.
                                              import atualizado 2026-08-13 pro novo caminho)
                  proto_femm_mesh_v2_arch_smoke.py (smoke test ponta a ponta de
                                              FNO_BipartiteGNN sobre chunks reais —
                                              NnCfg→build_loaders→model→step_fn→backward,
                                              confirma gradiente finito em F1 e F2
                                              separadamente; 2026-08-10)
data/raw/         CSVs gerados — fonte única; parsers por arquitetura lêem daqui
data/torch/       tensores .pt (data_chunks/)
data/external/    dados brutos de terceiros (gitignore conteúdo)
data/temp/        temp FEMM (gitignore)
data/models/      pesos salvos .pth (gitignore)
docs/             historico_implementacao.txt, CLAUDE_backup_2026-05-28.txt
```

---

## Contratos de interface

### ARCH_REGISTRY — `src/neural_op/archs/__init__.py`

Registro central de arquiteturas. Cada entrada é um `ArchEntry` (dataclass):

| Campo | Tipo | Descrição |
|---|---|---|
| `cls` | `type` | classe do modelo |
| `cfg_cls` | `type` | dataclass de config correspondente |
| `loader_mode` | `str` | `'grid'`, `'qtree'` ou `'femm_mesh_v2'` — passado direto ao `build_loaders` |
| `model_kwargs` | `callable` | `(arch_cfg) -> dict` filtrando campos não-construtores (ex: `lambda_loss`) |
| `make_step_fn` | `callable` | `(arch_cfg) -> step_fn(batch, model, loss_fn, device)` |
| `eval_fn` | `callable` | `(model, chunk_data, eval_cfg) -> None` |
| `metric_fn` | `callable` | `(arch_cfg) -> metric_fn(batch, model, device) -> (mae_hw, mae_graph\|None)` — MAE bruto para `metrics.jsonl` (ver seção `metrics.jsonl` abaixo); `2026-07-19` |

`ArchEntry.make_model(arch_cfg)` é atalho para `cls(**model_kwargs(arch_cfg))`.

Arquiteturas registradas: `FNO2d` (grid), `FNO_GNN` (qtree), `FNO_GNN_v2` (qtree — subclasse de
`FNO_GNN` com `edge_dim=5`, delta_mu direcional; requer parser `FNO_GNN_V2_PARSER`), `MaskedFNO2d`
(grid), `MaskedFNO_GNN` (qtree), `GNN_PostBase` (qtree), `FNO_BipartiteGNN` (`loader_mode`
`'femm_mesh_v2'` — grafo duplo vértices+elementos, ver "Grafo duplo vértices+elementos"
abaixo), `FNORef` (grid, condicional via `try/except ImportError`).

```python
# Adicionar nova arquitetura:
ARCH_REGISTRY['MinhaArch'] = ArchEntry(
    cls=MinhaArch, cfg_cls=MinhaConfig, loader_mode='grid',
    model_kwargs=lambda cfg: asdict(cfg),
    make_step_fn=lambda cfg: minha_step_fn,
    eval_fn=minha_eval_fn,
)
```

#### `GNN_PostBase` — treino em duas etapas (não end-to-end)

GNN treinado isoladamente sobre a saída de um `FNO2d` ou `FNO_GNN` **já treinado e congelado**
(sem gradiente fluindo pelo modelo base — só o GNN novo treina). Usa os mesmos chunks qtree
de `FNO_GNN`; reaproveita `fno_gnn_eval_fn` para plot (mesma assinatura de `forward`).

`GNN_PostBaseConfig` (`training.py`):
- `base_run_dir`/`base_checkpoint` (`'best'`/`'latest'`/`'final'`) — run já treinada a usar como base.
- `gnn_node_width`/`gnn_n_layers` — hiperparâmetros do GNN novo.
- `base_arch`/`base_arch_cfg`/`base_epoch` — **snapshot somente-leitura** (`init=False`), preenchido em
  `__post_init__` lendo `base_run_dir/config.json` + checkpoint. Grava os hiperparâmetros do modelo
  base no `config.json` desta run nova, para não depender de `base_run_dir` continuar existindo.
- `node_in_ch` (`init=False`) — features de nó do GNN treinável; auto-detectado em
  `NnCfg.__post_init__` a partir do **dataset desta run nova** (não de `base_run_dir`).
- `base_out_ch` (`init=False`) — canais de saída do modelo base congelado (2=Bx,By; 1=A);
  lido do `config.json` do **próprio `base_run_dir`** dentro de `GNN_PostBaseConfig.__post_init__`
  (`out_channels` se base for `FNO2d`, `grid_out_ch` se `FNO_GNN`/variantes) — não recalculado a
  partir do dataset novo, é propriedade fixa da arquitetura já treinada. 2026-08-05.
- `base_normalize`/`base_norm_stats` (`init=False`) — snapshot da normalização usada no treino
  do modelo base (idem `base_out_ch`, lido do `config.json` do próprio `base_run_dir`) — ver
  seção "Normalização" abaixo. 2026-08-06.
- Fallback: se `base_run_dir` for movido/apagado depois, `GNN_PostBase._load_frozen_base` reconstrói a
  arquitetura a partir do snapshot; os pesos congelados corretos vêm do checkpoint da própria run de
  `GNN_PostBase` (que já embute o `state_dict` completo do modelo base, mesmo congelado).

### PARSER_REGISTRY — `src/data_gen/parsers/__init__.py`

`PARSER_REGISTRY = {'FNO_GNN': …, 'FNO_GNN_v2': …, 'FNO2D': …, 'MaskedFNO2d': …, 'MaskedFNO_GNN': …, 'FEMM_MESH': …, 'FEMM_MESH_A': …, 'FEMM_MESH_V2': …, 'FEMM_MESH_V2_A': …}`

Cada parser é uma instância de `MotorQtreeParserConfig` com campos:
`name`, `node_x_cols`, `node_y_cols`, `x_hw_cols`, `y_hw_cols`, `edge_attr_cols`, `build_graph`,
`target_field` (`'B'` padrão | `'A'`, ver abaixo).

| Parser | build_graph | node_x_cols | edge_attr_cols | target_field |
|--------|-------------|-------------|----------------|--------------|
| `FNO_GNN_PARSER` | True | [1,2,3,4,5] | [0,1,2,3] | B |
| `FNO_GNN_V2_PARSER` | True | [1,2,3,4,5] | [0,1,2,3,**4**] | B |
| `FNO2D_PARSER` | False | [1,2] | [] | B |
| `MASKED_FNO2D_PARSER` | False | [1,2] | [] | B |
| `MASKED_FNO_GNN_PARSER` | True | [1,2,3,4,5] | [0,1,2,3] | B |
| `FEMM_MESH_PARSER` | True | [1,2,3,4,5] | [0,1,2,3] | B |
| `FEMM_MESH_A_PARSER` | True | [1,2,3,4,5] | [0,1,2,3] | **A** |
| `FEMM_MESH_V2_PARSER` | True | [] (ignorado) | [] (ignorado) | B |
| `FEMM_MESH_V2_A_PARSER` | True | [] (ignorado) | [] (ignorado) | **A** |

**`FEMM_MESH_V2_PARSER`/`FEMM_MESH_V2_A_PARSER`** (`femm_mesh_v2_b.py`/`femm_mesh_v2_a.py`,
2026-08-19) — únicos parsers de `mode='femm_mesh_v2'` (arch `FNO_BipartiteGNN`). Reaproveitam
`MotorQtreeParserConfig` só pelo campo `target_field`; `node_x_cols`/`node_y_cols`/`x_hw_cols`/
`y_hw_cols`/`edge_attr_cols` ficam vazios e são ignorados porque
`src/data_gen/parsers/femm_mesh_v2.py::parse_ans_gzip_sample` já produz o formato final (grafo
de vértices + grafo de elementos + arestas cruzadas + grade H×W) — não há seleção de coluna a
fazer, diferente do resto do `PARSER_REGISTRY`. `scripts/gen_npz_structures.py::_run_femm_mesh_v2`
lê só `PARSER_REGISTRY[npz_parser].target_field` e repassa pra `parse_ans_gzip_sample`.
Substitui o campo dedicado `DatagenConfig.femm_mesh_v2_target_field` (descontinuado — `# [REMOVIDO]`
em `src/configs/datagen.py`), que fazia `mode='femm_mesh_v2'` ser o único modo fora do contrato
`npz_parser`→`PARSER_REGISTRY` usado pelo resto do projeto. `DatagenConfig.npz_parser` default
mudou de `'FEMM_MESH_A'` (chave de `mode='femm_mesh'`, vestigial — nunca era lido em
`mode='femm_mesh_v2'`) para `'FEMM_MESH_V2'`, preservando o comportamento default anterior
(`target_field='B'`). `DatagenConfig.femm_mesh_v2_dataset_name` agora importa `PARSER_REGISTRY`
localmente (dentro da property, mesmo padrão de `NnCfg.__post_init__` importar `ARCH_REGISTRY`
dentro do método) em vez de um campo solto, pra não puxar `matplotlib.tri`/`scipy` pro topo do
módulo de config.

**`target_field`** (`MotorQtreeParserConfig`, `_base.py`, 2026-08-05): `'B'` (padrão) faz
`node_y`/`y_hw` virem de `node_y`/`y_hw` brutos (Bx,By), fatiados por `node_y_cols`/`y_hw_cols`
— comportamento de sempre, sem mudança. `'A'` faz `node_y`/`y_hw` serem reconstruídos a partir
de `node_A`/`a_hw` (potencial vetor, 1 canal escalar) — **só suportado no pipeline
`mode='femm_mesh'`** (`_apply_parser_femm_mesh` em `scripts/gen_npz_structures.py`); o caminho
qtree/grid (`apply_parser_config`, `_base.py`) tem um `assert target_field=='B'` porque não tem
`node_A` disponível. `node_y_cols`/`y_hw_cols` são ignorados quando `target_field='A'`.
`FEMM_MESH_A_PARSER` (`femm_mesh_a.py`) tem a mesma entrada (`node_x_cols`) do
`FEMM_MESH_PARSER` — só o alvo muda; B não é exportado por esse parser.

Chunks gerados por `FNO_GNN_PARSER` e `MASKED_FNO_GNN_PARSER` são intercambiáveis.
`FNO_GNN_V2_PARSER` (2026-07-17) difere apenas por incluir a coluna 4 (`delta_mu`,
direcional) em `edge_attr` — gera chunks `[E,5]`, **não** intercambiáveis com os
demais (`[E,4]`); usado exclusivamente pelo arch `FNO_GNN_v2` (`_EDGE_DIM=5`).
`FEMM_MESH_PARSER` (2026-07-27) opera sobre um layout de origem diferente (malha real do
FEMM, não quadtree — `node_x[:,3]` é `node_dual_area`, `edge_attr` só tem 4 colunas fixas,
sem `shared_length`) — **não intercambiável** com os demais parsers mesmo com índices
numericamente iguais; usado só por `mode='femm_mesh'`, ver "Chunks — malha real do FEMM"
abaixo. Aplicado pelo branch `mode='femm_mesh'` de `scripts/gen_npz_structures.py`
(unificado 2026-07-27 — antes era um script próprio, `apply_parser_femm_mesh.py`).

### Configs — `src/configs/`

**`DatagenConfig`** (`datagen.py`): parâmetros de geometria, simulação e geração de dados.
Chaves comportamentais: `mode` (`'grid'`/`'qtree'`/`'femm_mesh'`/`'femm_mesh_v2'`), `npz_parser`
(chave em `PARSER_REGISTRY` — usado por `gen_npz_structures.py` em `'grid'`/`'qtree'`/
`'femm_mesh'`; **ignorado** em `mode='femm_mesh_v2'`, que não usa `PARSER_REGISTRY` — ver
"Grafo duplo vértices+elementos" acima), `dataset` (define subdiretórios em `data/`),
`max_depth`, `homogeneity_threshold`, `femm_mesh_max_workers` (só `mode='femm_mesh'`).
`parsed_dataset_name` (property) = `f"{dataset}_{npz_parser}"` — nome de subdiretório usado só
pelo `mode='femm_mesh'` (`data/temp/samples_npz/` e `data/torch/data_chunks/` — mesma
pasta-base do pipeline grid/qtree, mas com sufixo, que o grid/qtree não usa), pra não
misturar chunks de parsers diferentes.
`n_phases` removido — fase do rotor fixada em 0 em `generate_one_batch`.

`npz_parser` default = `'FEMM_MESH'` (corrigido 2026-07-27 — antes era `'FNO_GNN'`, mismatch
com `mode` default `'femm_mesh'`).

`mode='femm_mesh'` (2026-07-23): `scripts/generate_data.py` e `scripts/build_data_chunks.py`
despacham automaticamente pra `generate_data_femm_mesh.py`/`build_data_chunks_femm_mesh.py`
quando esse modo está ativo (checam `MODE` no `__main__`/`main()` antes de seguir o fluxo
`check_data`/`generate_one_batch`, que só aceitam `'grid'`/`'qtree'`) — mesmos dois comandos
de sempre (`python -m scripts.generate_data`, `python -m scripts.build_data_chunks`)
funcionam pros três modos. Ver seção "Simetria 120°..." mais abaixo para o formato de dados.

**`generate_samples_constrained`** (`motor_model.py`) — constantes de geometria fixas:

| Parâmetro | Valor |
|-----------|-------|
| `stator_inner_diameter` | 60 mm |
| `rotor_outer_diameter` | 90 mm |
| `slot_Hs0` | 1 mm |
| `slot_Hs1` | 1.2 mm |
| `slot_Hs2` | 3.7 mm |
| `stack_length` | 1 mm (não afeta B-field 2D) |

Variáveis: `stator_outer ∈ [73.8, 81]` mm, `rotor_inner = stator_outer + 2·gap` (`gap ∈ [0.5, 2]` mm),
`pole_thickness`, `pole_embrance ∈ [0.6, 0.90]`, `Bs0/Bs1/Bs2` como frações do arco no raio correspondente.
Distribuição: controlada por `DISTRIBUTION` global (`'uniform'` ou `'triangular'`).

**`NnCfg`** (`training.py`): treino completamente independente do datagen.
- `arch`: chave em `ARCH_REGISTRY`; `arch_cfg`: auto-instanciado em `__post_init__` se `None`; tipo validado contra `arch` se passado explicitamente.
- `loss`: chave em `LOSS_REGISTRY`.
- `resume_run`/`resume_checkpoint`/`resume_modified`: controle de retomada de treino.
- `monitor_cfg`: instância de `MonitorCfg` (checkpoint a cada N épocas, early stop em heartbeats, save_best).
- `normalize`/`norm_stats`: normalização z-score de x_hw/y_hw/node_x/node_y — ver seção
  "Normalização" abaixo.

**Arch configs** — campos 1:1 com construtores dos modelos; `model_kwargs` filtra campos extras (`lambda_loss`, `data_res`). Instâncias: `FNOConfig`, `FNO_GNNConfig`, `FNO_GNN_v2Config`, `MaskedFNO2dConfig`, `MaskedFNO_GNNConfig`, `SingleMatFNOConfig`, `FNORefConfig`. Exceção: `GNN_PostBaseConfig` tem campos extras `init=False` (snapshot do modelo base — ver seção `GNN_PostBase` acima).

`FNO_GNN_v2Config` (`training.py`) — subclasse vazia de `FNO_GNNConfig` (mesmos campos); existe só
para que `NnCfg.__post_init__` valide `arch='FNO_GNN_v2'` ↔ tipo do `arch_cfg` corretamente e para
o `config.json` da run registrar qual variante foi treinada.

**Canais/dimensões auto-detectados do dataset** (`_detect_chunk_dims`, `training.py`, 2026-08-05):
generalização do mecanismo que já existia só para `edge_dim`. `NnCfg.__post_init__` lê o primeiro
chunk real de `self.dataset` uma única vez e injeta, via campos `init=False`:
- `FNOConfig.in_channels`/`out_channels` — de `x_hw`/`y_hw` (usado por `FNO2d`/`FNO_ref`).
- `FNO_GNNConfig.grid_in_ch`/`grid_out_ch`/`node_in_ch`/`edge_dim` — de `x_hw`/`y_hw`/`node_x`/
  `edge_attr` (usado por `FNO_GNN`/`FNO_GNN_v2`/`FNO_GNN_Field`); herdado por `FNO_GNN_v2Config`.
- `GNN_PostBaseConfig.node_in_ch`/`edge_dim` — mesma fonte (dataset desta run nova); `base_out_ch`
  é caso à parte, ver acima.

Motivação: `FEMM_MESH_A_PARSER` gera chunks com `y_hw`/`node_y` de 1 canal (A) em vez de 2 (Bx,By)
— sem auto-detecção, `FNO_GNN._GRID_OUT_CH=2`/`_NODE_IN_CH=5` hardcoded (e equivalentes em
`GNN_PostBase`) quebravam com `RuntimeError` de shape mismatch. `FNO2d`/`GNN`/`FNO_GNN`/
`GNN_PostBase` não têm mais nenhuma constante de canal hardcoded — só `FNOConfig.in_channels`
tem nome próprio (não `grid_in_ch`) porque é repassado direto ao construtor de `FNO2d` via
`asdict(cfg)`, que espera esse nome exato.

**Não coberto por essa mudança** (decisão do usuário, 2026-08-05): `MaskedFNO2dConfig`/
`MaskedFNO_GNNConfig` continuam com `out_channels=8`/`_NODE_IN_CH`/`_GNN_IN_CH` hardcoded —
8 = `4 materiais × 2 campos`, não é um número diretamente auto-detectável de `y_hw.shape[1]`
(precisaria de uma decisão de design própria: `n_materiais(4) × out_ch_detectado`?). Não quebra
hoje porque não existe combinação Masked+alvo-A, mas fica como limitação conhecida.

**Efeito colateral da auto-detecção**: os chunks obsoletos `data/torch/data_chunks/mesh_138x276`/
`mesh_138x276old` (node_x com as 9 colunas cheias, não as 5 já filtradas pelo parser — ver
"Pendências conhecidas") **antes** causavam erro de shape imediato e claro ao apontar `FNO_GNN`
pra eles (`_NODE_IN_CH=5` hardcoded vs `node_x` real de 9). Agora `node_in_ch` é detectado como 9
silenciosamente, sem erro — a rede de proteção "mismatch de shape = falha alto" que existia meio
que por acidente não existe mais. Aviso "não usar" desses datasets no CLAUDE.md continua sendo a
única proteção (decisão do usuário: suficiente por ora, sem checagem de código adicional).

### Normalização — `src/neural_op/normalization.py` (2026-08-06)

`Normalizer`: z-score por canal (mean/std) de `x_hw`/`y_hw`/`node_x`/`node_y`, mesmo
padrão do `UnitGaussianNormalizer` da referência do FNO — corrige alvos com escala
física pequena (ex: `A`/potencial vetor, `~1e-3`) onde MSE bruta cai perto de zero sem a
rede aprender a estrutura real. Stats calculadas 1x sobre **todos** os chunks de treino
(streaming, custo ≈ 1 época de I/O) e cacheadas em
`data/torch/data_chunks/<dataset>/norm_stats.json`; gravadas também no `config.json` da
run (`NnCfg.normalize`/`norm_stats`) pra o eval nunca recalcular.

**Exclusões** (mean=0/std=1, canal fica cru) — dependem do arch, não são regra fixa:
- `x_hw[:,0]` (mu_r) — só nos archs que fazem threshold físico sobre ele em runtime
  (`_make_material_masks`): `MaskedFNO2d`, `FNO2d_SingleMat`, `MaskedFNO_GNN`.
- `node_x[:, {0,2,3,4}]` (mu_r, cell_area, r_base, c_base) — sempre que há grafo, usados
  crus por `_interpolate_fno_to_nodes`/eval/bench.
- `y_hw`/`node_y` (o alvo) e `node_x[:,1]` (M) sempre normalizados; `edge_attr` fora do escopo.

**Integração**: `CUDAPrefetcher(normalizer=...)` normaliza o batch no treino, transparente
pro `step_fn`. `model.normalizer` (atributo simples) é atrelado ao modelo em
`scripts/train.py`/`scripts/eval.py`; as `metric_fn`/`eval_fn` de cada arch decodificam
pred/y de volta pra unidade física antes de calcular `mae_hw`/`mae_graph` ou plotar
(`eval.py` carrega o chunk bruto direto, sem `CUDAPrefetcher`, então também encoda a
entrada manualmente antes de chamar o modelo).

**`GNN_PostBase`**: a base congelada usa a normalização *dela própria*
(`GNN_PostBaseConfig.base_normalize`/`base_norm_stats`, snapshot de
`base_run_dir/config.json`) — `_base_pred_at_nodes` faz round-trip decode/encode só ao
redor da chamada à base, pra ela sempre receber a escala que aprendeu.

**`EvalCfg`** (`eval.py`): `run_dir`, `checkpoint` (`'best'`/`'latest'`/`'final'`), `chunk_name`, `sample_idx`. `arch` e `dataset` lidos de `run_dir/config.json`.
`error_plot_mode` (`2026-07-19`): `'percent'` (default, erro norm. por `B_ref`, %) ou `'absolute'`
(erro bruto `|pred-alvo|`, mesma unidade do campo — Tesla) — alterna o painel de erro nas 5
`*_eval_fn` de `src/neural_op/archs/eval.py` via helpers `_err_display`/`_err_vmax`. `error_cap`
só se aplica no modo `'percent'`; no modo `'absolute'` o vmax sempre auto-escala.

### LOSS_REGISTRY — `src/neural_op/losses.py`

| Chave | Assinatura | Usada por |
|-------|-----------|-----------|
| `'mse'`, `'mae'`, `'relative_l2'` | `(out, y)` | FNO2d, FNO_GNN, FNORef |
| `'masked_fno_loss'` | `(pred8, y, masks)` | MaskedFNO2d |
| `'masked_fno_gnn_loss'` | `(y_hw_8, y_hw, masks, y_nodes_8, node_y, material_ids, lambda_loss)` | MaskedFNO_GNN |

### ModelManager — `src/training/model_manager.py`

Escreve em `data/logs/{problem}/{arch}/run_XXXX/`:
- `config.json` — `asdict(NnCfg)` + n_params, device, start_time
- `metrics.jsonl` — `{epoch, train_loss, test_loss, lr, train_time_s, eval_time_s, samples_per_s}`
- `status.txt` — `running` → `done` / `stopped` / `failed`
- `checkpoints/best.pth`, `checkpoints/latest.pth`, `model_final.pth` — `{model_state_dict, optimizer_state_dict, epoch}`

`run_XXXX` é local ao par `{problem}/{arch}`. Sem registro global.
`status.txt = running` + ausência de `model_final.pth` → interrupção abrupta.

**`metrics.jsonl` — campos `mae_hw`/`mae_graph` (2026-07-19)**: além de
`epoch`/`train_loss`/`test_loss`/`lr`/`train_time_s`/`eval_time_s`/`samples_per_s`, cada
linha traz `mae_hw` e `mae_graph` — MAE bruto (sem máscara, sem normalização por `B_ref`),
calculado só no heartbeat (mesmo ritmo de `checkpoint_every`) sobre o **test set**, via
`ArchEntry.metric_fn` (`fit(..., metric_fn=entry.metric_fn(arch_cfg))` em `scripts/train.py`).
`mae_hw` sempre compara a saída em grade H×W do estágio FNO contra `y_hw`; `mae_graph`
compara a saída final em nós/grafo contra `node_y` — `None` quando o arch não produz saída
em grafo (`FNO2d`, `MaskedFNO2d`, `FNO2d_SingleMat`, `FNO_ref`). Implementado em
`compute_mae_metrics` (`src/neural_op/training_utils.py`).

### MonitorCfg — critério de parada (`src/configs/monitor.py` / `src/neural_op/monitor.py`, 2026-09-13)

`gl_threshold` (float, %, default `5.0`) — critério ativo por padrão. Implementa
*generalization loss* (Prechelt, 1998): `GL(t) = 100·(test_loss(t)/melhor_test_loss - 1)`;
para quando `GL(t) > gl_threshold`. Adimensional (não depende da escala da loss, ao
contrário de `early_stop_min_delta`) e mede diretamente a deterioração relativa da
representatividade do modelo em relação ao melhor ponto já visto — motivado por overfitting
lento observado em runs de `FNO_BipartiteGNN` sobre `mesh_ans_138x276_B` (`test_loss`/
`mae_graph` melhoram até certo heartbeat e depois pioram monotonicamente enquanto
`train_loss` continua caindo). `early_stop_patience`/`early_stop_min_delta` (contagem de
heartbeats sem melhora, valores de patience/delta escolhidos à mão) continuam disponíveis
como critério alternativo — os dois campos coexistem em `TrainingMonitor.step`, cada um só
age se não for `None`. Validado reproduzindo `run_0015` (`mesh_ans_138x276_B/
FNO_BipartiteGNN`) fora do loop de treino: `GL_5` para na época 389 (`test_loss=0,1007`,
melhor `0,0956` na época 109) — mesmo ponto calculado manualmente, sem falso positivo nos
heartbeats anteriores.

**GL sustentado + warm-up (2026-10-02)**: o GL acima parava no PRIMEIRO heartbeat acima do
limiar — na bateria `mesh_ans_138x276_unified_best_mse_mae` isso encerrou 7 de 8 runs cedo
demais: ruído de um heartbeat isolado no início (FNO2d mse na época 39, +6,8%; Bipartite mae
na 39, +5,2%; Bipartite mse na 69) ou pico de instabilidade da otimização (FNO_GNN/
GNN_PostBase — `train_loss` também explode, ex. 0,107→1,33, não é overfitting). Novos campos
de `MonitorCfg`: `gl_patience` (default 3 — GL precisa ficar acima de `gl_threshold` em N
heartbeats CONSECUTIVOS; volta abaixo zera a contagem; 1 = comportamento antigo) e
`min_epochs` (default 100 — nenhum critério de parada, GL nem patience, age antes; 0 =
desativado). Bloco antigo comentado `# [REMOVIDO]` em `TrainingMonitor.step`. Validado
reexecutando o `TrainingMonitor` real sobre os `metrics.jsonl`: com `gl_patience=1,
min_epochs=0` reproduz exatamente as 8 paradas registradas; com os defaults novos nenhuma das
7 paradas prematuras dispara, e o overfitting real de `run_0015` continua detectado (época 429
em vez de 389 — `best.pth` é o mesmo, salvo independentemente). Configs antigas carregam sem
mudança (campos novos têm default; `getattr` com fallback no monitor).

**Paciência por estagnação ligada por padrão (2026-10-03)**: `early_stop_patience` default
`None` → `10` (heartbeats sem novo best, i.e. 100 épocas; `early_stop_min_delta=1e-6`;
respeita `min_epochs`). GL só pega PIORA sustentada — sem isso, run estagnada rodava até
`n_epochs` (ex: FNO2d mae unified, best na época 209, rodou até 499). Simulado sobre 22
runs mesh_* com histórico longo: ~1540 épocas economizadas, nenhuma run com best >1% pior.
Testado antes `5` (decisão revertida pelo usuário no mesmo dia): ~2560 épocas
economizadas, mas 6 runs perdiam 0,4%–5,3% no best (melhoras lentas espaçadas >50 épocas).

### Dataloader — `src/neural_op/dataloaders/grid_loader.py`

`build_loaders(cfg, arch_entry)` — seleciona modo via `arch_entry.loader_mode`.
Split feito ao nível de chunk; `train_epoch`/`eval_epoch` usam contador de batches (sem `__len__`).
`CUDAPrefetcher` é agnóstico ao tipo de batch (tensor, dict, list, tuple).
Windows: `num_workers=0` por padrão.

---

## Formatos de dados

### Grade regular
`[B, C, H=80, W=240]` — 80 radial × 240 angular, 0°–120°.

### Chunks — `data/torch/data_chunks/<dataset>/data_chunk_XXXX.pt`

Gerados por `build_data_chunks.py` a partir de `.npz` produzidos por `gen_npz_structures.py`.
Modo detectado automaticamente pelo primeiro `.npz` (`has_graph = 'edge_index' in files`).

**Modo grid** (`GridSampleUnifier` — FNO2d, MaskedFNO2d):
```
x_hw:  [B, Cx, H, W]  float32  entrada na grade base (point query — fronteiras nítidas)
y_hw:  [B, Cy, H, W]  float32  saída na grade base
dim:   (H, W)
```

**Modo qtree** (`QtreeSampleUnifier` — FNO_GNN, MaskedFNO_GNN):
```
x_hw:       [B, Cx, H, W]  float32  entrada na grade baixa (média ponderada qtree)
y_hw:       [B, Cy, H, W]  float32  saída na grade baixa
x_hw_grid:  [B, Cx, H, W]  float32  entrada na grade base (point query — fronteiras nítidas)
y_hw_grid:  [B, Cy, H, W]  float32  saída na grade base (point query — fronteiras nítidas)
node_x:     [S_tot, N_x]   float32  features de entrada por nó (colunas do parser)
node_y:     [S_tot, N_y]   float32  features de saída por nó
edge_index: [2, E_tot]     int64    arestas bidirecionais; índices globais no chunk
edge_attr:  [E_tot, N_ea]  float32  features de aresta (colunas do parser)
L:          [B]            int64    nós por amostra
E_L:        [B]            int64    arestas (bidirecionais) por amostra
dim:        (H, W)
```

**Distinção `x_hw` vs `x_hw_grid`** (modo qtree):

| Campo | Origem | Fronteiras |
|-------|--------|-----------|
| `x_hw` | `qtree_to_lowres_hw` — média ponderada por área das folhas | **suavizadas** em células mistas |
| `x_hw_grid` | point query Shapely no centro de cada célula base | **nítidas** |
| `y_hw` | `qtree_to_lowres_hw` das folhas | suavizadas |
| `y_hw_grid` | point query FEMM no centro de cada célula base | nítidas |

MaskedFNO2d usa `x_hw_grid` para que `_make_material_masks` classifique sem pixels mistos.
FNO_GNN usa `x_hw` (média) — GNN corrige nas fronteiras.

### CSVs raw — `data/raw/` (fonte única)

Um único `depth_qt_` por amostra (DFS de `save_material_mu_qtree`). M e B amostrados a posteriori.

| Prefixo CSV | Depth | Origem | Conteúdo |
|-------------|-------|--------|---------|
| `Mu_r_qt_` | `depth_qt_` | Shapely | mu_r no ponto central (point query) |
| `Frac_dom_qt_` | `depth_qt_` | Shapely | fração do material dominante na bbox cartesiana |
| `Mag_M_qt_` | `depth_qt_` | Shapely | M no ponto central, sem média |
| `Mag_Bx_qt_` | `depth_qt_` | FEMM | campo radial |
| `Mag_By_qt_` | `depth_qt_` | FEMM | campo angular |
| `Mu_r_grid_` | — | Shapely | point query H×W; base de `x_hw_grid` |
| `Mag_M_grid_` | — | Shapely | point query H×W |
| `Mag_Bx_grid_` | — | FEMM | point query H×W; base de `y_hw_grid` |
| `Mag_By_grid_` | — | FEMM | point query H×W |

Campos derivados em parse (não salvos): `material_id`, `cell_area`, `r_base`, `c_base`.
`normal_x`/`normal_y` — pós-processamento futuro (gradiente de frac_dom); placeholder zeros.

### Features por nó — `node_x` (motor quadtree)

Tabela completa de índices disponíveis; cada parser seleciona o subconjunto via `node_x_cols`.

| Índice | Feature | Detalhes |
|--------|---------|---------|
| `[:,0]` | material_id | categórico: ferro=0, ar=1, ima=2, bobina=3 (derivado de mu_r) |
| `[:,1]` | mu_r | permeabilidade no ponto central da célula |
| `[:,2]` | M | magnetização no ponto central (sem média) |
| `[:,3]` | cell_area | área em unidades de célula base = (1/2^depth)² |
| `[:,4]` | r_base | coord. radial normalizada [0,1] do centro da célula base |
| `[:,5]` | c_base | coord. angular normalizada [0,1] do centro da célula base |
| `[:,6]` | frac_dom | fração de área do material dominante na célula |
| `[:,7]` | normal_x | normal à interface mais próxima, zero em homogêneas *(futuro)* |
| `[:,8]` | normal_y | normal à interface mais próxima, zero em homogêneas *(futuro)* |

**`node_y`** — `[S, 2]`: `[:,0]` = Bx (campo radial, T), `[:,1]` = By (campo angular, T).

### Features por aresta — `edge_attr` (motor quadtree)

| Índice | Feature | Detalhe | Lazy? |
|--------|---------|---------|-------|
| `[:,0]` | delta_r | diferença radial j−i em unidades de célula base | não |
| `[:,1]` | delta_c | diferença angular com wrap | não |
| `[:,2]` | shared_length | comprimento da fronteira compartilhada = min(side_i, side_j) | não |
| `[:,3]` | center_dist | distância euclidiana entre centros | não |
| `[:,4]` | delta_mu | mu_r[i] − mu_r[j] (**origem − destino**) | **sim** — só se `4 ∈ edge_attr_cols` |

**Nota sobre `delta_mu` (2026-07-17)**: convenção de sinal **invertida** em relação a
`delta_r`/`delta_c` (que são `destino − origem`, j−i). `delta_mu` é `origem − destino` (i−j):
positivo quando a aresta sai de material de alta permeabilidade para baixa (ex: ferro→ar),
negativo no sentido inverso. Escolha deliberada para uso pelo `FNO_GNN_v2`/`FNO_GNN_V2_PARSER` —
nenhum parser anterior usava essa coluna, então não há impacto em runs já treinadas.
`mu_r` (node_x col 1) já é o valor real de `PERMEABILITY[material]` (não `material_id`) desde a
implementação original — assume só ~5 valores fixos (ferro=5000, ímãs=1.05, cobre=0.999,
vácuo=1.0), sem variação contínua dentro do material (sem modelo de saturação nesta etapa).

### Canais de `x_hw` e `y_hw` — `[B, 2, H, W]`

| Canal | `x_hw` | `y_hw` |
|-------|--------|--------|
| 0 | Mu_r | Bx |
| 1 | M | By |

### Convenções do grafo

- **Nós**: cada folha do quadtree é um nó (S nós por amostra)
- **Arestas**: adjacência física — fronteira compartilhada de comprimento positivo
- **Bidirecional**: par (i,j) e (j,i) presentes; delta_r/delta_c/delta_mu invertidos no reverso; shared_length e center_dist idênticos
- **Wrap angular**: coluna 0 conecta com coluna W−1 (anel); radial sem wrap
- **Slicing**: nós via `cumsum(L)`, arestas via `cumsum(E_L)`; `edge_index` com índices globais no chunk

### MaskedFNO_GNN — canais de saída

| Tensor | Shape | Layout |
|--------|-------|--------|
| `y_hw_8` (FNO, grade) | `[B, 8, H, W]` | `[:,2m:2m+2]` = (Bx_m, By_m) do material m |
| `y_nodes_8` (GNN, nós) | `[S_tot, 8]` | mesmo layout por coluna |
| `assemble_grid(pred8, masks)` | `[B, 2, H, W]` | seleciona canais pelo pixel mask |
| `assemble_nodes(y_nodes_8, ids)` | `[S_tot, 2]` | seleciona canais pelo `material_id` do nó |

`material_id` nos nós derivado de `node_x[:,1]` (mu_r) via `_material_ids_from_mu` —
mesmos thresholds de `_make_material_masks`: `mu>10`→ferro, `1.01<mu<10`→ima, `mu<0.9995`→cobre, else→ar.

### Batch emitido pelo dataloader

**mode='grid'** — tuple `(x, y)`, shape `[B, 2, H, W]` cada.

**mode='qtree'** — dict (campos já no device via CUDAPrefetcher):
```
x_hw        [B, 2, H, W]   canal 0=Mu_r, canal 1=M
y_hw        [B, 2, H, W]   canal 0=Bx, canal 1=By
node_x      [S_tot, N_x]   features de nó (entrada)
node_y      [S_tot, N_y]   features de nó (alvo)
edge_index  [2, E_tot]     arestas com índices globais no batch
edge_attr   [E_tot, N_ea]  features de aresta
L           [B]            nós por amostra
E_L         [B]            arestas por amostra
```

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

## Estrutura de logs — `data/logs/`

```
data/logs/{problem}/{arch}/run_XXXX/
    config.json       # asdict(NnCfg) + n_params, device, start_time
    metrics.jsonl     # {epoch, train_loss, test_loss, lr, train_time_s, eval_time_s, samples_per_s}
    status.txt        # running → done / stopped / failed
    notes.txt         # edição manual
    checkpoints/
        best.pth      # model_state_dict + optimizer_state_dict + epoch
        latest.pth
    model_final.pth
```

Git: apenas `.gitkeep` de cada nível é commitado; `run_*/` ignorado.

**Gitignore — padrão para novo `{problem}/{arch}`:**
```
!/data/logs/{problem}/
/data/logs/{problem}/*
!/data/logs/{problem}/.gitkeep
!/data/logs/{problem}/{arch}/
/data/logs/{problem}/{arch}/*
!/data/logs/{problem}/{arch}/.gitkeep
```

---

## Benchmark — `src/bench/`

Datasets externos convertidos para `data_chunk_*.pt` por parsers em `scripts/parsers/`.
Formato de saída idêntico ao do motor: `{x_hw: [B,C,H,W], y_hw: [B,C,H,W]}`.

`DATASET_REGISTRY` em `src/bench/datasets/__init__.py` — cada `DatasetEntry` tem: `in_channels`, `out_channels`, `resolution`, `has_interface`, `chunk_dir`, `description`.

`eval_results.jsonl` — schema livre, cada linha auto-descritiva:
```jsonl
{"eval_dataset": "darcy_beta1", "relative_l2": 0.031, "interface_error": 0.058, "timestamp": "..."}
```

Implementado: Darcy (PDEBench, 10k amostras, 128×128, `in_ch=1`/`out_ch=1`), parsers, treino FNO2d + FNORef.
Pendente: `src/bench/metrics.py`, `src/bench/eval_manager.py`, `scripts/bench_eval.py`.

---

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
best 269; pico 18,8 GiB). Teste: `python -m scripts.pos_bateria_smooth_elem_eval [N]` (mesmas
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
  `mesh_ans_138x276_smooth_best_mse_mae` (seta `rbc.PROBLEM` pra base do GNN_PostBase
  ser o FNO2d desta bateria). Precheck: 125 chunks por arch, senão para antes de treinar.
  `TEST_SPLIT = 0.30` (`NnCfg.test_split`): teste = 37 chunks (subconjunto dos 88 de teste
  da oficial), treino = os mesmos 37 da oficial (verificado com `split_chunk_paths`,
  seed 12), 51 chunks fora — reservados p/ avaliação final independente (parada/best.pth
  escolhidos no teste de 37). Comparar com a bateria oficial: reavaliar os modelos dela
  nos mesmos 37 (ou nos 51).
  `python -m scripts.run_best_configs_smooth`.
- **Falta**: gerar as 4000 e rodar a bateria (GNN_PostBase estoura a RTX 4080 16 GB — VM).

## Pendências conhecidas

- **Datasets da bateria com B suavizado do FEMM (2026-10-09)** — parser e script de chunks
  e script da bateria prontos; falta gerar as 4000
  (`python -m scripts.build_smooth_ans_chunks_direct`) e rodar
  `python -m scripts.run_best_configs_smooth`; ver "Suavização de B do FEMM a partir do
  `.ans`" acima.

- **[PÓS-TRABALHO] Divergência de treino na bateria definitiva (2026-10-04)** — 3 dos 8 runs de
  `mesh_ans_138x276_unified_best_mse_mae` divergiram (train_loss explode junto com test_loss —
  instabilidade, não overfitting) e foram parados pelo GL: FNO_GNN mse (época 224, best 209),
  FNO_GNN mae (época 88, best 79 — antes de `min_epochs`), GNN_PostBase mse (época 102, logo após
  o corte de lr; trava em ~0,166 e não recupera, best 99). Já acontecia na bateria anterior.
  Decisão do usuário: usar o `best.pth` desses runs por enquanto; investigar (ex: clipping de
  gradiente) só depois do trabalho pronto. GNN_PostBase mae terminou em n_epochs (best 479/500)
  — decisão: sem mais épocas.
- **[PÓS-TRABALHO] `relative_l2` (B6)** — fora da bateria por ora (só mse/mae); cogitar incluir
  depois que o pipeline estiver garantido, para ver se dá o mesmo resultado.

- **Datasets unificados `mesh_ans_138x276_unified/<arch>` (2026-10-01)** — script pronto e
  validado (smoke de 64 amostras); falta rodar as 4000 (`python -m
  scripts.build_unified_ans_chunks_direct`, ou a versão com npz
  `scripts.build_unified_ans_datasets`) e retreinar/reavaliar as 4 archs (mse e mae) sobre
  eles, refazendo a tabela de erro ponto a ponto/integral de superfície. Treino:
  `python -m scripts.run_best_configs` (já adaptado — 8 treinos, GNN_PostBase com base =
  FNO2d da própria bateria com a mesma loss). Testado em sandbox (chunks de 64 amostras,
  2 épocas): FNO2d/FNO_GNN/FNO_BipartiteGNN ok; **GNN_PostBase dá CUDA OOM na RTX 4080
  16 GB com batch 32** (independe da base — reproduzido com a base antiga; mesma config que
  rodou na bateria de 14/09, provavelmente em outra máquina) — decidir: rodar em máquina
  com GPU maior, ou reduzir batch_size só do GNN_PostBase. Falta: reconstruir e COMMITAR o script de avaliação da
  tabela (o antigo, `.claude_scratch/eval_surface_integral*.py`, só sobrou em `.pyc`).
  Ver "Datasets unificados a partir do raw mesh_ans_138x276".
- **Interpolação grafo↔grade H×W (malha real do FEMM) — resolvida, não é mais pendência**
  (atualizado 2026-07-27): `r_base`/`c_base` por nó + `_interpolate_fno_to_nodes`
  (`src/neural_op/archs/fno_gnn.py`, `F.grid_sample` bilinear) já fazem essa interpolação —
  reaproveitado tal qual por `GNN_PostBase`. Barato (op de GPU), sem necessidade de teste
  extra além do que já roda em `FNO_GNN`/`GNN_PostBase` normalmente.
- **B por nó via fórmula fechada (numpy, sem COM) — possibilidade de otimização, não
  implementada em produção** (achado 2026-08-07, ver "Extração de dados direto do arquivo
  .ans" acima): `B=curl(A)` por elemento P1 bate com `mo_getb(smooth='off')` a nível de
  ponto flutuante quando calculado em metros — dá pra vetorizar 100% em numpy e eliminar o
  loop de `mo_getb` por nó em `femm_mesh.py::generate_mesh_sample` (~3,5s/amostra, um dos
  dois custos residuais de COM). Não migrado ainda: precisa decidir se o dataset deve manter
  o campo suavizado atual (`mo_smooth` default, o que já está nos 4000 chunks existentes) ou
  trocar para o valor bruto por elemento — não são idênticos (ver nota acima) — e isso muda
  o groundtruth de treino, não é uma troca neutra.
- **`mode='femm_mesh_v2'` — pipeline (parser+chunks+arquitetura+eval) implementado, falta
  construir os chunks e treinar em escala** (atualizado 2026-08-13, ver "Grafo duplo
  vértices+elementos" acima): `parse_ans_gzip_sample` deriva grafo de vértices + grafo de
  elementos + arestas cruzadas + grade H×W direto do `.ans.gz`, sem resimular — a conectividade
  de elementos não precisou de rastreamento de faces via `nbr1-3` (ideia cogitada em
  2026-08-07, descartada): `elems` (`n1,n2,n3` por triângulo) já vem pronto no `.ans`, suficiente
  pra tudo (malha de vértices via `_build_edges`, arestas cruzadas via incidência direta
  elemento→seus-3-vértices). `FNO_BipartiteGNN` (`ARCH_REGISTRY`) treina e calcula
  `mae_hw`/`mae_graph`. Status por etapa:
  1. **raw** (`generate_data_femm_mesh_v2.py`) — as 4000 amostras (`dataset='mesh_ans_138x276'`)
     já foram geradas (`data/raw/mesh_ans_138x276/*.ans.gz`, 07/08).
  2. **npz staging** (`gen_npz_structures.py::_run_femm_mesh_v2`) — as 4000 já foram processadas
     1x sequencialmente (10-11/08, ~6h18min); paralelizado em 2026-08-13 (ver entrada na árvore
     de arquivos acima), validado byte-a-byte idêntico, mas não re-executado em escala (nada a
     re-rodar — os 4000 `.npz` já existem em `data/temp/samples_npz/mesh_ans_138x276/`).
  3. **chunks** (`build_data_chunks_femm_mesh_v2.py`) — **ainda não rodado nenhuma vez**;
     `data/torch/data_chunks/mesh_ans_138x276/` não existe. Esse é o passo que falta pra
     destravar treino real em escala (hoje só validado com chunks sintéticos pequenos em
     `tests/proto_femm_mesh_v2_arch_smoke.py`/testes ad-hoc de sessão, não com o dataset real).
  4. **`eval_fn`/plot de malha dupla** — implementado 2026-08-13 (`femm_mesh_v2_eval_fn` +
     `EvalCfg.show_b`, ver seção "Eval" acima); não era mais stub, já não é pendência.
  5. **`scripts/eval.py`** — não precisou de adaptação nenhuma: já despacha genericamente via
     `ARCH_REGISTRY[arch].eval_fn`, funciona pra `FNO_BipartiteGNN` sem mudança nenhuma no
     script; não era pendência real, só não tinha sido testado.
- **`multi_process_v2` sem `max_tasks_per_child` (2026-08-20)** — encontrado ao rodar
  `gen_npz_structures.py` numa VDI Linux (Python 3.10): `ProcessPoolExecutor(max_tasks_per_child=
  ...)` só existe a partir do Python 3.11, `TypeError` em versões anteriores. Corrigido trocando a
  reciclagem de worker (necessária pelo vazamento de `matplotlib.tri`/`scipy` documentado acima)
  de "pool único, `max_tasks_per_child=1`, refill contínuo" pra "pool novo por RODADA de até
  `num_workers` tasks, sem refill no meio da rodada" — cada worker processa exatamente 1 task
  antes do pool inteiro ser destruído, mesma granularidade de reciclagem, sem depender do kwarg
  3.11+; roda igual em qualquer versão do Python. Custo: pequena bolha de pipeline por rodada
  (workers ociosos esperando a task mais lenta da rodada). Validado com 6 amostras reais,
  `MAX_WORKERS` forçado pra 2 (3 rodadas) — os 6 `.npz` gerados corretamente. `generate_data.py`/
  `generate_data_femm_mesh.py`/`generate_data_femm_mesh_v2.py` têm o mesmo padrão
  `max_tasks_per_child=1`, mas são Windows-only (abrem FEMM/COM) — não quebram na VDI Linux, não
  mexidos nesta sessão; mesma correção pode ser necessária se algum dia rodarem em Python <3.11.
- **`FNO_BipartiteGNN_v3`/parser `FEMM_MESH_V3`/`FEMM_MESH_V3_A` (2026-08-20)** — ver seção
  "`node_cell_count` + projeção de 8 vizinhos" acima: código completo (parser, arch,
  normalização, registro) implementado e validado só num smoke test ad-hoc de 4 amostras
  reais (forward+backward+metric_fn OK, gradiente finito em F1/F2/lift). Staging (`npz_parser
  ='FEMM_MESH_V3'`, mesmo `mode='femm_mesh_v2'`) e chunks em escala (4000 amostras) ainda não
  rodados — próximo passo antes de treinar de verdade.
- **Malha FEMM (`mode='femm_mesh'`) — pipeline pronto e unificado, falta validação end-to-end
  do dataset completo** (atualizado 2026-07-27): pipeline unificado com o do qtree/grid —
  mesmos 3 comandos (`generate_data`/`gen_npz_structures`/`build_data_chunks`), mesmas
  pastas-base (`data/raw/`, `data/temp/samples_npz/`, `data/torch/data_chunks/`) — ver
  "Filosofia raw → npz → chunks" e "Chunks — malha real do FEMM" acima. `FNO_GNN`/
  `GNN_PostBase` já treinam sobre esse layout **sem precisar de arch nova** — `edge_dim`/
  `grid_in_ch`/`grid_out_ch`/`node_in_ch` são auto-detectados a partir dos chunks reais
  (`_detect_chunk_dims` em `training.py`, ver "Canais/dimensões auto-detectados" acima), não
  mais hardcoded; `node_x[:,3]/[:,4]` (r_base/c_base) já bate com a convenção usada por
  `_interpolate_fno_to_nodes`. Testado com
  smoke test de 5 amostras (shapes conferem com o contrato) — falta rodar as 4000 completas
  (`python -m scripts.build_data_chunks`) e um treino real ponta a ponta pra confirmar. Sem
  arestas de wrap identificadas separadamente do resto do grafo no `edge_index` — se algum
  arch precisar tratá-las diferente (ex: não passar por conv espacial), precisa de um jeito de
  marcá-las (hoje indistinguíveis das arestas normais além de terem
  `delta_r=delta_c=center_dist=0`).
- **Chunks obsoletos `mesh_138x276`/`mesh_138x276old`** (`data/torch/data_chunks/`,
  encontrados 2026-07-27): sobras de uma versão anterior de `build_data_chunks_femm_mesh.py`
  que lia direto do staging bruto sem aplicar parser nenhum — `node_x` tem as 9 colunas cheias,
  não as 5 do `FEMM_MESH_PARSER`. **Nota 2026-08-05**: desde a auto-detecção de canais
  (`node_in_ch`, ver "Canais/dimensões auto-detectados" acima), apontar `FNO_GNN`/
  `GNN_PostBase` pra esses chunks não dá mais erro de shape — aceita as 9 colunas
  silenciosamente, sem o guard-rail acidental que existia antes. Diferem só pelas chaves de grade (`old` = `x_hw_grid`/`y_hw_grid`, anterior ao rename de
  2026-07-27; `mesh_138x276` = `x_hw`/`y_hw`, posterior ao rename mas anterior à etapa de
  parser). Nenhum dos dois é utilizável hoje como dataset de treino — mantidos por enquanto
  (não apagados), não usar.
- **Eval — auditoria de pendências (2026-08-05)**: `_plot_fno_gnn_mesh`/`fno_eval_fn`
  (`src/neural_op/archs/eval.py`) corrigidos pra suportar alvo escalar (A) além de vetorial
  (Bx,By) — ver seção `_detect_chunk_dims` acima pro motivo. Restam, cientes e não corrigidas:
  `masked_fno_eval_fn`/`masked_fno_gnn_eval_fn` continuam fixos em Bx,By (mesma limitação de
  `MaskedFNO2dConfig`/`MaskedFNO_GNNConfig`, ver acima); o branch qtree "puro" (não-mesh) de
  `fno_gnn_eval_fn` também hardcodeia 2 canais — inofensivo hoje porque `target_field='A'` só
  existe no pipeline `femm_mesh` (bloqueado por assert no caminho qtree), mas quebraria do
  mesmo jeito que `_plot_fno_gnn_mesh` quebrava se isso mudar no futuro.
- `tests/smoke_test_refactor.py` (achado 2026-08-05, não corrigido — fora do escopo da sessão):
  `CHUNK_PATHS` (linha ~33) aponta pra `data/torch/quad_chunks/uniform/` — path (`quad_chunks`)
  e dataset (`uniform`) antigos, nenhum dos dois existe mais. 2 dos 7 subtestes falham
  ("treino curto", "early stop", + 2 "SKIP" em cascata) por `FileNotFoundError` em
  `build_loaders`. Não é causado por nenhuma mudança recente — a doc "33/33 OK" no topo do
  arquivo já está desatualizada. `tests/profile_fno_gnn.py` também estava quebrado (imports de
  `src.neural_op.models`/`src.config`/`training_utils.mse_loss`, todos migrados/removidos) —
  comentado por inteiro com `# [REMOVIDO]` nessa sessão (mesmo padrão de `models.py`).
- `normal_x`/`normal_y` (node_x cols 7,8) — gradiente de frac_dom; placeholder zeros por enquanto
- Benchmark Fase D: `scripts/bench.py` + `src/bench/metrics.py` implementados (2026-06-22, pool global de pixels, treino+teste, CPU/GPU); falta só `src/bench/eval_manager.py`
- Novos parsers para GINO/MGNO (sessão futura)
- `todo_build_data_chunks.md` em memory: build_data_chunks não detecta chunks incompletos no re-run
- **Losses focadas em erro percentual (média/p95)**: `LOSS_REGISTRY` atual (`mse`/`mae`/`relative_l2`/`masked_fno_loss`/`single_material_fno_loss`/`masked_fno_gnn_loss`) só minimiza erro médio (quadrático ou absoluto), sem normalização pela região relevante (threshold) nem foco na cauda (p95). Candidatos a explorar: quantile/pinball loss (penaliza p95 assimetricamente) ou peso extra nos top-k piores pixels do batch.
  Motivação (2026-06-22): no benchmark agregado (`scripts/bench.py`), o erro médio do FNO (estágio base) está **menor** que o do GNN do `GNN_PostBase` treinado em cima dele — esperado o oposto, já que o GNN deveria refinar/reduzir o erro do FNO. Investigar se é falta de loss adequada no treino do GNN, `lambda_loss`, overfitting, ou outro problema antes de trocar a loss.
- **Limpeza futura dos pipelines** (2026-07-23): planejada uma limpeza geral em `src/data_gen/`,
  `src/neural_op/`, `scripts/` para remover código não utilizado (funções/branches mortas,
  parsers/archs sem uso, scripts obsoletos) — fazer só depois que o método de geração via malha
  real do FEMM (ver seção abaixo) estiver estável em produção, para não remover algo que ainda
  vai ser reaproveitado. Seguir a regra de comentar com `# [REMOVIDO]` em vez de apagar, por método.
