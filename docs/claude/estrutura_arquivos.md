# Estrutura de arquivos (detalhada)

> Movido de `CLAUDE.md` em 2026-10-11 (limpeza do CLAUDE.md), texto sem alteração. Referências "acima"/"abaixo" podem apontar para seções que agora estão em outro arquivo de `docs/claude/` — índice no `CLAUDE.md`. Cópia integral do original: `docs/CLAUDE_backup_2026-10-11.md`.


```
src/configs/      __init__.py  (re-exporta DatagenConfig, NnCfg, FNOConfig, MaskedFNO2dConfig,
                                FNO_GNNConfig, FNO_GNN_v2Config, MaskedFNO_GNNConfig,
                                SingleMatFNOConfig, GNN_PostBaseConfig, MonitorCfg, EvalCfg, BenchCfg)
                  datagen.py   (DatagenConfig — geometria, simulação, qtree, chunks)
                  training.py  (NnCfg + arch configs; arch/loss derivados do TIPO de
                                arch_cfg/loss_cfg em __post_init__; defaults = bateria smooth)
                  loss.py      (configs de loss + LOSS_CFG_REGISTRY)
                  monitor.py   (MonitorCfg)
                  eval.py      (EvalCfg)
                  bench.py     (BenchCfg)
                  battery.py   (BatteryRun, BatteryCfg — 2026-10-13)
                  batteries/smooth.py (BATTERY: a única bateria do projeto, 10 runs)
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
src/bench/        metrics.py     (METRICS_REGISTRY por arch — usado por scripts/bench.py; Darcy
                                  e datasets externos removidos em 2026-10-13)
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
                  train.py       (treino avulso: run(_nn) + `__main__` = run(NnCfg()), defaults de
                                  src/configs = bateria smooth; run() reaproveitado pelo runner
                                  de bateria)
                  run_battery.py (runner de bateria — `python -m scripts.run_battery smooth
                                  [--dry-run]`; definições em src/configs/batteries/; manifesto em
                                  data/logs/_batteries/<nome>/; 2026-10-13. Substitui
                                  run_best_configs*.py, run_elem_proto*.py, run_bateria_definitiva,
                                  run_vm_*.sh — eliminados em 2026-10-13)
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
                  bench.py       (métricas agregadas de uma run — BenchCfg)
tests/            gen_test.py, test_chunk_plot.py, test_graph_plot.py (manuais, abrem plot)
                  battery_smoke.py (bateria de smoke: python -m scripts.run_battery tests.battery_smoke)
                  smoke_test_refactor.py (DESATIVADO 2026-10-13 — usa arch/loss por string)
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
