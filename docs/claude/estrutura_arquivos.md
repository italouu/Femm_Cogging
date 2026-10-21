# Estrutura de arquivos

> Árvore após a limpeza dos obsoletos (2026-10-21, commits O1–O6). Tudo o que está em `src/`
> e `scripts/` é alcançável a partir dos pontos de entrada abaixo (ou é a bateria carregada
> dinamicamente por `run_battery`). A árvore antiga, com os pipelines CSV/qtree/unificado,
> está em `docs/_arquivo/CLAUDE_backup_2026-10-11.md`.

## Pontos de entrada (`python -m scripts.<nome>`, a partir da raiz)

| Script | O que faz |
|---|---|
| `generate_data_femm_mesh_v2` | raw: sorteio (`BLDC_Process.generate_samples` + `add_phase_currents`) → `valid_designs.csv` + FEMM por amostra → `data/raw/<dataset>/sample_XXXXXX.ans.gz` (Windows, pyfemm) |
| `fix_stack_length` | correção pontual do `stack_length [mm]` no `valid_designs.csv` de um raw antigo |
| `build_smooth_datasets` | os 5 datasets de uma vez: chama `build_smooth_ans_chunks_direct` (FNO2d, FNO_GNN, GNN_PostBase, FNO_BipartiteGNN) e `build_elem_smooth_chunks` (Elem) |
| `train` | treino de um `NnCfg` (`scripts/train.py::run`) |
| `run_battery <nome> [--dry-run]` | bateria `src/configs/batteries/<nome>.py::BATTERY` (oficial: `bldc`) |
| `postprocess <subcomando>` | eval, figures, surface, regions, torque, saturation, timing, train-table |

## Árvore

```
scripts/
    generate_data_femm_mesh_v2.py   raw (.ans.gz) + valid_designs.csv
    fix_stack_length.py             correção do stack length no CSV
    build_smooth_datasets.py        orquestra os 5 datasets (precheck, disco, contagem final)
    build_smooth_ans_chunks_direct.py   chunks das 4 archs (parse_ans_gzip_sample_smooth)
    build_elem_smooth_chunks.py     chunks da Elem (parse_ans_gzip_sample_elem_smooth) + precheck
    train.py                        run(NnCfg): loaders, normalização, fit, ModelManager
    run_battery.py                  bateria com manifesto/retomada (data/logs/_batteries/<nome>/)
    postprocess.py                  executável do pós-processamento (código em src/postprocess/)

src/configs/
    __init__.py         re-exporta DatagenConfig, NnCfg, arch configs, MonitorCfg, loss configs
    datagen.py          DatagenConfig (dataset, grade, sorteio, correntes, chunks, workers)
    training.py         NnCfg + FNOConfig, FNO_GNNConfig, GNN_PostBaseConfig,
                        FNO_BipartiteGNNConfig, FNO_BipartiteGNN_ElemConfig
                        (arch/loss derivados do TIPO de arch_cfg/loss_cfg; canais detectados
                        dos chunks em __post_init__; defaults = bateria bldc)
    loss.py             MseLossCfg, MaeLossCfg, LOSS_CFG_REGISTRY
    monitor.py          MonitorCfg (heartbeat, GL, paciência, min_epochs)
    battery.py          BatteryRun, BatteryCfg
    batteries/bldc.py   BATTERY: 10 runs (5 archs × mse/mae), dataset BLDC_138x276/<arch>

src/data_gen/
    motor_constants.py  materiais, N_POLES_SECTOR, WINDING_SECTOR, STACK_LENGTH_MM,
                        FEMM_MATERIAL_DEFS (literais puros, sem femm)
    motor_model.py      BLDC_Process (parâmetros, generate_samples, add_phase_currents,
                        export_params) → BLDC_FEMM_Model (materiais, primitivos de desenho)
                        → BLDC_FEMM_Model_Sym120 (setor 120°, periódico)
                        → BLDC_FEMM_Model_Sym120_Annular (truncamento radial)
    femm_mesh.py        save_ans_gzip_sample (desenha + malha + mi_analyze, grava .ans.gz)
    chunks.py           LAYOUT_BY_ARCH, write_chunk (escrita atômica dos data_chunk_XXXX.pt)
    parsers/            raw .ans.gz -> arrays (sem femm; roda no Linux)
        ans_parsing.py          leitura do .ans (_read_ans_gz, _parse_solution), materiais e
                                correntes por rótulo, arestas/wrap, áreas, material do nó,
                                grade polar, _to_polar
        ans_b_smoothing.py      porte numpy do B suavizado do FEMM (element_b, nodal_b, point_b)
        femm_mesh_v2.py         parse_ans_gzip_sample: grafo de vértices + elementos + arestas
                                cruzadas + x_hw (target_field 'B' = curl(A), None = sem gabarito)
        femm_mesh_smooth.py     parse_ans_gzip_sample_smooth -> layouts de SMOOTH_ARCHS
                                (gabarito B suavizado, polar)
        femm_mesh_elem.py       parse_ans_gzip_sample_elem (layout Elem; contrato do layout na
                                docstring; 'B' = gabarito exato do torque), check_elem_layout
        femm_mesh_elem_smooth.py  parse_ans_gzip_sample_elem_smooth (Elem com B suavizado)

src/neural_op/
    archs/__init__.py   ArchEntry, ARCH_REGISTRY (FNO2d, FNO_GNN, GNN_PostBase,
                        FNO_BipartiteGNN, FNO_BipartiteGNN_Elem)
    archs/_blocks.py    MLP, SpectralConv, FNO_Blocks, EdgeConvLayer, GNN, BipartiteGNN, ...
    archs/fno.py        FNO2d, fno_step_fn, fno_metric_fn
    archs/fno_gnn.py    FNO_GNN, _interpolate_fno_to_nodes, _rescale_fno_to_node_space
    archs/gnn_post_base.py   GNN_PostBase (GNN sobre FNO2d congelado)
    archs/femm_mesh_v2_gnn.py   FNO_BipartiteGNN, step/metric da bipartite
    archs/femm_mesh_elem_gnn.py FNO_BipartiteGNN_Elem
    archs/interp.py     interpolate_grid_to_nodes (cell_centered_border)
    dataloaders/grid_loader.py  ChunkStreamDataset, ChunkMeshV1Dataset, ChunkFemmMeshV2Dataset,
                        collates, CUDAPrefetcher, split_chunk_paths, build_loaders
    normalization.py    Normalizer (z-score, stats dos chunks de treino, _LAYOUTS por arch)
    losses.py           LOSS_REGISTRY (mse, mae)
    monitor.py          TrainingMonitor
    training_utils.py   fit, checkpoints, contagem de parâmetros

src/training/model_manager.py   ModelManager (run_XXXX, config.json, metrics.jsonl, status)

src/postprocess/       pós-processamento oficial (só gabarito smooth)
    samples.py          designs, caminhos, subsets teste/fora, parse smooth na hora
    inference.py        load_model/load_set/find_runs, predict, floor_nodes
    metrics.py          erro de superfície (Accum, quadraturas malha × malha / grade × malha)
    fields.py           |B|, elementos <-> nós
    regions.py          interface/interior/canto
    torque.py           torque FEMM × gabarito × modelos
    saturation.py       saturação do aço × curva B-H
    timing.py           custo FEMM × pré-processamento + inferência
    train_table.py      tabela de treino a partir dos logs
    plots/              primitives, training, errors, fields, saturation

docs/
    claude/             contratos_e_formatos.md, pipelines_e_archs.md, estrutura_arquivos.md
    resultados_baterias.md
    figures/            figuras de geometria/grade
    _arquivo/           histórico (backups do CLAUDE.md, pendências e reestruturação antigas,
                        malha FEMM v1, panorama de treinos, bateria wrap circular)

resultados/<dataset>/dados/   saída versionada do postprocess (plots/ fora do git)

data/   (fora do git, estrutura via .gitkeep)
    raw/<dataset>/            sample_XXXXXX.ans.gz + valid_designs.csv
    torch/data_chunks/<dataset>/<arch>/   data_chunk_XXXX.pt + caches norm_stats_train_v2_*.json
    logs/<problem>/<arch>/run_XXXX/       config.json, metrics.jsonl, checkpoints/, model_final.pth
    logs/_batteries/<nome>/manifest.json
    temp/                     temporários (parse, FEMM, testes)

tests/  (local, fora do git) — scripts avulsos de validação/protótipo; teste de equivalência
        da limpeza: python -m tests.equiv_gambiarras chunks|predict|train
```
