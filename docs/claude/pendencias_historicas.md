# Pendências conhecidas (lista completa, até 2026-10-10)

> Movido de `CLAUDE.md` em 2026-10-11 (limpeza do CLAUDE.md), texto sem alteração. Referências "acima"/"abaixo" podem apontar para seções que agora estão em outro arquivo de `docs/claude/` — índice no `CLAUDE.md`. Cópia integral do original: `docs/CLAUDE_backup_2026-10-11.md`.


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
