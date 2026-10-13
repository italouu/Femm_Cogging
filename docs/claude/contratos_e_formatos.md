# Contratos de interface, formatos de dados, logs e benchmark

> Movido de `CLAUDE.md` em 2026-10-11 (limpeza do CLAUDE.md), texto sem alteração. Referências "acima"/"abaixo" podem apontar para seções que agora estão em outro arquivo de `docs/claude/` — índice no `CLAUDE.md`. Cópia integral do original: `docs/CLAUDE_backup_2026-10-11.md`.

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
- **Desde 2026-10-13 a arch e a loss são definidas pelo TIPO de `arch_cfg`/`loss_cfg`**
  (ex: `NnCfg(arch_cfg=FNOConfig(...), loss_cfg=MaeLossCfg())`). `arch`/`loss` são
  `init=False`, derivados em `__post_init__` por tipo **exato** em `ARCH_REGISTRY`
  (`cfg_cls`) / `LOSS_CFG_REGISTRY`; classe sem correspondência única → `TypeError`.
  Continuam em `asdict` → `config.json` (`"arch"`, `"loss"`), usados para a pasta de log e
  para reconstruir a run (`ARCH_REGISTRY[cfg['arch']].cfg_cls.from_dict(...)`).
  Por isso cada classe de config tem de mapear para uma única arch/loss
  (`FNO_GNN_Field`, que compartilhava `FNO_GNNConfig`, saiu do registro).
- **Defaults** de `NnCfg` e das arch configs = configs escolhidas da bateria smooth (por arch,
  loss mae); `NnCfg()` sozinho = `FNO_BipartiteGNN_Elem` mae em
  `mesh_ans_138x276_smooth/FNO_BipartiteGNN_Elem`, `problem='mesh_ans_138x276_smooth'`
  (é o que `python -m scripts.train` treina). `scheduler` só aceita `'step'`.
- `resume_run`/`resume_checkpoint`/`resume_modified`: controle de retomada de treino.
- `monitor_cfg`: instância de `MonitorCfg` (checkpoint a cada N épocas, early stop em heartbeats, save_best).
- `normalize`/`norm_stats`: normalização z-score de x_hw/y_hw/node_x/node_y — ver seção
  "Normalização" abaixo.

**Arch configs** — campos 1:1 com construtores dos modelos; `model_kwargs` filtra campos extras (`lambda_loss`, `data_res`). Instâncias: `FNOConfig`, `FNO_GNNConfig`, `FNO_GNN_v2Config`, `MaskedFNO2dConfig`, `MaskedFNO_GNNConfig`, `SingleMatFNOConfig`, `FNORefConfig`. Exceção: `GNN_PostBaseConfig` tem campos extras `init=False` (snapshot do modelo base — ver seção `GNN_PostBase` acima).

`FNO_GNN_v2Config` (`training.py`) — subclasse vazia de `FNO_GNNConfig` (mesmos campos); existe só
para que `NnCfg.__post_init__` derive `arch='FNO_GNN_v2'` do tipo do `arch_cfg` corretamente e para
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

`scripts/bench.py` + `BenchCfg` + `src/bench/metrics.py` (`METRICS_REGISTRY` por arch): métricas
agregadas de uma run sobre todas as chunks do dataset dela. Os datasets externos (Darcy/PDEBench,
`src/bench/datasets/`, `scripts/parsers/darcy.py`) e a arch `FNO_ref` foram **removidos em 2026-10-13**.

---
