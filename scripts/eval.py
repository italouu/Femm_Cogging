import glob
import json
import torch
from pathlib import Path
from src.configs.eval import EvalCfg
from src.neural_op.archs import ARCH_REGISTRY
from src.neural_op.dataloaders.grid_loader import split_chunk_paths

# ── Configuração ───────────────────────────────────────────────────────────────
_eval = EvalCfg()

# ── Carregar config da run ────────────────────────────────────────────────────
run_dir  = Path(_eval.run_dir)
cfg_dict = json.loads((run_dir / 'config.json').read_text())
arch     = cfg_dict['arch']
dataset  = cfg_dict['dataset']

if _eval.dataset_override is not None:
    dataset = _eval.dataset_override
    if _eval.chunk_index is not None:
        print(f"  [AVISO] dataset_override='{dataset}' definido — ignorando chunk_index "
              f"(split de treino só é reproduzível no dataset original da run). Usando chunk_name='{_eval.chunk_name}'.")

# ── Reconstruir e carregar modelo ─────────────────────────────────────────────
entry = ARCH_REGISTRY[arch]
if hasattr(entry.cfg_cls, 'from_dict'):
    # archs com campos init=False (ex: GNN_PostBaseConfig) — reconstrói direto do
    # snapshot salvo, sem rechamar __post_init__ (que dependeria de base_run_dir existir)
    arch_cfg = entry.cfg_cls.from_dict(cfg_dict['arch_cfg'])
else:
    arch_cfg = entry.cfg_cls(**cfg_dict['arch_cfg'])
model = entry.make_model(arch_cfg)
model.eval()

if _eval.checkpoint == 'final':
    ckpt_path = run_dir / 'model_final.pth'
else:
    ckpt_path = run_dir / 'checkpoints' / f'{_eval.checkpoint}.pth'

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd   = {k: v for k, v in ckpt['model_state_dict'].items() if k != '_metadata'}
model.load_state_dict(sd)
print(f"Modelo carregado: {ckpt_path}  (epoch {ckpt['epoch']})")

# ── Resolver chunk dentro do split treino/teste da run ────────────────────────
chunk_paths  = sorted(glob.glob(f'data/torch/data_chunks/{dataset}/data_chunk_*.pt'))
chunk_dir    = Path(f'data/torch/data_chunks/{dataset}')
split_path   = run_dir / 'split.json'

if split_path.exists() and _eval.dataset_override is None:
    # split.json é a fonte de verdade gravada no momento do treino — reproduz o
    # split exato mesmo que o diretório do dataset tenha mudado desde então
    # (chunks regenerados/adicionados/removidos quebrariam a reconstrução por seed).
    split_dict  = json.loads(split_path.read_text())
    train_paths = [str(chunk_dir / name) for name in split_dict['train']]
    test_paths  = [str(chunk_dir / name) for name in split_dict['test']]

    missing = [p for p in train_paths + test_paths if not Path(p).exists()]
    if missing:
        print(f"  [AVISO] {len(missing)} chunk(s) do split.json não existem mais em "
              f"{chunk_dir} — dataset mudou desde o treino desta run. Faltando: "
              f"{[Path(p).name for p in missing]}")
else:
    if _eval.dataset_override is None:
        print("  [AVISO] split.json não encontrado nesta run (run anterior a esta "
              "mudança) — reconstruindo split via train_split/seed. Se o dataset "
              "mudou desde o treino, este split pode divergir do original.")
    train_paths, test_paths = split_chunk_paths(
        chunk_paths, cfg_dict['train_split'], cfg_dict['split_seed'],
        cfg_dict.get('test_split')
    )
split_paths = train_paths if _eval.split == 'train' else test_paths

if _eval.chunk_index is not None and _eval.dataset_override is None:
    chunk_path = Path(split_paths[_eval.chunk_index])
else:
    chunk_path = Path(f'data/torch/data_chunks/{dataset}/{_eval.chunk_name}.pt')
    if chunk_path not in (Path(p) for p in split_paths):
        other = 'treino' if _eval.split == 'test' else 'teste'
        print(f"  [AVISO] {chunk_path.name} não pertence ao split '{_eval.split}' "
              f"desta run (pertence a {other} ou não existe no dataset).")

print(f"Chunk: {chunk_path.name}  (split='{_eval.split}')")

# ── Carregar chunk e avaliar ───────────────────────────────────────────────────
d = torch.load(chunk_path, map_location='cpu')
entry.eval_fn(model, d, _eval)
