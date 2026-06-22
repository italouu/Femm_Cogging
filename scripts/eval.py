import json
import torch
from pathlib import Path
from src.configs.eval import EvalCfg
from src.neural_op.archs import ARCH_REGISTRY

# ── Configuração ───────────────────────────────────────────────────────────────
_eval = EvalCfg()

# ── Carregar config da run ────────────────────────────────────────────────────
run_dir  = Path(_eval.run_dir)
cfg_dict = json.loads((run_dir / 'config.json').read_text())
arch     = cfg_dict['arch']
dataset  = cfg_dict['dataset']

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

# ── Carregar chunk e avaliar ───────────────────────────────────────────────────
chunk_path = Path(f'data/torch/data_chunks/{dataset}/{_eval.chunk_name}.pt')
d = torch.load(chunk_path, map_location='cpu')
entry.eval_fn(model, d, _eval)
