import glob
import json
import torch
from pathlib import Path
from src.configs.bench import BenchCfg
from src.neural_op.archs import ARCH_REGISTRY
from src.bench.metrics import METRICS_REGISTRY, aggregate

# ── Configuração ───────────────────────────────────────────────────────────────
_bench = BenchCfg()

# ── Carregar config da run ────────────────────────────────────────────────────
run_dir  = Path(_bench.run_dir)
cfg_dict = json.loads((run_dir / 'config.json').read_text())
arch     = cfg_dict['arch']
dataset  = cfg_dict['dataset']

# ── Reconstruir e carregar modelo ─────────────────────────────────────────────
entry = ARCH_REGISTRY[arch]
if hasattr(entry.cfg_cls, 'from_dict'):
    arch_cfg = entry.cfg_cls.from_dict(cfg_dict['arch_cfg'])
else:
    arch_cfg = entry.cfg_cls(**cfg_dict['arch_cfg'])
model = entry.make_model(arch_cfg)
model.eval()

if _bench.checkpoint == 'final':
    ckpt_path = run_dir / 'model_final.pth'
else:
    ckpt_path = run_dir / 'checkpoints' / f'{_bench.checkpoint}.pth'

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd   = {k: v for k, v in ckpt['model_state_dict'].items() if k != '_metadata'}
model.load_state_dict(sd)
print(f"Modelo carregado: {ckpt_path}  (epoch {ckpt['epoch']})")

# ── Avaliar todas as chunks do dataset ────────────────────────────────────────
collect_fn  = METRICS_REGISTRY[arch]
chunk_paths = sorted(glob.glob(f'data/torch/data_chunks/{dataset}/data_chunk_*.pt'))
print(f"Avaliando {len(chunk_paths)} chunks (treino+teste) de '{dataset}'...")

pooled = {}
for cp in chunk_paths:
    d = torch.load(cp, map_location='cpu')
    for stage, arr in collect_fn(model, d, _bench.irrelevance_threshold).items():
        pooled.setdefault(stage, []).append(arr)

# ── Agregar (pool global de pixels relevantes) e reportar ───────────────────
print(f"\nResultado agregado — {arch} (pool de pixels relevantes em todas as chunks):")
for stage, (mean, median, p95, n) in aggregate(pooled).items():
    print(f"  {stage:>5} — média={mean:.2f}%  mediana={median:.2f}%  p95={p95:.2f}%  (n={n} pixels)")
