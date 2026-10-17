"""PROTÓTIPO (2026-10-16) — componentes de B treinadas em redes INDEPENDENTES (FNO_BipartiteGNN_Elem, mae).

ATENÇÃO (2026-10-17): escrito para o dataset CARTESIANO (Bx, By). Desde 2026-10-17 os parsers
smooth já geram o alvo polar (Br, Bθ) — com o dataset novo, 'br'/'bt' aqui girariam o alvo
duas vezes; nele, use o canal direto (rotulado 'bx' = Br, 'by' = Bθ) ou adapte o script.

    bx / by : componentes cartesianas (canal 0 / 1 do dataset, sem rotação)
    br / bt : componentes polares Br / Bθ (alvo girado na hora pelo ângulo de cada ponto)

Não altera src/ nem o dataset: usa o mesmo dataset da bateria bldc
(data/torch/data_chunks/BLDC_138x276/FNO_BipartiteGNN_Elem/) e a mesma config
(src/configs/batteries/bldc.py::elem_mae), com 3 mudanças feitas só em memória:
  - arch_cfg.grid_out_ch = 1 (FNO e GNN saem com 1 canal);
  - norm_stats de y_hw/node_y = só a componente escolhida
      bx/by: as do norm_stats.json do dataset (sem recálculo);
      br/bt: mean/std de (Br, Bθ) sobre todos os chunks (mesmo critério de Normalizer.fit),
             calculadas 1x e cacheadas em data/temp/proto_split_polar/ (fora do dataset);
  - problem = 'BLDC_138x276_split_<comp>' -> logs em data/logs/BLDC_138x276_split_<comp>/FNO_BipartiteGNN_Elem/.
Durante a execução, Normalizer.encode_batch (chamado pelo CUDAPrefetcher em todo batch de treino
e teste) é trocado por uma versão que (br/bt) gira y_hw/node_y para (Br, Bθ) e corta a componente
escolhida antes do z-score. Loss, step_fn, metric_fn e o treino (scripts.train.run) são os de sempre.

Ângulo de cada ponto (setor DatagenConfig.ang_1..ang_2 = 0..120°):
  y_hw  [B,2,H,W] : centro do pixel, θ_j = ang_1 + (j + 0,5)/W · (ang_2 − ang_1)  (_grid_polar_xy)
  node_y [M,2]    : centróide do elemento, θ = ang_1 + node_x[:, 1] · (ang_2 − ang_1)
                    (node_x[:, 1] = c_base do centróide, cru — fora do z-score na Elem)
  Br = Bx·cosθ + By·sinθ ;  Bθ = −Bx·sinθ + By·cosθ

O config.json da run guarda arch_cfg.grid_out_ch = 1, as stats de 1 componente e
norm_stats.split = {'comp', 'frame', 'channel'}; src/postprocess/inference.load_model/predict
reconstroem o modelo sem mudança (saída [*, 1], na componente da run). B se monta juntando as
duas runs (br + bt -> girar de volta para x,y pelo mesmo ângulo).

Uso (raiz):
    python -m scripts.proto_split_bxby br                 # treino completo (500 épocas, parada do MonitorCfg)
    python -m scripts.proto_split_bxby bt
    python -m scripts.proto_split_bxby br --smoke         # 4 chunks, 1 época -> data/logs/BLDC_138x276_split_smoke/
    python -m scripts.proto_split_bxby br --stats-only    # só calcula/mostra as stats polares
"""
import argparse
import glob as _glob
import json
import math
from pathlib import Path

import torch

from src.configs.batteries.bldc import elem_mae
from src.configs.datagen import DatagenConfig
from src.neural_op.normalization import Normalizer

# componente -> (referencial, canal dentro do referencial)
COMP = {'bx': ('xy', 0), 'by': ('xy', 1), 'br': ('polar', 0), 'bt': ('polar', 1)}
_DG = DatagenConfig()
ANG_1, ANG_2 = math.radians(_DG.ang_1), math.radians(_DG.ang_2)
STATS_DIR = Path('data/temp/proto_split_polar')
_ORIG_ENCODE_BATCH = Normalizer.encode_batch


# ── rotação (Bx, By) -> (Br, Bθ) ──────────────────────────────────────────────
def _rot(bx, by, th):
    c, s = torch.cos(th), torch.sin(th)
    return bx * c + by * s, -bx * s + by * c


def polar_hw(y_hw):
    """[B,2,H,W] cartesiano -> [B,2,H,W] (Br, Bθ); θ no centro do pixel (coluna)."""
    W = y_hw.shape[-1]
    th = ANG_1 + (torch.arange(W, device=y_hw.device, dtype=y_hw.dtype) + 0.5) / W * (ANG_2 - ANG_1)
    br, bt = _rot(y_hw[:, 0], y_hw[:, 1], th.view(1, 1, W))
    return torch.stack([br, bt], dim=1)


def polar_nodes(node_y, node_x):
    """[M,2] cartesiano -> [M,2] (Br, Bθ); θ do centróide (node_x[:, 1] cru)."""
    th = ANG_1 + node_x[:, 1].to(node_y.dtype) * (ANG_2 - ANG_1)
    br, bt = _rot(node_y[:, 0], node_y[:, 1], th)
    return torch.stack([br, bt], dim=1)


# ── stats de (Br, Bθ) ─────────────────────────────────────────────────────────
def polar_stats(dataset: str) -> dict:
    """mean/std por canal de (Br, Bθ) em y_hw e node_y, sobre todos os chunks do dataset
    (mesmo critério de Normalizer.fit); somas em float64; cache em STATS_DIR."""
    cache = STATS_DIR / f"norm_stats_polar_{dataset.replace('/', '__')}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    paths = sorted(_glob.glob(f'data/torch/data_chunks/{dataset}/data_chunk_*.pt'))
    print(f'  [split] stats de (Br, Btheta) em {len(paths)} chunks (1x, cache {cache})...', flush=True)
    acc = {k: [0, torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)]
           for k in ('y_hw', 'node_y')}
    for i, p in enumerate(paths):
        d = torch.load(p, map_location='cpu', weights_only=False)
        for key, v in (('y_hw', polar_hw(d['y_hw'].double()).permute(1, 0, 2, 3).reshape(2, -1)),
                       ('node_y', polar_nodes(d['node_y'].double(), d['node_x']).T)):
            acc[key][0] += v.shape[1]
            acc[key][1] += v.sum(1)
            acc[key][2] += (v * v).sum(1)
        del d
        if (i + 1) % 20 == 0:
            print(f'    {i + 1}/{len(paths)}', flush=True)
    stats = {}
    for key, (n, s, s2) in acc.items():
        mean = s / n
        std = (s2 / n - mean ** 2).clamp(min=1e-12).sqrt()
        stats[key] = {'mean': mean.tolist(), 'std': std.tolist()}
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(stats))
    return stats


# ── config e patch ────────────────────────────────────────────────────────────
def build_cfg(comp: str, smoke: bool = False):
    frame, k = COMP[comp]
    nn = elem_mae({})
    assert nn.normalize and nn.norm_stats, 'o corte da componente é feito em encode_batch: precisa de normalize=True'
    nn.problem = 'BLDC_138x276_split_smoke' if smoke else f'BLDC_138x276_split_{comp}'
    nn.arch_cfg.grid_out_ch = 1
    st = nn.norm_stats['stats']
    src = st if frame == 'xy' else polar_stats(nn.dataset)
    for key in ('y_hw', 'node_y'):
        st[key] = {m: [src[key][m][k]] for m in ('mean', 'std')}
    nn.norm_stats['split'] = {'comp': comp, 'frame': frame, 'channel': k}   # só registro (from_dict ignora)
    if smoke:
        nn.n_epochs = 1
    return nn


def patch_encode_batch(comp: str):
    frame, k = COMP[comp]

    def encode_batch(self, batch):
        if isinstance(batch, dict):
            batch = dict(batch)
            y_hw, node_y = batch['y_hw'], batch['node_y']
            if frame == 'polar':
                y_hw, node_y = polar_hw(y_hw), polar_nodes(node_y, batch['node_x'])
            batch['y_hw'] = y_hw[:, k:k + 1]
            batch['node_y'] = node_y[:, k:k + 1]
        return _ORIG_ENCODE_BATCH(self, batch)
    Normalizer.encode_batch = encode_batch


class _FirstChunks:
    """Substitui o módulo glob dentro de scripts.train (smoke): só os n primeiros chunks."""
    def __init__(self, n):
        self.n = n

    def glob(self, pattern):
        return sorted(_glob.glob(pattern))[:self.n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('comp', choices=sorted(COMP))
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--stats-only', action='store_true')
    a = ap.parse_args()

    nn = build_cfg(a.comp, a.smoke)
    print(f"[split] {a.comp} {nn.norm_stats['split']} | y_hw {nn.norm_stats['stats']['y_hw']} | "
          f"node_y {nn.norm_stats['stats']['node_y']}", flush=True)
    if a.stats_only:
        return
    import scripts.train as train
    patch_encode_batch(a.comp)
    if a.smoke:
        train.glob = _FirstChunks(4)
    print(f"[split] dataset {nn.dataset} | logs data/logs/{nn.problem}/{nn.arch}/", flush=True)
    train.run(nn)


if __name__ == '__main__':
    main()
