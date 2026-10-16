"""PROTÓTIPO (2026-10-16) — Bx e By treinados em redes INDEPENDENTES (FNO_BipartiteGNN_Elem, mae).

Não altera src/ nem o dataset: usa o mesmo dataset da bateria bldc
(data/torch/data_chunks/BLDC_138x276/FNO_BipartiteGNN_Elem/) e a mesma config
(src/configs/batteries/bldc.py::elem_mae), com 3 mudanças feitas só em memória:
  - arch_cfg.grid_out_ch = 1 (FNO e GNN saem com 1 canal);
  - norm_stats de y_hw/node_y reduzidas ao canal escolhido (mesmo norm_stats.json, sem recálculo);
  - problem = 'BLDC_138x276_split_<bx|by>' -> logs em data/logs/BLDC_138x276_split_<bx|by>/FNO_BipartiteGNN_Elem/.
Durante a execução, Normalizer.encode_batch (chamado pelo CUDAPrefetcher em todo batch de treino
e teste) é trocado por uma versão que corta y_hw/node_y no canal escolhido antes do z-score.
Loss, step_fn, metric_fn e o treino (scripts.train.run) são os de sempre.

O config.json da run guarda arch_cfg.grid_out_ch = 1 e as stats de 1 canal (+ norm_stats.split_channel),
então src/postprocess/inference.load_model/predict reconstroem o modelo sem mudança (saída [*, 1]);
B = (Bx, By) se monta juntando as duas runs.

Uso (raiz):
    python -m scripts.proto_split_bxby bx                 # treino completo (500 épocas, parada do MonitorCfg)
    python -m scripts.proto_split_bxby by
    python -m scripts.proto_split_bxby bx --smoke         # 4 chunks, 1 época -> data/logs/BLDC_138x276_split_smoke/
"""
import argparse
import glob as _glob

from src.configs.batteries.bldc import elem_mae
from src.neural_op.normalization import Normalizer

CHANNEL = {'bx': 0, 'by': 1}
_ORIG_ENCODE_BATCH = Normalizer.encode_batch


def build_cfg(comp: str, smoke: bool = False):
    k = CHANNEL[comp]
    nn = elem_mae({})
    assert nn.normalize and nn.norm_stats, 'o corte do canal é feito em encode_batch: precisa de normalize=True'
    nn.problem = 'BLDC_138x276_split_smoke' if smoke else f'BLDC_138x276_split_{comp}'
    nn.arch_cfg.grid_out_ch = 1
    st = nn.norm_stats['stats']
    for key in ('y_hw', 'node_y'):
        st[key] = {m: [st[key][m][k]] for m in ('mean', 'std')}
    nn.norm_stats['split_channel'] = k          # só registro (Normalizer.from_dict ignora)
    if smoke:
        nn.n_epochs = 1
    return nn


def patch_encode_batch(k: int):
    def encode_batch(self, batch):
        if isinstance(batch, dict):
            batch = dict(batch)
            batch['y_hw'] = batch['y_hw'][:, k:k + 1]
            batch['node_y'] = batch['node_y'][:, k:k + 1]
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
    ap.add_argument('comp', choices=sorted(CHANNEL))
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    import scripts.train as train
    nn = build_cfg(a.comp, a.smoke)
    patch_encode_batch(CHANNEL[a.comp])
    if a.smoke:
        train.glob = _FirstChunks(4)
    print(f"[split] {a.comp} (canal {CHANNEL[a.comp]}) | dataset {nn.dataset} | logs data/logs/{nn.problem}/{nn.arch}/",
          flush=True)
    train.run(nn)


if __name__ == '__main__':
    main()
