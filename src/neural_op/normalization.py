import json
from pathlib import Path

import torch


# ── Layout de entrada por arch (R1, 2026-10-21) ───────────────────────────────
# Regra única: toda feature de entrada é normalizada por z-score com stats dos chunks de
# TREINO; só ficam crus os canais de posição lidos pela interpolação grade -> pontos
# (archs/interp.py::interpolate_grid_to_nodes, que espera r e θ normalizados em [0,1]).
# Saídas (y_hw/node_y) e arestas (edge_attr/cross_edge_attr) normalizadas por inteiro.
# arch -> {chave: (nomes das colunas, colunas cruas)}. Normalizer.fit falha se o nº de
# colunas do chunk não bater com os nomes (layout BLDC_138x276, com J).
_GRID_X = ('mu_r', 'M', 'J')
_GRID_Y = ('B_r', 'B_theta')
_LAYOUTS = {
    'FNO2d': {
        'x_hw': (_GRID_X, ()),
        'y_hw': (_GRID_Y, ()),
    },
    # node_x lido em fno_gnn.py::_interpolate_fno_to_nodes (colunas 3, 4)
    'FNO_GNN': {
        'x_hw':      (_GRID_X, ()),
        'y_hw':      (_GRID_Y, ()),
        'node_x':    (('mu_r', 'M', 'dual_area', 'r_base', 'c_base', 'J'), (3, 4)),
        'node_y':    (_GRID_Y, ()),
        'edge_attr': (('delta_r', 'delta_c', 'center_dist', 'delta_mu'), ()),
    },
    # node_x lido em _interpolate_fno_to_nodes_v2 (colunas 0, 1); elem_x não é interpolado
    'FNO_BipartiteGNN': {
        'x_hw':            (_GRID_X, ()),
        'y_hw':            (_GRID_Y, ()),
        'node_x':          (('r_base', 'c_base'), (0, 1)),
        'node_y':          (_GRID_Y, ()),
        'elem_x':          (('mu_r', 'M', 'area', 'r_base_c', 'c_base_c', 'J'), ()),
        'edge_attr':       (('delta_r', 'delta_c', 'center_dist'), ()),
        'cross_edge_attr': (('dist_centroid_vertex',), ()),
    },
    # FNO_BipartiteGNN_Elem._fno_at: node_x = elementos (centróide, colunas 0, 1) e
    # elem_x = vértices (colunas 0, 1; interpolado só com aux_fno=True)
    # (contrato do layout Elem: docstring de src/data_gen/parsers/femm_mesh_elem.py)
    'FNO_BipartiteGNN_Elem': {
        'x_hw':            (_GRID_X, ()),
        'y_hw':            (_GRID_Y, ()),
        'node_x':          (('r_base_c', 'c_base_c', 'mu_r', 'M', 'area', 'J'), (0, 1)),
        'node_y':          (_GRID_Y, ()),
        'elem_x':          (('r_base', 'c_base'), (0, 1)),
        'edge_attr':       (('delta_r', 'delta_c', 'center_dist'), ()),
        'cross_edge_attr': (('dist_centroid_vertex',), ()),
    },
}
# GNN_PostBase: mesmo layout do FNO_GNN (node_x lido por _interpolate_fno_to_nodes)
_LAYOUTS['GNN_PostBase'] = _LAYOUTS['FNO_GNN']

# Archs legadas (saem na limpeza, item 5): exclusões antigas mantidas, sem checagem de
# colunas -- Masked*/SingleMat fazem threshold físico sobre mu_r cru (x_hw[:,0]) no forward.
_LEGACY_RAW = {
    'MaskedFNO2d':         {'x_hw': {0}},
    'FNO2d_SingleMat':     {'x_hw': {0}},
    'MaskedFNO_GNN':       {'x_hw': {0}, 'node_x': {0, 2, 3, 4}},
    'FNO_GNN_v2':          {'node_x': {0, 2, 3, 4}},
    'FNO_BipartiteGNN_v3': {'node_x': {0, 1}},
}

# chaves normalizadas por Normalizer.fit (as presentes no chunk)
_NORM_KEYS = ('x_hw', 'y_hw', 'node_x', 'node_y', 'elem_x', 'edge_attr', 'cross_edge_attr')


def _check_layout(arch: str, sample: dict, dataset: str) -> dict:
    """Confere o chunk contra _LAYOUTS[arch] e devolve {chave: colunas cruas}. Falha alto se
    faltar chave do layout ou se o nº de colunas divergir."""
    if arch in _LEGACY_RAW:
        return {k: set(v) for k, v in _LEGACY_RAW[arch].items()}
    if arch not in _LAYOUTS:
        raise KeyError(f"Normalizer: arch '{arch}' sem layout declarado em _LAYOUTS")
    excl = {}
    for key, (names, raw) in _LAYOUTS[arch].items():
        if key not in sample:
            raise ValueError(f"Normalizer: '{dataset}' sem a chave '{key}' do layout de {arch}")
        n = sample[key].shape[1]
        if n != len(names):
            raise ValueError(
                f"Normalizer: '{dataset}' {key} tem {n} coluna(s), layout de {arch} espera "
                f"{len(names)} {names} -- colunas cruas {tuple(raw)} podem apontar para outra feature")
        if raw:
            excl[key] = set(raw)
    return excl


def _fit_stats_from_chunks(chunk_paths, keys):
    """
    Mean/std por canal, para cada chave em `keys`, sobre todos os chunks —
    streaming (Welford/Chan, combinação de blocos) para nunca manter mais de
    um chunk em memória. Um torch.load por chunk (mesmo custo de I/O de uma
    época de treino), feito 1x — resultado cacheado em disco por Normalizer.fit.
    """
    count, mean, M2 = {}, {}, {}
    for p in chunk_paths:
        d = torch.load(p, map_location='cpu', weights_only=False)
        for key in keys:
            if key not in d:
                continue
            t = d[key]
            flat = t.permute(1, 0, 2, 3).reshape(t.shape[1], -1) if t.dim() == 4 else t.T
            n = flat.shape[1]
            if n == 0:
                continue
            b_mean = flat.mean(dim=1)
            b_M2   = flat.var(dim=1, unbiased=False) * n
            if key not in count:
                count[key], mean[key], M2[key] = 0, torch.zeros_like(b_mean), torch.zeros_like(b_mean)
            n_a, mean_a, M2_a = count[key], mean[key], M2[key]
            n_ab  = n_a + n
            delta = b_mean - mean_a
            mean[key]  = mean_a + delta * n / n_ab
            M2[key]    = M2_a + b_M2 + delta ** 2 * n_a * n / n_ab
            count[key] = n_ab
        del d

    stats = {}
    for key in mean:
        var = M2[key] / max(count[key], 1)
        std = var.clamp(min=1e-12).sqrt()
        stats[key] = {'mean': mean[key].tolist(), 'std': std.tolist()}
    return stats


class Normalizer:
    """
    z-score por canal para x_hw/y_hw/node_x/node_y/elem_x/edge_attr/cross_edge_attr. Stats 1x sobre
    os chunks de treino de um dataset (ver fit); colunas cruas (posição lida pela
    interpolação grade -> pontos) por arch em _LAYOUTS (legadas: _LEGACY_RAW).
    encode/decode fazem broadcast por canal, suportando
    tensores [B,C,H,W] (grade) e [S,C] (nós).
    """

    def __init__(self, stats: dict, exclude: dict = None, train_chunks: dict = None):
        self.stats   = stats or {}
        self.exclude = exclude or {}
        # chunks de treino usados nas stats -- {'sha1': hash do nome do cache, 'n', 'cache'};
        # gravado no config.json da run (norm_stats.train_chunks). {} em runs antigas.
        self.train_chunks = train_chunks or {}

    @classmethod
    def fit(cls, dataset: str, arch: str, train_split: float, split_seed=None, test_split=None,
            force_recompute: bool = False) -> 'Normalizer':
        """Stats só dos chunks de TREINO -- mesmo split de build_loaders
        (split_chunk_paths sobre sorted(glob), como scripts/train.py). Cache por split:
        norm_stats_train_v2_<hash dos nomes dos chunks de treino>.json (v2 = com as
        arestas; caches sem versão, sem edge_attr/cross_edge_attr, não são reaproveitados)."""
        import glob
        import hashlib
        from src.neural_op.dataloaders.grid_loader import split_chunk_paths
        chunk_dir = Path(f'data/torch/data_chunks/{dataset}')
        all_paths = sorted(glob.glob(f'data/torch/data_chunks/{dataset}/data_chunk_*.pt'))
        if not all_paths:
            raise FileNotFoundError(f"Normalizer.fit: nenhum chunk encontrado em '{chunk_dir}'.")
        chunk_paths, _ = split_chunk_paths(all_paths, train_split, split_seed, test_split)
        names = sorted(Path(p).name for p in chunk_paths)
        tag = hashlib.sha1('\n'.join(names).encode()).hexdigest()[:10]

        sample    = torch.load(chunk_paths[0], map_location='cpu', weights_only=False)
        exclude = _check_layout(arch, sample, dataset)
        keys    = [k for k in _NORM_KEYS if k in sample]
        del sample

        cache_path = chunk_dir / f'norm_stats_train_v2_{tag}.json'
        if cache_path.exists() and not force_recompute:
            stats = json.loads(cache_path.read_text())
        else:
            print(f"  [Normalizer] calculando mean/std de {len(chunk_paths)} chunk(s) de treino "
                  f"em '{dataset}' (1x, cacheado em {cache_path.name})...", flush=True)
            stats = _fit_stats_from_chunks(chunk_paths, keys)
            cache_path.write_text(json.dumps(stats))
        return cls(stats, exclude=exclude,
                   train_chunks={'sha1': tag, 'n': len(chunk_paths), 'cache': cache_path.name})

    @classmethod
    def from_dict(cls, d: dict) -> 'Normalizer':
        exclude = {k: set(v) for k, v in d.get('exclude', {}).items()}
        return cls(d.get('stats', {}), exclude=exclude, train_chunks=d.get('train_chunks'))

    def to_dict(self) -> dict:
        return {
            'stats':   self.stats,
            'exclude': {k: sorted(v) for k, v in self.exclude.items()},
            'train_chunks': self.train_chunks,
        }

    def _mean_std(self, key, device, dtype):
        if key not in self.stats:
            return None
        mean = torch.tensor(self.stats[key]['mean'], device=device, dtype=dtype)
        std  = torch.tensor(self.stats[key]['std'],  device=device, dtype=dtype)
        for idx in self.exclude.get(key, ()):
            mean[idx] = 0.0
            std[idx]  = 1.0
        return mean, std

    @staticmethod
    def _view(t, v):
        return v.view(1, -1, 1, 1) if t.dim() == 4 else v.view(1, -1)

    def encode(self, t, key):
        ms = self._mean_std(key, t.device, t.dtype)
        if ms is None:
            return t
        mean, std = ms
        return (t - self._view(t, mean)) / self._view(t, std)

    def decode(self, t, key):
        ms = self._mean_std(key, t.device, t.dtype)
        if ms is None:
            return t
        mean, std = ms
        return t * self._view(t, std) + self._view(t, mean)

    def encode_batch(self, batch):
        """Encode de todas as chaves conhecidas de um batch — tuple (x,y) do
        loader 'grid' ou dict dos loaders 'mesh_v1'/'femm_mesh_v2'. Usado pelo CUDAPrefetcher,
        transparente para step_fn (loss passa a ser calculada em espaço
        normalizado sem nenhuma mudança nos step_fn de cada arch)."""
        if isinstance(batch, dict):
            out = dict(batch)
            for key in _NORM_KEYS:
                if key in out:
                    out[key] = self.encode(out[key], key)
            return out
        x, y = batch
        return self.encode(x, 'x_hw'), self.encode(y, 'y_hw')
