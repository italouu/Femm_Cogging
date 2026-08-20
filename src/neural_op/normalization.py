import json
from pathlib import Path

import torch

# Arquiteturas que fazem threshold físico sobre mu_r (x_hw[:,0]) dentro do
# forward (_make_material_masks: mu_r>10 ferro, 1.01<mu_r<10 ima, mu_r<0.9995
# cobre) — para essas, a coluna mu_r de x_hw fica de fora do z-score, senão os
# limiares quebram silenciosamente. Arch sem essa lógica (FNO2d, FNO_GNN,
# FNO_ref/Darcy, ...) normalizam x_hw por inteiro.
_MASKED_ARCHS = {'MaskedFNO2d', 'FNO2d_SingleMat', 'MaskedFNO_GNN'}

# node_x[:, {0,2,3,4}] = mu_r, cell_area/node_dual_area, r_base, c_base — usadas
# cruas por _interpolate_fno_to_nodes (grid_sample espera [0,1] literal),
# reconstrução de geometria (eval.py/bench/metrics.py) e mapas de material nos
# plots. Nunca normalizadas, para qualquer arch de grafo (estrutural ao
# dataset, não ao arch) que usa o layout node_x clássico (qtree/femm_mesh v1).
_NODE_X_STRUCTURAL = {0, 2, 3, 4}

# FNO_BipartiteGNN (mode='femm_mesh_v2', ver src/data_gen/femm_mesh_v2.py):
# node_x = [r_base, c_base] — layout diferente do clássico acima (só posição,
# 2 colunas, ambas usadas cruas por _interpolate_fno_to_nodes_v2). Excluído
# por completo do z-score, não pelos índices fixos de _NODE_X_STRUCTURAL
# (que presumem 5+ colunas e estourariam IndexError aqui).
_NODE_X_FULLY_STRUCTURAL_ARCHS = {'FNO_BipartiteGNN'}

# FNO_BipartiteGNN_v3 (2026-08-20, ver src/neural_op/archs/femm_mesh_v3_gnn.py
# e src/data_gen/parsers/femm_mesh_v3.py; renomeado de FNO_BipartiteGNN_v2
# no mesmo dia, pra alinhar com o nome do parser FEMM_MESH_V3): node_x =
# [r_base, c_base, node_cell_count] — só as colunas 0,1 (posição) são
# estruturais/cruas (usadas por _interpolate_fno_to_nodes_v3 pra
# grid_sample/gather de célula); a coluna 2 (contagem de nós por célula) é
# uma feature real, normalizada normalmente — diferente de FNO_BipartiteGNN,
# onde node_x é 100% estrutural.
_NODE_X_PARTIAL_STRUCTURAL_ARCHS = {'FNO_BipartiteGNN_v3': {0, 1}}


def _exclude_channels(arch: str, has_graph: bool, node_x_ch: int = None) -> dict:
    """Canais que ficam de fora do z-score (identidade: mean=0, std=1)."""
    excl = {}
    if arch in _MASKED_ARCHS:
        excl['x_hw'] = {0}
    if has_graph:
        if arch in _NODE_X_PARTIAL_STRUCTURAL_ARCHS:
            excl['node_x'] = set(_NODE_X_PARTIAL_STRUCTURAL_ARCHS[arch])
        elif arch in _NODE_X_FULLY_STRUCTURAL_ARCHS:
            excl['node_x'] = set(range(node_x_ch)) if node_x_ch else set()
        else:
            excl['node_x'] = set(_NODE_X_STRUCTURAL)
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
    z-score por canal para x_hw/y_hw/node_x/node_y. Stats computadas 1x sobre
    todos os chunks de treino de um dataset (ver fit), com canais estruturais
    (mu_r em archs mascarados, geometria de node_x) excluídos — ver
    _exclude_channels. encode/decode fazem broadcast por canal, suportando
    tensores [B,C,H,W] (grade) e [S,C] (nós).
    """

    def __init__(self, stats: dict, exclude: dict = None):
        self.stats   = stats or {}
        self.exclude = exclude or {}

    @classmethod
    def fit(cls, dataset: str, arch: str, force_recompute: bool = False) -> 'Normalizer':
        chunk_dir   = Path(f'data/torch/data_chunks/{dataset}')
        chunk_paths = sorted(str(p) for p in chunk_dir.glob('data_chunk_*.pt'))
        if not chunk_paths:
            raise FileNotFoundError(f"Normalizer.fit: nenhum chunk encontrado em '{chunk_dir}'.")

        sample    = torch.load(chunk_paths[0], map_location='cpu', weights_only=False)
        has_graph = 'node_x' in sample
        has_elem  = 'elem_x' in sample   # grafo de elementos (mode='femm_mesh_v2')
        node_x_ch = sample['node_x'].shape[1] if has_graph else None
        del sample
        keys = (['x_hw', 'y_hw']
                + (['node_x', 'node_y'] if has_graph else [])
                + (['elem_x'] if has_elem else []))

        cache_path = chunk_dir / 'norm_stats.json'
        if cache_path.exists() and not force_recompute:
            stats = json.loads(cache_path.read_text())
        else:
            print(f"  [Normalizer] calculando mean/std de {len(chunk_paths)} chunk(s) "
                  f"em '{dataset}' (1x, cacheado em {cache_path.name})...", flush=True)
            stats = _fit_stats_from_chunks(chunk_paths, keys)
            cache_path.write_text(json.dumps(stats))
        return cls(stats, exclude=_exclude_channels(arch, has_graph, node_x_ch))

    @classmethod
    def from_dict(cls, d: dict) -> 'Normalizer':
        exclude = {k: set(v) for k, v in d.get('exclude', {}).items()}
        return cls(d.get('stats', {}), exclude=exclude)

    def to_dict(self) -> dict:
        return {
            'stats':   self.stats,
            'exclude': {k: sorted(v) for k, v in self.exclude.items()},
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
        loader 'grid' ou dict do loader 'qtree'. Usado pelo CUDAPrefetcher,
        transparente para step_fn (loss passa a ser calculada em espaço
        normalizado sem nenhuma mudança nos step_fn de cada arch)."""
        if isinstance(batch, dict):
            out = dict(batch)
            for key in ('x_hw', 'y_hw', 'node_x', 'node_y', 'elem_x'):
                if key in out:
                    out[key] = self.encode(out[key], key)
            return out
        x, y = batch
        return self.encode(x, 'x_hw'), self.encode(y, 'y_hw')
