"""
Inferência direto dos chunks do dataset (sem parse do .ans) e campos salvos por amostra.

Cada run usa os chunks do próprio dataset (config.json['dataset']) e o subset do próprio
split.json; a amostra i do chunk c é a i-ésima de samples.samples_with_subset (mesma ordem
de gravação dos chunks). Saída por amostra:

    data/temp/inference/<arch>_<loss>_<run>/<tamanho>_<mtime_ns do best.pth>/<subset>/sample_XXXXXX.npz

com (float32, T, desnormalizados):
    out_hw  [2, H, W]   saída da grade do FNO (todas as archs)
    y       por arch:   FNO2d -> ausente (só grade)
                        FNO_GNN, GNN_PostBase, FNO_BipartiteGNN -> [S, 2] por vértice
                        FNO_BipartiteGNN_Elem -> [M, 2] por elemento
    y_aux   FNO_SplitGNN: [M, 2] por elemento (N_aux)
    y_base  FNO_SplitGNN: [K, 2] por cópia vértice×material (N_base)
Mesmas contas de inference.predict / predict_split (codificação, forward, decodificação).
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.postprocess.inference import DEVICE, _dec, _enc, load_model
from src.postprocess.samples import samples_with_subset

ROOT       = Path(__file__).resolve().parents[2]
CHUNK_ROOT = ROOT / 'data' / 'torch' / 'data_chunks'
OUT_ROOT   = ROOT / 'data' / 'temp' / 'inference'

GRID_ARCHS   = ('FNO2d',)
VERTEX_ARCHS = ('FNO_GNN', 'GNN_PostBase')
BIP_ARCHS    = ('FNO_BipartiteGNN', 'FNO_BipartiteGNN_Elem', 'FNO_SplitGNN')


def pred_dir(run_dir: Path, subset='test'):
    """Diretório das predições de uma run (tag = identidade do best.pth)."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / 'config.json').read_text())
    st = (run_dir / 'checkpoints' / 'best.pth').stat()
    return (OUT_ROOT / f"{cfg['arch']}_{cfg['loss']}_{run_dir.name}"
            / f'{st.st_size}_{st.st_mtime_ns}' / subset)


def load_prediction(run_dir: Path, sample_idx: int, subset='test'):
    """{'out_hw', 'y' | 'y_aux'/'y_base', ...} da amostra; None se não foi inferida."""
    p = pred_dir(run_dir, subset) / f'sample_{sample_idx:06d}.npz'
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def iter_chunk_samples(d, arch):
    """Chunk (dict do torch.load) -> amostras na ordem de gravação, com índices locais
    (mesmos offsets de loaders.Chunk{Vertex,Bipartite}Dataset, sem embaralhar).
    Cada amostra tem as chaves de entrada no formato de 1 amostra (x_hw [1,C,H,W],
    L [1] ...)."""
    B = int(d['x_hw'].shape[0])
    if arch in GRID_ARCHS:
        for i in range(B):
            yield dict(x_hw=d['x_hw'][i:i + 1])
        return
    off = lambda n: torch.cat([torch.zeros(1, dtype=torch.long), d[n].cumsum(0)])
    n_off, e_off = off('L'), off('E_L')
    bip = arch in BIP_ARCHS
    if bip:
        m_off, c_off = off('elem_L'), off('C_L')
    for i in range(B):
        ns, ne = int(n_off[i]), int(n_off[i + 1])
        es, ee = int(e_off[i]), int(e_off[i + 1])
        s = dict(x_hw=d['x_hw'][i:i + 1], node_x=d['node_x'][ns:ne],
                 edge_index=d['edge_index'][:, es:ee] - ns, edge_attr=d['edge_attr'][es:ee],
                 L=d['L'][i:i + 1])
        if bip:
            ms, me = int(m_off[i]), int(m_off[i + 1])
            cs, ce = int(c_off[i]), int(c_off[i + 1])
            cei = d['cross_edge_index'][:, cs:ce].clone()
            cei[0] -= ms
            cei[1] -= ns
            s.update(elem_x=d['elem_x'][ms:me], cross_edge_index=cei,
                     cross_edge_attr=d['cross_edge_attr'][cs:ce], elem_L=d['elem_L'][i:i + 1])
        yield s


@torch.inference_mode()
def predict_chunk_sample(arch, model, nz, s):
    """1 amostra do chunk -> dict de arrays float32 (ver docstring do módulo)."""
    g = lambda k: s[k].to(DEVICE)
    e = lambda k: _enc(nz, s[k], k).to(DEVICE)
    cpu = lambda t: t.float().cpu().numpy()
    if arch == 'FNO2d':
        return dict(out_hw=cpu(_dec(nz, model(e('x_hw')), 'y_hw')[0]))
    if arch in VERTEX_ARCHS:
        out_hw, y = model(e('x_hw'), e('node_x'), g('edge_index'), e('edge_attr'), g('L'))
        return dict(out_hw=cpu(_dec(nz, out_hw, 'y_hw')[0]), y=cpu(_dec(nz, y, 'node_y')))
    if arch == 'FNO_SplitGNN':
        out_hw, y_aux, y_base = model(e('x_hw'), e('node_x'), e('elem_x'), g('edge_index'),
                                      e('edge_attr'), g('cross_edge_index'),
                                      e('cross_edge_attr'), g('L'), g('elem_L'))
        return dict(out_hw=cpu(_dec(nz, out_hw, 'y_hw')[0]),
                    y_aux=cpu(_dec(nz, y_aux, 'node_y')), y_base=cpu(_dec(nz, y_base, 'elem_y')))
    out_hw, y = model(e('x_hw'), e('node_x'), e('elem_x'), g('edge_index'), e('edge_attr'),
                      g('cross_edge_index'), e('cross_edge_attr'), g('L'))
    return dict(out_hw=cpu(_dec(nz, out_hw, 'y_hw')[0]), y=cpu(_dec(nz, y, 'node_y')))


def _save(path, arrays):
    """Gravação atômica (tmp + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def infer_run(run_dir: Path, subset='test', log=print):
    """Inferência de todas as amostras do subset da run a partir dos chunks; retomável
    (amostras já salvas são puladas; chunk todo salvo nem é lido). -> (n_novas, n_total)."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / 'config.json').read_text())
    arch, out = cfg['arch'], pred_dir(run_dir, subset)
    by_chunk = {}
    for name, _, idx, _ in samples_with_subset(run_dir, subset):
        by_chunk.setdefault(name, []).append(idx)
    todo = {n: ids for n, ids in by_chunk.items()
            if not all((out / f'sample_{i:06d}.npz').exists() for i in ids)}
    n_tot = sum(len(v) for v in by_chunk.values())
    if not todo:
        log(f'  {arch}_{cfg["loss"]}: {n_tot} amostras já salvas')
        return 0, n_tot
    model, nz, _, epoch = load_model(run_dir)
    meta = dict(arch=arch, loss=cfg['loss'], run=run_dir.name, epoch=int(epoch), subset=subset)
    n_new = 0
    for k, (name, ids) in enumerate(todo.items(), 1):
        d = torch.load(CHUNK_ROOT / cfg['dataset'] / name, map_location='cpu', weights_only=False)
        assert int(d['x_hw'].shape[0]) == len(ids), f'{name}: {d["x_hw"].shape[0]} != {len(ids)}'
        for idx, s in zip(ids, iter_chunk_samples(d, arch)):
            p = out / f'sample_{idx:06d}.npz'
            if p.exists():
                continue
            _save(p, dict(predict_chunk_sample(arch, model, nz, s), sample_idx=idx, chunk=name,
                          **{f'meta_{a}': b for a, b in meta.items()}))
            n_new += 1
        del d
        log(f'  {arch}_{cfg["loss"]}: chunk {name} ({k}/{len(todo)})')
    del model
    torch.cuda.empty_cache()
    return n_new, n_tot
