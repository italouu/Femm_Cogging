import torch
import numpy as np

from src.neural_op.archs.fno_mat import _make_material_masks


# ── Helpers vetorizados (todo o chunk de uma vez, sem loop por amostra) ──────

def _magnitude(y, pred):
    """y, pred : [B, C, H, W] — magnitude vetorial (C>=2) ou absoluta (C==1)."""
    if y.shape[1] >= 2:
        mag_true = (y[:, 0]**2 + y[:, 1]**2).sqrt()
        mag_pred = (pred[:, 0]**2 + pred[:, 1]**2).sqrt()
    else:
        mag_true = y[:, 0].abs()
        mag_pred = pred[:, 0].abs()
    return mag_true, mag_pred


def _mask_and_ref(mag_true, thr, extra_mask=None):
    """
    mag_true : [B, H, W]
    Retorna mask [B,H,W] bool e B_ref [B] (RMS de |y| na região relevante;
    1.0 quando a amostra não tem região relevante) — mesma definição usada
    em src.neural_op.archs.eval._masked_metrics, só que por amostra do batch.
    """
    mask = mag_true >= thr
    if extra_mask is not None:
        mask = mask & extra_mask
    B = mag_true.shape[0]
    B_ref = torch.ones(B, device=mag_true.device)
    for b in range(B):
        if mask[b].any():
            B_ref[b] = mag_true[b][mask[b]].pow(2).mean().sqrt()
    return mask, B_ref


def _pool_masked_pct(err, mask, B_ref):
    """err, mask : [B,H,W]; B_ref : [B] → 1D array com erro normalizado [%]
    de todos os pixels relevantes de todas as amostras do chunk, já concatenados
    (pool global de pixels, não média por amostra)."""
    pct = err / B_ref.view(-1, 1, 1) * 100
    return pct[mask].cpu().numpy()


def _qtree_render_batched(values, node_x, L, H, W):
    """
    Equivalente batched de src.neural_op.archs.eval._qtree_render: projeta nós
    de TODAS as amostras do chunk em grades H×W via scatter_add, usando L para
    saber a qual amostra cada nó pertence (mesma técnica de _qtree_render, só
    que sem o loop por amostra de fno_gnn_eval_fn/masked_fno_gnn_eval_fn).

    values : [S_tot, C]
    node_x : [S_tot, N]  — col 3=r_base, col 4=c_base
    L      : [B]         nós por amostra
    """
    device = values.device
    B = len(L)
    C = values.shape[1]
    batch_idx = torch.repeat_interleave(torch.arange(B, device=device), L)
    rows = (node_x[:, 3] * H).long().clamp(0, H - 1)
    cols = (node_x[:, 4] * W).long().clamp(0, W - 1)
    flat = batch_idx * (H * W) + rows * W + cols

    grid  = torch.zeros(B * H * W, C, device=device)
    count = torch.zeros(B * H * W, 1, device=device)
    grid.scatter_add_(0, flat.unsqueeze(1).expand_as(values), values)
    count.scatter_add_(0, flat.unsqueeze(1), torch.ones(len(flat), 1, device=device))
    count.clamp_(min=1)
    return (grid / count).view(B, H, W, C).permute(0, 3, 1, 2)


# ── Coleta de erros por arquitetura (1 chunk inteiro → dict[stage] = array%) ─

def collect_fno2d(model, d, thr):
    x, y = d['x_hw'], d['y_hw']
    with torch.no_grad():
        pred = model(x)
    mag_true, mag_pred = _magnitude(y, pred)
    err = (mag_pred - mag_true).abs()
    mask, B_ref = _mask_and_ref(mag_true, thr)
    return {'pred': _pool_masked_pct(err, mask, B_ref)}


def collect_single_mat_fno2d(model, d, thr):
    x, y = d['x_hw'], d['y_hw']
    with torch.no_grad():
        pred  = model(x)
        masks = _make_material_masks(x[:, 0])
    mask_m = masks[:, getattr(model, '_single_mat_id', 0)]
    mag_true, mag_pred = _magnitude(y, pred)
    err = (mag_pred - mag_true).abs()
    mask, B_ref = _mask_and_ref(mag_true, thr, extra_mask=mask_m)
    return {'pred': _pool_masked_pct(err, mask, B_ref)}


def collect_masked_fno2d(model, d, thr):
    x, y = d['x_hw'], d['y_hw']
    with torch.no_grad():
        pred8     = model(x)
        masks     = _make_material_masks(x[:, 0])
        assembled = model.assemble(pred8, masks)
    mag_true, mag_pred = _magnitude(y, assembled)
    err = (mag_pred - mag_true).abs()
    mask, B_ref = _mask_and_ref(mag_true, thr)
    return {'pred': _pool_masked_pct(err, mask, B_ref)}


def collect_fno_gnn(model, d, thr):
    """FNO_GNN, FNO_GNN_Field, GNN_PostBase — mesma assinatura de forward."""
    x_hw, node_x   = d['x_hw'], d['node_x']
    edge_index     = d['edge_index']
    edge_attr      = d['edge_attr']
    L              = d['L']
    y_hw_grid      = d['y_hw_grid'] if 'y_hw_grid' in d else d['y_hw']
    H, W           = int(d['dim'][0]), int(d['dim'][1])

    with torch.no_grad():
        y_hw_fno, y_nodes = model(x_hw, node_x, edge_index, edge_attr, L)

    mag_true, mag_fno = _magnitude(y_hw_grid, y_hw_fno)
    err_fno = (mag_fno - mag_true).abs()

    y_nodes_hw       = _qtree_render_batched(y_nodes, node_x, L, H, W)
    _, mag_gnn       = _magnitude(y_hw_grid, y_nodes_hw)
    err_gnn = (mag_gnn - mag_true).abs()

    mask, B_ref = _mask_and_ref(mag_true, thr)
    return {
        'fno': _pool_masked_pct(err_fno, mask, B_ref),
        'gnn': _pool_masked_pct(err_gnn, mask, B_ref),
    }


def collect_masked_fno_gnn(model, d, thr):
    x_hw, node_x   = d['x_hw'], d['node_x']
    edge_index     = d['edge_index']
    edge_attr      = d['edge_attr']
    L              = d['L']
    y_hw_grid      = d['y_hw_grid'] if 'y_hw_grid' in d else d['y_hw']
    H, W           = int(d['dim'][0]), int(d['dim'][1])

    with torch.no_grad():
        y_hw_8, masks, y_nodes_2 = model(x_hw, node_x, edge_index, edge_attr, L)
        fno_assembled = model.assemble_grid(y_hw_8, masks)

    mag_true, mag_fno = _magnitude(y_hw_grid, fno_assembled)
    err_fno = (mag_fno - mag_true).abs()

    y_nodes_hw   = _qtree_render_batched(y_nodes_2, node_x, L, H, W)
    _, mag_gnn   = _magnitude(y_hw_grid, y_nodes_hw)
    err_gnn = (mag_gnn - mag_true).abs()

    mask, B_ref = _mask_and_ref(mag_true, thr)
    return {
        'fno': _pool_masked_pct(err_fno, mask, B_ref),
        'gnn': _pool_masked_pct(err_gnn, mask, B_ref),
    }


METRICS_REGISTRY = {
    'FNO2d':           collect_fno2d,
    'FNO_ref':         collect_fno2d,
    'FNO2d_SingleMat': collect_single_mat_fno2d,
    'MaskedFNO2d':     collect_masked_fno2d,
    'FNO_GNN':         collect_fno_gnn,
    'FNO_GNN_Field':   collect_fno_gnn,
    'GNN_PostBase':    collect_fno_gnn,
    'MaskedFNO_GNN':   collect_masked_fno_gnn,
}


def aggregate(pooled):
    """pooled : dict[stage] = list[np.ndarray] → dict[stage] = (mean, median, p95, n)."""
    out = {}
    for stage, arrs in pooled.items():
        e = np.concatenate(arrs)
        out[stage] = (float(e.mean()), float(np.median(e)), float(np.percentile(e, 95)), len(e))
    return out
