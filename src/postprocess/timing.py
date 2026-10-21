"""
Custo computacional: FEMM × pré-processamento + inferência (P8, 2026-10-20).

Método de referência: scripts/pos_bateria_timing.py (git show d912dc3^:...) -- aquecimento,
repetições com torch.cuda.synchronize, limite de memória da GPU no Windows (sem limite o
driver WDDM transborda para a RAM em vez de dar OOM). Mudanças: gabarito/entradas dos
parsers SMOOTH (com o canal J), as 5 archs (inclui a Elem) e só o mi_analyze no cronômetro
do FEMM (a malha é dada para os dois lados: geometria e malha ficam fora da comparação).

  femm_solve_timed    — redesenha (fase do rotor, amplitude e ângulo da corrente do CSV), gera
                        a malha (fora do cronômetro) e cronometra só o mi_analyze.
  preprocess_timed    — .ans resolvido -> entradas de cada arch, por etapa, com as MESMAS
                        funções dos parsers smooth (femm_mesh_v2 / femm_mesh_smooth /
                        femm_mesh_elem); check_preprocess confere igualdade com os parsers.
  encode_timed        — numpy -> GPU + z-score (inference._enc), por arch.
  time_inference      — best.pth, torch.inference_mode, lote 1 e potências de 2 até OOM;
                        forward + desnormalização (+ grade -> nós no FNO2d).
"""
import gzip
import os
import platform
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from src.data_gen.parsers.ans_parsing import (
    _MAGNET_ID, _MU_BY_ID, _build_bidirectional_edge_attrs, _build_edges, _element_areas,
    _grid_polar_xy, _node_current_density, _node_magnet_polarity, _node_material_priority,
    _node_material_stats, _parse_label_currents, _parse_label_materials, _parse_solution,
    _wrap_edge_pairs,
)
from src.data_gen.parsers.femm_mesh_elem import _build_dual_edges
from src.data_gen.parsers.femm_mesh_v2 import _build_trifinder, _grid_const_per_element
from src.neural_op.archs.interp import interpolate_grid_to_nodes
from src.postprocess.inference import DEVICE, _dec, _enc

# etapas do pré-processamento usadas por cada arch (a soma é o custo da arch)
ARCH_STAGES = {
    'FNO2d':                 ('read_ans', 'materials', 'x_hw', 'node_pos'),
    'FNO_GNN':               ('read_ans', 'materials', 'x_hw', 'node_pos', 'graph_topology', 'v1_graph'),
    'GNN_PostBase':          ('read_ans', 'materials', 'x_hw', 'node_pos', 'graph_topology', 'v1_graph'),
    'FNO_BipartiteGNN':      ('read_ans', 'materials', 'x_hw', 'node_pos', 'graph_topology', 'bip_graph'),
    'FNO_BipartiteGNN_Elem': ('read_ans', 'materials', 'x_hw', 'node_pos', 'graph_topology', 'bip_graph',
                              'elem_graph'),
}
LAYOUT = {'FNO2d': 'v1', 'FNO_GNN': 'v1', 'GNN_PostBase': 'v1', 'FNO_BipartiteGNN': 'bip',
          'FNO_BipartiteGNN_Elem': 'elem'}


def _now():
    return time.perf_counter()


def iqr_stats(x):
    x = np.asarray(x, dtype=np.float64)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return dict(median=float(med), q1=float(q1), q3=float(q3), iqr=float(q3 - q1),
                mean=float(x.mean()), n=int(x.size))


# --------------------------------------------------------------------------- #
# hardware / software (origem: pos_bateria_timing.hardware_info)
# --------------------------------------------------------------------------- #
def _femm_exe_version():
    try:
        import winreg
        import win32api
        clsid = winreg.QueryValue(winreg.HKEY_CLASSES_ROOT, r'femm.ActiveFEMM\CLSID')
        try:
            exe = winreg.QueryValue(winreg.HKEY_CLASSES_ROOT, rf'CLSID\{clsid}\LocalServer32')
        except OSError:     # FEMM é 32 bits -- registrado em WOW6432Node
            exe = winreg.QueryValue(winreg.HKEY_CLASSES_ROOT, rf'WOW6432Node\CLSID\{clsid}\LocalServer32')
        exe = exe.strip('"').split('"')[0].split(' /')[0]
        info = win32api.GetFileVersionInfo(exe, '\\')
        ms, ls = info['FileVersionMS'], info['FileVersionLS']
        return dict(exe=exe, file_version=f'{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}',
                    mtime=time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(exe))))
    except Exception as e:      # noqa: BLE001
        return dict(error=repr(e))


def _ps(cmd):
    try:
        return subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True,
                              text=True, timeout=30).stdout.strip() or None
    except Exception:           # noqa: BLE001
        return None


def hardware_info():
    def ver(p):
        try:
            return metadata.version(p)
        except metadata.PackageNotFoundError:
            return None
    try:
        drv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:           # noqa: BLE001
        drv = None
    return dict(
        os=platform.platform(), cpu=_ps('(Get-CimInstance Win32_Processor).Name') or platform.processor(),
        cpu_logical_cores=os.cpu_count(),
        ram_gib=_ps('[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)'),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        gpu_mem_gib=(round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)
                     if torch.cuda.is_available() else None),
        nvidia_driver=drv, python=platform.python_version(), torch=torch.__version__,
        cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        numpy=np.__version__, scipy=ver('scipy'), shapely=ver('shapely'), pyfemm=ver('pyfemm'),
        femm=_femm_exe_version())


# --------------------------------------------------------------------------- #
# FEMM: só o mi_analyze no cronômetro
# --------------------------------------------------------------------------- #
def femm_solve_timed(row, work_dir: Path):
    """Mesma sequência de femm_mesh.save_ans_gzip_sample; cronometra openfemm, geometria e
    malha à parte (fora da comparação) e o mi_analyze (solução). Retorna (tempos s, .ans)."""
    import femm
    from src.data_gen.motor_constants import stack_length_mm
    from src.data_gen.motor_model import BLDC_FEMM_Model_Sym120_Annular
    from src.postprocess.torque import params_of
    work_dir = Path(work_dir).resolve()
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    p = params_of(row)
    tm = {}
    cwd0 = os.getcwd()
    os.chdir(work_dir)
    try:
        model = BLDC_FEMM_Model_Sym120_Annular(motor_params=p, phase=p['rotor_phase']['value'])
        s = _now(); femm.openfemm('bHide'); femm.main_resize(1000, 1000); tm['openfemm'] = _now() - s
        s = _now()
        femm.newdocument(0)
        femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, stack_length_mm(p), 200)
        model.draw_motor(); femm.mi_zoomnatural(); femm.mi_saveas(str(work_dir / 'model.fem'))
        tm['geometry'] = _now() - s
        s = _now(); femm.mi_createmesh(); tm['mesh'] = _now() - s
        s = _now(); femm.mi_analyze(); tm['solve'] = _now() - s
        s = _now(); femm.closefemm(); tm['closefemm'] = _now() - s
    finally:
        os.chdir(cwd0)
    return tm, work_dir / 'model.ans'


def check_mesh_vs_raw(nodes, elems, raw_gz):
    """Malha (coordenadas, conectividade, rótulos) e A do .ans novo × o do raw."""
    _, rn, re_ = _parse_solution_gz(raw_gz)
    same = rn.shape == nodes.shape and re_.shape == elems.shape
    out = dict(mesh_equal_raw=bool(same and np.array_equal(rn[:, :2], nodes[:, :2])
                                   and np.array_equal(re_, elems)),
               n_nodes=int(nodes.shape[0]), n_elems=int(elems.shape[0]))
    out['max_abs_dA_vs_raw'] = float(np.abs(rn[:, 2] - nodes[:, 2]).max()) if same else None
    return out


def _parse_solution_gz(gz):
    tmp = Path(str(gz) + '.tmp_timing.ans')
    with gzip.open(gz, 'rb') as fi, open(tmp, 'wb') as fo:
        shutil.copyfileobj(fi, fo)
    try:
        return _parse_solution(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# pré-processamento por etapa (funções dos parsers smooth; só as entradas)
# --------------------------------------------------------------------------- #
def preprocess_timed(ans_path, r_in, r_ext, ang_1=0.0, ang_2=120.0, n_r=138, n_a=276):
    """.ans -> entradas de cada layout (v1 / bip / elem), cronometrado por etapa.
    Retorna (tempos s {etapa}, arrays {layout: {chave: array}}, (nodes, elems))."""
    t = {}
    a1, a2 = np.deg2rad(ang_1), np.deg2rad(ang_2)

    s = _now()
    lines, nodes, elems = _parse_solution(str(ans_path))
    t['read_ans'] = _now() - s

    s = _now()
    label_material_id, label_mu, label_M = _parse_label_materials(lines)
    elem_material_id = label_material_id[elems[:, 3]]
    elem_mu, elem_M = label_mu[elems[:, 3]], label_M[elems[:, 3]]
    area = _element_areas(nodes, elems)
    elem_J = _parse_label_currents(lines, elems, area)[elems[:, 3]]
    t['materials'] = _now() - s

    # x_hw (femm_mesh_v2.parse_ans_gzip_sample)
    s = _now()
    centroids = nodes[elems[:, :3], :2].mean(axis=1)
    Xg, Yg = _grid_polar_xy(r_in, r_ext, a1, a2, n_r, n_a)
    _, trifinder = _build_trifinder(nodes, elems)
    x_hw = np.stack([_grid_const_per_element(trifinder, f, centroids, Xg, Yg).reshape(n_r, n_a)
                     for f in (elem_mu, elem_M, elem_J)], axis=0).astype(np.float32)
    t['x_hw'] = _now() - s

    s = _now()
    r_node = np.hypot(nodes[:, 0], nodes[:, 1])
    th_node = np.arctan2(nodes[:, 1], nodes[:, 0])
    r_base = (r_node - r_in) / (r_ext - r_in)
    c_base = (th_node - a1) / (a2 - a1)
    t['node_pos'] = _now() - s

    s = _now()
    edges_undirected = _build_edges(elems)
    wrap_1, wrap_2 = _wrap_edge_pairs(nodes, ang_1, ang_2)
    t['graph_topology'] = _now() - s

    # v1 (FNO_GNN / GNN_PostBase; femm_mesh_smooth): material por prioridade
    s = _now()
    node_material_id = _node_material_priority(elems, elem_material_id, nodes.shape[0])
    _, _frac, node_dual_area = _node_material_stats(nodes, elems, area, elem_material_id)
    node_mu = _MU_BY_ID[node_material_id]
    node_M = np.where(node_material_id == _MAGNET_ID,
                      _node_magnet_polarity(nodes, elems, area, elem_material_id, elem_M),
                      0.0).astype(np.float32)
    node_J = _node_current_density(elems, area, elem_J.astype(np.float32), elem_material_id,
                                   node_material_id)
    ei_v1, ea_v1 = _build_bidirectional_edge_attrs(nodes, edges_undirected, wrap_1, wrap_2,
                                                   r_base, c_base, node_mu, n_r, n_a)
    nx_v1 = np.stack([node_mu.astype(np.float32), node_M, node_dual_area.astype(np.float32),
                      r_base.astype(np.float32), c_base.astype(np.float32), node_J], axis=1)
    t['v1_graph'] = _now() - s

    # bipartite (femm_mesh_v2.parse_ans_gzip_sample)
    s = _now()
    rb32, cb32 = r_base.astype(np.float32), c_base.astype(np.float32)
    i = np.concatenate([edges_undirected[:, 0], wrap_1])
    j = np.concatenate([edges_undirected[:, 1], wrap_2])
    n_wrap = len(wrap_1)
    cd = np.hypot(nodes[j, 0] - nodes[i, 0], nodes[j, 1] - nodes[i, 1]).astype(np.float32)
    dr = ((rb32[j] - rb32[i]) * n_r).astype(np.float32)
    dc = ((cb32[j] - cb32[i]) * n_a).astype(np.float32)
    if n_wrap:
        cd[-n_wrap:] = 0.0; dr[-n_wrap:] = 0.0; dc[-n_wrap:] = 0.0
    ea_bp = np.concatenate([np.stack([dr, dc, cd], 1), np.stack([-dr, -dc, cd], 1)]).astype(np.float32)
    ei_bp = np.stack([np.concatenate([i, j]), np.concatenate([j, i])]).astype(np.int64)
    nx_bp = np.stack([rb32, cb32], axis=1)
    n_el = elems.shape[0]
    r_e = np.hypot(centroids[:, 0], centroids[:, 1])
    th_e = np.arctan2(centroids[:, 1], centroids[:, 0])
    elem_x = np.stack([elem_mu, elem_M, area.astype(np.float32),
                       ((r_e - r_in) / (r_ext - r_in)).astype(np.float32),
                       ((th_e - a1) / (a2 - a1)).astype(np.float32), elem_J], axis=1).astype(np.float32)
    tri_idx = np.repeat(np.arange(n_el, dtype=np.int64), 3)
    vtx_idx = elems[:, :3].reshape(-1).astype(np.int64)
    cea = np.linalg.norm(nodes[vtx_idx, :2] - centroids[tri_idx], axis=1).astype(np.float32)[:, None]
    cei = np.stack([tri_idx, vtx_idx], axis=0)
    t['bip_graph'] = _now() - s

    # Elem (femm_mesh_elem.parse_ans_gzip_sample_elem): elementos = grafo principal
    s = _now()
    nx_el = elem_x[:, [3, 4, 0, 1, 2, 5]].astype(np.float32)
    rc, cc = elem_x[:, 3].astype(np.float64), elem_x[:, 4].astype(np.float64)
    i_int, j_int, i_wrap, j_wrap = _build_dual_edges(nodes, elems, ang_1, ang_2)
    dx = centroids[j_int] - centroids[i_int]
    phi = -np.deg2rad(ang_2 - ang_1)
    cj = centroids[j_wrap]
    cj_rot = np.stack([cj[:, 0] * np.cos(phi) - cj[:, 1] * np.sin(phi),
                       cj[:, 0] * np.sin(phi) + cj[:, 1] * np.cos(phi)], axis=1)
    dw = cj_rot - centroids[i_wrap]
    ie, je = np.concatenate([i_int, i_wrap]), np.concatenate([j_int, j_wrap])
    d_r = np.concatenate([(rc[j_int] - rc[i_int]) * n_r, (rc[j_wrap] - rc[i_wrap]) * n_r]).astype(np.float32)
    d_c = np.concatenate([(cc[j_int] - cc[i_int]) * n_a,
                          (cc[j_wrap] - 1.0 - cc[i_wrap]) * n_a]).astype(np.float32)
    dist = np.concatenate([np.hypot(dx[:, 0], dx[:, 1]), np.hypot(dw[:, 0], dw[:, 1])]).astype(np.float32)
    ea_el = np.concatenate([np.stack([d_r, d_c, dist], 1), np.stack([-d_r, -d_c, dist], 1)]).astype(np.float32)
    ei_el = np.stack([np.concatenate([ie, je]), np.concatenate([je, ie])]).astype(np.int64)
    cei_el = cei[[1, 0]].copy()
    t['elem_graph'] = _now() - s

    L_n = np.array([nodes.shape[0]], np.int64)
    arrays = dict(
        v1=dict(x_hw=x_hw, node_x=nx_v1, edge_index=ei_v1, edge_attr=ea_v1, L=L_n),
        bip=dict(x_hw=x_hw, node_x=nx_bp, elem_x=elem_x, edge_index=ei_bp, edge_attr=ea_bp,
                 cross_edge_index=cei, cross_edge_attr=cea, L=L_n),
        elem=dict(x_hw=x_hw, node_x=nx_el, elem_x=nx_bp, edge_index=ei_el, edge_attr=ea_el,
                  cross_edge_index=cei_el, cross_edge_attr=cea, L=np.array([n_el], np.int64)))
    return t, arrays, (nodes, elems)


def check_preprocess(arrays, ref):
    """Igualdade exata das entradas × parsers smooth (ref = samples.parse_smooth(...,
    with_elem=True) do mesmo .ans) -> {layout.chave: bool}."""
    out = {}
    for lay, keys in (('v1', ('x_hw', 'node_x', 'edge_index', 'edge_attr', 'L')),
                      ('bip', ('x_hw', 'node_x', 'elem_x', 'edge_index', 'edge_attr',
                               'cross_edge_index', 'cross_edge_attr', 'L')),
                      ('elem', ('x_hw', 'node_x', 'elem_x', 'edge_index', 'edge_attr',
                                'cross_edge_index', 'cross_edge_attr', 'L'))):
        for k in keys:
            a, b = np.asarray(arrays[lay][k]), np.asarray(ref[lay][k])
            out[f'{lay}.{k}'] = bool(a.shape == b.shape and np.array_equal(a, b))
    return out


def encode_timed(runs, arrays):
    """numpy -> GPU + z-score (inference._enc), por arch (1ª run de cada arch). {arch: s}."""
    out, first = {}, {}
    for r in runs.values():
        first.setdefault(r['arch'], r)
    for arch, r in first.items():
        d, nrm = arrays[LAYOUT[arch]], r['normalizer']
        g = lambda k: torch.from_numpy(np.ascontiguousarray(d[k]))
        torch.cuda.synchronize()
        s = _now()
        _enc(nrm, g('x_hw')[None], 'x_hw').to(DEVICE)
        if arch == 'FNO2d':
            g('node_x').to(DEVICE)                   # r/c dos nós para a interpolação
        else:
            _enc(nrm, g('node_x'), 'node_x').to(DEVICE)
            for k in ('edge_index', 'L'):
                g(k).to(DEVICE)
            _enc(nrm, g('edge_attr'), 'edge_attr').to(DEVICE)
            if LAYOUT[arch] != 'v1':
                _enc(nrm, g('elem_x'), 'elem_x').to(DEVICE)
                g('cross_edge_index').to(DEVICE)
                _enc(nrm, g('cross_edge_attr'), 'cross_edge_attr').to(DEVICE)
        torch.cuda.synchronize()
        out[arch] = _now() - s
    return out


# --------------------------------------------------------------------------- #
# inferência (origem: pos_bateria_timing._collate/_prepare_inputs/_forward/_time_batch)
# --------------------------------------------------------------------------- #
def collate(samples, layout):
    """Concatena amostras (dicts de inference.to_torch_sample()[layout]) com offsets -- a
    regra do collate dos loaders (nós/arestas; + elem_x no bip/elem: linha 0 das arestas
    cruzadas = auxiliar (elem_x), linha 1 = principal (node_x) nos dois layouts)."""
    out = dict(x_hw=torch.cat([s['x_hw'] for s in samples], 0),
               L=torch.cat([s['L'] for s in samples]),
               node_x=torch.cat([s['node_x'] for s in samples], 0),
               edge_attr=torch.cat([s['edge_attr'] for s in samples], 0))
    eis, ceis, cas, exs = [], [], [], []
    no = eo = 0
    for s in samples:
        eis.append(s['edge_index'] + no)
        if layout != 'v1':
            c = s['cross_edge_index'].clone()
            c[0] += eo
            c[1] += no
            ceis.append(c); cas.append(s['cross_edge_attr']); exs.append(s['elem_x'])
            eo += s['elem_x'].shape[0]
        no += s['node_x'].shape[0]
    out['edge_index'] = torch.cat(eis, 1)
    if layout != 'v1':
        out.update(cross_edge_index=torch.cat(ceis, 1), cross_edge_attr=torch.cat(cas, 0),
                   elem_x=torch.cat(exs, 0))
    return out


def _prepare(arch, nrm, batch):
    """Entrada já codificada e no device (pré-processamento fora da medição)."""
    g = {k: v.to(DEVICE) for k, v in batch.items()}
    x = _enc(nrm, g['x_hw'], 'x_hw')
    if arch == 'FNO2d':
        return dict(x=x, r=g['node_x'][:, 3], c=g['node_x'][:, 4], L=g['L'])
    out = dict(x=x, nx=_enc(nrm, g['node_x'], 'node_x'), ei=g['edge_index'],
               ea=_enc(nrm, g['edge_attr'], 'edge_attr'), L=g['L'])
    if LAYOUT[arch] != 'v1':
        out.update(ex=_enc(nrm, g['elem_x'], 'elem_x'), ci=g['cross_edge_index'],
                   ca=_enc(nrm, g['cross_edge_attr'], 'cross_edge_attr'))
    return out


def _forward(arch, model, nrm, inp):
    """Forward + desnormalização (+ grade -> nós no FNO2d, como inference.predict)."""
    if arch == 'FNO2d':
        out_hw = _dec(nrm, model(inp['x']), 'y_hw')
        return interpolate_grid_to_nodes(out_hw, inp['r'], inp['c'], inp['L'])
    if LAYOUT[arch] == 'v1':
        _, yn = model(inp['x'], inp['nx'], inp['ei'], inp['ea'], inp['L'])
    else:
        _, yn = model(inp['x'], inp['nx'], inp['ex'], inp['ei'], inp['ea'], inp['ci'], inp['ca'], inp['L'])
    return _dec(nrm, yn, 'node_y')


@torch.inference_mode()
def time_batch(arch, run, batch, n_warmup, n_reps):
    model, nrm = run['model'], run['normalizer']
    inp = _prepare(arch, nrm, batch)
    for _ in range(n_warmup):
        _forward(arch, model, nrm, inp)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    ts = []
    for _ in range(n_reps):
        torch.cuda.synchronize()
        s = _now()
        _forward(arch, model, nrm, inp)
        torch.cuda.synchronize()
        ts.append(_now() - s)
    peak = torch.cuda.max_memory_allocated()
    del inp
    return dict(ms=iqr_stats(1e3 * np.array(ts)), peak_mem_gib=peak / 2 ** 30,
                peak_mem_above_resident_gib=(peak - base) / 2 ** 30)
