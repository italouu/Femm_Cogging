"""
pos_bateria_timing.py
---------------------
Tarefa 3 (T7) do pedido pós-bateria: custo computacional de inferência,
FEMM e modelos no MESMO hardware (máquina Windows com FEMM).

  python -m scripts.pos_bateria_timing femm    # 3a + 3b (50 amostras de teste)
  python -m scripts.pos_bateria_timing infer   # 3c (8 modelos best.pth)

3a -- FEMM, por amostra, mesmas configurações de
      src/data_gen/femm_mesh.py::save_ans_gzip_sample (Sym120_Annular, phase=0,
      mi_probdef(0,'millimeters','planar',1e-8,0,200)): geometria (newdocument +
      probdef + draw_motor [desenho + materiais] + zoomnatural + saveas), malha
      (mi_createmesh), solução (mi_analyze), pós (leitura do .ans + curl(A) P1 +
      média simples nos nós). openfemm/closefemm medidos à parte (como na geração,
      1 sessão FEMM por amostra).
3b -- pré-processamento das entradas, por arquitetura, a partir do .ans recém
      resolvido -- mesmas funções de femm_mesh_unified/femm_mesh_v2, cronometradas
      por etapa; saída conferida contra parse_ans_gzip_sample_unified (igualdade
      exata) em todas as amostras.
3c -- inferência na GPU, torch.inference_mode(), batch 1 e batch máximo
      (potências de 2 até OOM), 10 aquecimentos + 100 repetições com
      torch.cuda.synchronize(); inclui desnormalização da saída e, no FNO2d, a
      interpolação grade->nós. Pico de memória com reset por medição.

Saída: pos_bateria/timing.json
"""
import gzip
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from scripts.pos_bateria_common import (
    DEVICE, OUT_DIR, design_rows, test_samples, load_runs, parse_worker, to_torch_sample,
)
from src.data_gen.parsers.ans_parsing import (
    _parse_solution, _parse_block_materials, _block_magnet_polarity, _build_edges,
    _parse_label_materials,
    _element_areas, _node_material_stats, _node_magnet_polarity, _wrap_edge_pairs,
    _build_bidirectional_edge_attrs, _element_b_from_A, _node_mean_of_elements,
    _grid_polar_xy, _MU_BY_ID,
)
from src.data_gen.parsers.femm_mesh_v2 import _build_trifinder, _grid_const_per_element
from src.data_gen.parsers.femm_mesh_unified import parse_ans_gzip_sample_unified
from src.data_gen.motor_constants import N_POLES_SECTOR
from src.neural_op.archs.interp import interpolate_grid_to_nodes

N_FEMM     = int(os.environ.get("POS_N_FEMM", 50))
N_WARMUP   = 10
N_REPS     = 100
MEM_FRACTION = 0.95      # fração da VRAM dedicada liberada ao alocador (ver run_infer)
OUT_JSON   = OUT_DIR / 'timing.json'
FEMM_WORK  = Path('data/temp/pos_bateria_femm').resolve()
N_R, N_A   = 138, 276
ANG1, ANG2 = 0.0, 120.0


def _now():
    return time.perf_counter()


def _merge_json(key, value):
    d = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    d[key] = value
    OUT_JSON.write_text(json.dumps(d, indent=2))


def _iqr(x):
    x = np.asarray(x, dtype=np.float64)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return dict(median=float(med), q1=float(q1), q3=float(q3), iqr=float(q3 - q1),
                mean=float(x.mean()), n=int(x.size))


def motor_params_from_row(row):
    """valid_designs.csv -> dict {chave: {'unit','value'}} (inverso de
    BLDC_Process.export_params)."""
    out = {}
    for col, v in row.items():
        if ' [' in col:
            key, unit = col.split(' [')
            unit = unit.rstrip(']')
        else:
            key, unit = col, ''
        out[key] = {'unit': unit, 'value': float(v)}
    return out


# --------------------------------------------------------------------------- #
# Hardware
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
        ver = f'{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}'
        return dict(exe=exe, file_version=ver,
                    mtime=time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(exe))))
    except Exception as e:      # noqa: BLE001
        return dict(error=repr(e))


def hardware_info():
    cpu = platform.processor()
    try:
        cpu = subprocess.run(['powershell', '-NoProfile', '-Command',
                              '(Get-CimInstance Win32_Processor).Name'],
                             capture_output=True, text=True, timeout=30).stdout.strip() or cpu
    except Exception:           # noqa: BLE001
        pass
    try:
        ram = subprocess.run(['powershell', '-NoProfile', '-Command',
                              '[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:           # noqa: BLE001
        ram = None
    try:
        drv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:           # noqa: BLE001
        drv = None

    def ver(p):
        try:
            return metadata.version(p)
        except metadata.PackageNotFoundError:
            return None
    return dict(
        os=platform.platform(), cpu=cpu, cpu_logical_cores=os.cpu_count(), ram_gib=ram,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        gpu_mem_gib=(round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)
                     if torch.cuda.is_available() else None),
        nvidia_driver=drv, python=platform.python_version(), torch=torch.__version__,
        cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        numpy=np.__version__, scipy=ver('scipy'), matplotlib=ver('matplotlib'),
        pyfemm=ver('pyfemm'), femm=_femm_exe_version(),
    )


# --------------------------------------------------------------------------- #
# 3b -- pré-processamento cronometrado por etapa
# --------------------------------------------------------------------------- #
def preprocess_timed(ans_path, r_in, r_ext):
    """Reproduz as entradas de femm_mesh_unified (v1) e femm_mesh_v2 (bipartite)
    a partir do .ans, cronometrando cada etapa. Retorna (tempos_s, arrays)."""
    t = {}
    a1, a2 = np.deg2rad(ANG1), np.deg2rad(ANG2)

    s = _now()
    lines, nodes, elems = _parse_solution(str(ans_path))
    t['read_ans'] = _now() - s

    s = _now()
    # [REMOVIDO 2026-10-06] leitura antiga (quebra com polo partido — 15 blocos de ímã)
    # block_material_id, block_mu = _parse_block_materials(lines)
    # block_M = _block_magnet_polarity(block_material_id, N_POLES_SECTOR)
    # elem_material_id = block_material_id[elems[:, 3]]
    # elem_mu = block_mu[elems[:, 3]]
    # elem_M = block_M[elems[:, 3]]
    label_material_id, label_mu, label_M = _parse_label_materials(lines)
    elem_material_id = label_material_id[elems[:, 3]]
    elem_mu = label_mu[elems[:, 3]]
    elem_M = label_M[elems[:, 3]]
    area = _element_areas(nodes, elems)
    t['materials'] = _now() - s

    # x_hw: centros de pixel localizados nos triângulos + cópia de mu_r/M
    s = _now()
    centroids = nodes[elems[:, :3], :2].mean(axis=1)
    Xg, Yg = _grid_polar_xy(r_in, r_ext, a1, a2, N_R, N_A)
    tri, trifinder = _build_trifinder(nodes, elems)
    Mu_hw = _grid_const_per_element(trifinder, elem_mu, centroids, Xg, Yg).reshape(N_R, N_A)
    M_hw = _grid_const_per_element(trifinder, elem_M, centroids, Xg, Yg).reshape(N_R, N_A)
    x_hw = np.stack([Mu_hw, M_hw], axis=0)
    t['x_hw'] = _now() - s

    # posições normalizadas dos nós (todas as archs -- FNO2d usa p/ interpolar nos nós)
    s = _now()
    r_node = np.hypot(nodes[:, 0], nodes[:, 1])
    th_node = np.arctan2(nodes[:, 1], nodes[:, 0])
    r_base = (r_node - r_in) / (r_ext - r_in)
    c_base = (th_node - a1) / (a2 - a1)
    t['node_pos'] = _now() - s

    # arestas da malha + wrap (comum aos grafos)
    s = _now()
    edges_undirected = _build_edges(elems)
    wrap_1, wrap_2 = _wrap_edge_pairs(nodes, ANG1, ANG2)
    t['graph_topology'] = _now() - s

    # v1 (FNO_GNN / GNN_PostBase): material votado por área + edge_attr [E,4]
    s = _now()
    node_material_id, _frac, node_dual_area = _node_material_stats(nodes, elems, area, elem_material_id)
    node_mu = _MU_BY_ID[node_material_id]
    node_M = _node_magnet_polarity(nodes, elems, area, elem_material_id, elem_M)
    ei_v1, ea_v1 = _build_bidirectional_edge_attrs(nodes, edges_undirected, wrap_1, wrap_2,
                                                    r_base, c_base, node_mu, N_R, N_A)
    nx_v1 = np.stack([node_mu.astype(np.float32), node_M.astype(np.float32),
                      node_dual_area.astype(np.float32), r_base.astype(np.float32),
                      c_base.astype(np.float32)], axis=1)
    t['v1_graph'] = _now() - s

    # bipartite: node_x [r,c], edge_attr [E,3], elem_x [M,5], arestas cruzadas
    s = _now()
    rb32, cb32 = r_base.astype(np.float32), c_base.astype(np.float32)
    i = np.concatenate([edges_undirected[:, 0], wrap_1])
    j = np.concatenate([edges_undirected[:, 1], wrap_2])
    n_wrap = len(wrap_1)
    dx = nodes[j, 0] - nodes[i, 0]
    dy = nodes[j, 1] - nodes[i, 1]
    cd = np.hypot(dx, dy).astype(np.float32)
    dr = ((rb32[j] - rb32[i]) * N_R).astype(np.float32)
    dc = ((cb32[j] - cb32[i]) * N_A).astype(np.float32)
    if n_wrap:
        cd[-n_wrap:] = 0.0
        dr[-n_wrap:] = 0.0
        dc[-n_wrap:] = 0.0
    ea_v2 = np.concatenate([np.stack([dr, dc, cd], 1), np.stack([-dr, -dc, cd], 1)]).astype(np.float32)
    ei_v2 = np.stack([np.concatenate([i, j]), np.concatenate([j, i])]).astype(np.int64)
    nx_v2 = np.stack([rb32, cb32], axis=1)
    n_el = elems.shape[0]
    r_e = np.hypot(centroids[:, 0], centroids[:, 1])
    th_e = np.arctan2(centroids[:, 1], centroids[:, 0])
    elem_x = np.stack([elem_mu, elem_M, area.astype(np.float32),
                       ((r_e - r_in) / (r_ext - r_in)).astype(np.float32),
                       ((th_e - a1) / (a2 - a1)).astype(np.float32)], axis=1).astype(np.float32)
    tri_idx = np.repeat(np.arange(n_el, dtype=np.int64), 3)
    vtx_idx = elems[:, :3].reshape(-1).astype(np.int64)
    cea = np.linalg.norm(nodes[vtx_idx, :2] - centroids[tri_idx], axis=1).astype(np.float32)[:, None]
    cei = np.stack([tri_idx, vtx_idx], axis=0)
    t['bip_graph'] = _now() - s

    arrays = dict(x_hw=x_hw, nx_v1=nx_v1, ei_v1=ei_v1, ea_v1=ea_v1,
                  nx_v2=nx_v2, ei_v2=ei_v2, ea_v2=ea_v2, elem_x=elem_x, cei=cei, cea=cea)
    return t, arrays, (lines, nodes, elems)


def encode_timed(runs, arrays):
    """numpy -> torch -> GPU + z-score, por arquitetura (1º run de cada arch;
    stats de mse/mae diferem só nos valores). Retorna {arch: s}."""
    out = {}
    first = {}
    for k, r in runs.items():
        first.setdefault(r['arch'], r)
    for arch, r in first.items():
        nrm = r['normalizer']
        torch.cuda.synchronize()
        s = _now()
        x = nrm.encode(torch.from_numpy(arrays['x_hw'])[None].to(DEVICE), 'x_hw')
        if arch in ('FNO2d',):
            nx = torch.from_numpy(arrays['nx_v1']).to(DEVICE)   # r/c dos nós p/ interpolação
        elif arch in ('FNO_GNN', 'GNN_PostBase'):
            nx = nrm.encode(torch.from_numpy(arrays['nx_v1']).to(DEVICE), 'node_x')
            ei = torch.from_numpy(arrays['ei_v1']).to(DEVICE)
            ea = torch.from_numpy(arrays['ea_v1']).to(DEVICE)
        else:
            nx = nrm.encode(torch.from_numpy(arrays['nx_v2']).to(DEVICE), 'node_x')
            ex = nrm.encode(torch.from_numpy(arrays['elem_x']).to(DEVICE), 'elem_x')
            ei = torch.from_numpy(arrays['ei_v2']).to(DEVICE)
            ea = torch.from_numpy(arrays['ea_v2']).to(DEVICE)
            ci = torch.from_numpy(arrays['cei']).to(DEVICE)
            ca = torch.from_numpy(arrays['cea']).to(DEVICE)
        torch.cuda.synchronize()
        out[arch] = _now() - s
    return out


def _check_equal(arrays, ref):
    v1, bp = ref['FNO_GNN'], ref['FNO_BipartiteGNN']
    pairs = [('x_hw', arrays['x_hw'], bp['x_hw']), ('nx_v1', arrays['nx_v1'], v1['node_x']),
             ('ei_v1', arrays['ei_v1'], v1['edge_index']), ('ea_v1', arrays['ea_v1'], v1['edge_attr']),
             ('nx_v2', arrays['nx_v2'], bp['node_x']), ('ei_v2', arrays['ei_v2'], bp['edge_index']),
             ('ea_v2', arrays['ea_v2'], bp['edge_attr']), ('elem_x', arrays['elem_x'], bp['elem_x']),
             ('cei', arrays['cei'], bp['cross_edge_index']), ('cea', arrays['cea'], bp['cross_edge_attr'])]
    return {n: bool(a.shape == b.shape and np.array_equal(a, b)) for n, a, b in pairs}


# --------------------------------------------------------------------------- #
# 3a + 3b
# --------------------------------------------------------------------------- #
def run_femm():
    import femm
    from src.data_gen.motor_model import BLDC_FEMM_Model_Sym120_Annular

    rows = design_rows()
    samples = test_samples()
    pick = np.linspace(0, len(samples) - 1, N_FEMM).round().astype(int)
    picked = [samples[i] for i in pick]
    runs = load_runs()          # normalizers p/ etapa de encode (modelos não usados aqui)

    FEMM_WORK.mkdir(parents=True, exist_ok=True)
    cwd0 = os.getcwd()
    rec = []
    checks = []
    try:
        for n, (chunk, idx, gz_path) in enumerate(picked):
            row = rows[idx]
            r_in = float(row['inner_diameter [mm]']) / 2
            r_ext = float(row['outer_diameter [mm]']) / 2
            d = FEMM_WORK / f's{idx:06d}'
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            os.chdir(d)
            tm = {}
            try:
                # [REMOVIDO 2026-10-06] phase=0 — a geração agora aplica rotor_phase
                # model = BLDC_FEMM_Model_Sym120_Annular(motor_params=motor_params_from_row(row), phase=0)
                model = BLDC_FEMM_Model_Sym120_Annular(motor_params=motor_params_from_row(row),
                                                       phase=float(row['rotor_phase [deg]']))
                s = _now()
                femm.openfemm('bHide')
                femm.main_resize(1000, 1000)
                tm['openfemm'] = _now() - s

                s = _now()
                femm.newdocument(0)
                femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, 0, 200)
                model.draw_motor()
                femm.mi_zoomnatural()
                femm.mi_saveas(str(d / 'model.fem'))
                tm['geometry'] = _now() - s

                s = _now()
                femm.mi_createmesh()
                tm['mesh'] = _now() - s

                s = _now()
                femm.mi_analyze()
                tm['solve'] = _now() - s

                s = _now()
                femm.closefemm()
                tm['closefemm'] = _now() - s
            finally:
                os.chdir(cwd0)

            ans = d / 'model.ans'
            # pós-FEMM: leitura do .ans + B nodal (curl(A) P1 + média simples)
            s = _now()
            _lines, nodes, elems = _parse_solution(str(ans))
            bx, by = _element_b_from_A(nodes, elems, nodes[:, 2].astype(np.float32))
            nbx = _node_mean_of_elements(elems, bx, nodes.shape[0])
            nby = _node_mean_of_elements(elems, by, nodes.shape[0])
            tm['post_read_B'] = _now() - s

            # 3b -- pré-processamento das entradas (mesmo .ans)
            tp, arrays, _ = preprocess_timed(ans, r_in, r_ext)
            te = encode_timed(runs, arrays)
            tm.update({f'pre_{k}': v for k, v in tp.items()})
            tm.update({f'enc_{k}': v for k, v in te.items()})

            # conferências (não cronometradas): saída == parser oficial; malha == raw
            gz_tmp = d / 'model.ans.gz'
            with open(ans, 'rb') as fi, gzip.open(gz_tmp, 'wb') as fo:
                shutil.copyfileobj(fi, fo)
            ref = parse_ans_gzip_sample_unified(gz_tmp, r_in, r_ext, ang_1=ANG1, ang_2=ANG2,
                                                n_r=N_R, n_a=N_A, tmp_dir=d)
            eq = _check_equal(arrays, ref)
            nodeB_eq = bool(np.array_equal(np.concatenate([nbx, nby], 1), ref['FNO_BipartiteGNN']['node_y']))
            with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
                raw_lines = f.readlines()
            k0 = next(i for i, l in enumerate(raw_lines) if l.strip().startswith('[Solution]'))
            raw_nodes = int(raw_lines[k0 + 1])
            raw_elems = int(raw_lines[k0 + 2 + raw_nodes])
            raw_A = np.loadtxt(raw_lines[k0 + 2: k0 + 2 + raw_nodes])[:, 2]
            same_mesh = raw_nodes == nodes.shape[0] and raw_elems == elems.shape[0]
            maxdA = float(np.abs(raw_A - nodes[:, 2]).max()) if same_mesh else None
            checks.append(dict(sample_idx=idx, preproc_equal=eq, nodeB_equal=nodeB_eq,
                               mesh_equal_raw=same_mesh, max_abs_dA_vs_raw=maxdA,
                               n_nodes=int(nodes.shape[0]), n_elems=int(elems.shape[0])))
            rec.append(dict(sample_idx=idx, n_nodes=int(nodes.shape[0]), n_elems=int(elems.shape[0]),
                            **{k: v for k, v in tm.items()}))
            shutil.rmtree(d, ignore_errors=True)
            print(f"  [{n + 1}/{len(picked)}] amostra {idx}: nós={nodes.shape[0]} "
                  f"geom {tm['geometry']*1e3:.0f} ms  malha {tm['mesh']*1e3:.0f} ms  "
                  f"solução {tm['solve']*1e3:.0f} ms  pós {tm['post_read_B']*1e3:.0f} ms  "
                  f"pre(bip) {sum(tp.values())*1e3:.0f} ms  ok={all(eq.values()) and nodeB_eq} "
                  f"malha=raw:{same_mesh}", flush=True)
    finally:
        os.chdir(cwd0)

    keys = [k for k in rec[0] if k not in ('sample_idx', 'n_nodes', 'n_elems')]
    summary_ms = {k: _iqr([1e3 * r[k] for r in rec]) for k in keys}
    _merge_json('femm_preproc', dict(
        n_samples=len(rec), sample_idx=[r['sample_idx'] for r in rec],
        per_sample=rec, summary_ms=summary_ms, checks=checks,
        all_preproc_equal=all(all(c['preproc_equal'].values()) for c in checks),
        all_nodeB_equal=all(c['nodeB_equal'] for c in checks),
        n_mesh_equal_raw=sum(c['mesh_equal_raw'] for c in checks),
        n_nodes=_iqr([r['n_nodes'] for r in rec]),
    ))
    _merge_json('hardware', hardware_info())
    print(f'salvo em {OUT_JSON}')


# --------------------------------------------------------------------------- #
# 3c -- inferência
# --------------------------------------------------------------------------- #
def _collate(samples, layout):
    """Concatena amostras (dicts de to_torch_sample()[layout]) com offsets --
    mesma regra do collate dos loaders (nó/aresta; + elemento no bipartite)."""
    out = dict(x_hw=torch.cat([s['x_hw'] for s in samples], 0),
               L=torch.cat([s['L'] for s in samples]),
               node_x=torch.cat([s['node_x'] for s in samples], 0),
               edge_attr=torch.cat([s['edge_attr'] for s in samples], 0))
    eis, ceis, cas, exs = [], [], [], []
    no = eo = 0
    for s in samples:
        eis.append(s['edge_index'] + no)
        if layout == 'bip':
            c = s['cross_edge_index'].clone()
            c[0] += eo
            c[1] += no
            ceis.append(c)
            cas.append(s['cross_edge_attr'])
            exs.append(s['elem_x'])
            eo += s['elem_x'].shape[0]
        no += s['node_x'].shape[0]
    out['edge_index'] = torch.cat(eis, 1)
    if layout == 'bip':
        out.update(cross_edge_index=torch.cat(ceis, 1), cross_edge_attr=torch.cat(cas, 0),
                   elem_x=torch.cat(exs, 0))
    return out


def _prepare_inputs(arch, normalizer, batch):
    """Entrada já codificada e no device (pré-processamento fora da medição)."""
    g = {k: v.to(DEVICE) for k, v in batch.items()}
    enc = normalizer.encode
    if arch == 'FNO2d':
        return dict(x=enc(g['x_hw'], 'x_hw'), r=g['node_x'][:, 3], c=g['node_x'][:, 4], L=g['L'])
    if arch in ('FNO_GNN', 'GNN_PostBase'):
        return dict(x=enc(g['x_hw'], 'x_hw'), nx=enc(g['node_x'], 'node_x'),
                    ei=g['edge_index'], ea=g['edge_attr'], L=g['L'])
    return dict(x=enc(g['x_hw'], 'x_hw'), nx=enc(g['node_x'], 'node_x'), ex=enc(g['elem_x'], 'elem_x'),
                ei=g['edge_index'], ea=g['edge_attr'], ci=g['cross_edge_index'],
                ca=g['cross_edge_attr'], L=g['L'])


def _forward(arch, model, normalizer, inp, interp_mode):
    """Forward + desnormalização (+ interpolação grade->nós no FNO2d)."""
    if arch == 'FNO2d':
        out_hw = normalizer.decode(model(inp['x']), 'y_hw')
        return interpolate_grid_to_nodes(out_hw, inp['r'], inp['c'], inp['L'], mode=interp_mode)
    if arch in ('FNO_GNN', 'GNN_PostBase'):
        _, yn = model(inp['x'], inp['nx'], inp['ei'], inp['ea'], inp['L'])
    else:
        _, yn = model(inp['x'], inp['nx'], inp['ex'], inp['ei'], inp['ea'], inp['ci'], inp['ca'], inp['L'])
    return normalizer.decode(yn, 'node_y')


@torch.inference_mode()
def _time_batch(arch, r, batch):
    model, nrm = r['model'], r['normalizer']
    inp = _prepare_inputs(arch, nrm, batch)
    for _ in range(N_WARMUP):
        _forward(arch, model, nrm, inp, r['interp_mode'])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    ts = []
    for _ in range(N_REPS):
        torch.cuda.synchronize()
        s = _now()
        _forward(arch, model, nrm, inp, r['interp_mode'])
        torch.cuda.synchronize()
        ts.append(_now() - s)
    peak = torch.cuda.max_memory_allocated()
    del inp
    return dict(ms=_iqr(1e3 * np.array(ts)), peak_mem_gib=peak / 2 ** 30,
                peak_mem_above_resident_gib=(peak - base_mem) / 2 ** 30)


def run_infer():
    from concurrent.futures import ProcessPoolExecutor
    rows = design_rows()
    samples = test_samples()
    first_chunk = [s for s in samples if s[0] == samples[0][0]]
    with ProcessPoolExecutor(max_workers=8) as ex:
        parsed = list(ex.map(parse_worker, [p for _, _, p in first_chunk],
                             [rows[i] for _, i, _ in first_chunk]))
    ts = [to_torch_sample(s) for s in parsed]
    n_nodes = np.array([t['v1']['node_x'].shape[0] for t in ts])
    i_med = int(np.argsort(n_nodes)[len(n_nodes) // 2])     # amostra de nº de nós mediano (batch 1)
    del parsed

    # Windows/WDDM: sem limite, o driver transborda para a memória compartilhada do
    # sistema (sysmem fallback) em vez de dar OOM -- o batch "cabe" mas fica ordens de
    # grandeza mais lento. Limita o alocador a 95% da VRAM dedicada para o OOM ser real.
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
    runs = load_runs()
    # só 1 modelo na GPU por vez (pico de memória isolado)
    for r in runs.values():
        r['model'].to('cpu')
    torch.cuda.empty_cache()

    res = dict(batch1_sample_idx=int(first_chunk[i_med][1]), batch1_n_nodes=int(n_nodes[i_med]),
               chunk=first_chunk[0][0], n_warmup=N_WARMUP, n_reps=N_REPS,
               mem_fraction_limit=MEM_FRACTION, runs={})
    for k, r in runs.items():
        arch = r['arch']
        layout = 'bip' if arch == 'FNO_BipartiteGNN' else 'v1'
        r['model'].to(DEVICE).eval()
        torch.cuda.empty_cache()
        out = dict(arch=arch, loss=r['loss'], run=r['run'],
                   n_params=sum(p.numel() * (2 if p.is_complex() else 1) for p in r['model'].parameters()))
        out['batch1'] = _time_batch(arch, r, _collate([ts[i_med][layout]], layout))
        print(f"  {k:24s} batch 1: {out['batch1']['ms']['median']:.2f} ms "
              f"(IQR {out['batch1']['ms']['iqr']:.2f}), pico {out['batch1']['peak_mem_gib']:.2f} GiB", flush=True)
        sizes, bmax = {}, None
        b = 2
        while b <= 512:
            batch = _collate([ts[i % len(ts)][layout] for i in range(b)], layout)
            try:
                rr = _time_batch(arch, r, batch)
            except torch.cuda.OutOfMemoryError:
                del batch
                torch.cuda.empty_cache()
                break
            rr['ms_per_sample_median'] = rr['ms']['median'] / b
            sizes[b] = rr
            bmax = b
            print(f"      batch {b:3d}: {rr['ms']['median']:.1f} ms ({rr['ms_per_sample_median']:.2f} ms/amostra), "
                  f"pico {rr['peak_mem_gib']:.2f} GiB", flush=True)
            del batch
            torch.cuda.empty_cache()
            b *= 2
        out['batch_sweep'] = {str(kk): v for kk, v in sizes.items()}
        out['batch_max'] = bmax
        out['batch_max_note'] = ('maior potência de 2 que coube (até 512); amostras do 1º chunk de '
                                 'teste repetidas ciclicamente')
        r['model'].to('cpu')
        torch.cuda.empty_cache()
        res['runs'][k] = out
    _merge_json('inference', res)
    _merge_json('hardware', hardware_info())
    print(f'salvo em {OUT_JSON}')


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('femm', 'all'):
        run_femm()
    if what in ('infer', 'all'):
        run_infer()
