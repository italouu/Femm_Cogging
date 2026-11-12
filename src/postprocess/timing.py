"""
Custo computacional: FEMM × pré-processamento + inferência, com a MESMA malha dos dois lados
(a geração da malha não entra em nenhum tempo). Ver docs/claude/avaliacao.md §5.

Definições:
  t_FEMM — só o solver: duração do processo fkn.exe disparado pelo mi_analyze. O mi_analyze
           SEMPRE refaz a malha (roda o triangle.exe 2× antes do fkn.exe, com ou sem
           mi_createmesh antes -- check_analyze_remesh), então o tempo do mi_analyze inteiro
           inclui a malha e não é o t_FEMM. Os filhos do femm.exe são acompanhados por
           polling (POLL_S; o intervalo real é medido e gravado); o fkn inclui a leitura da
           malha e a escrita do .ans pelo próprio solver. Fora: abrir/fechar o FEMM,
           geometria, mi_createmesh, triangle.exe, ler o .ans.
  t_pre  — .ans resolvido -> entrada codificada na GPU, por etapa, com as MESMAS funções dos
           parsers (check_preprocess = igualdade exata): read_mesh (linhas + x, y e
           conectividade/rótulos; a coluna A da solução vai para read_solution, fora do
           t_pre -- ressalva: x, y e A estão na mesma linha do .ans, então os bytes de A
           são lidos do disco no read_mesh), materiais, x_hw (malha -> grade), grafos e
           encode (z-score + CPU -> GPU, com cuda.synchronize).
  t_inf  — best.pth, torch.inference_mode, lote 1: forward + desnormalização (+ grade -> nós
           no FNO2d, que entrega o campo na malha como o FEMM; parcelas grade/interpolação
           gravadas à parte), com cuda.synchronize antes e depois.

  femm_solve_timed      — redesenha (fase do rotor, corrente do CSV), gera a malha (fora) e
                          cronometra o mi_analyze com triangle.exe / fkn.exe por polling.
  check_analyze_remesh  — o mi_analyze remalha? (com/sem mi_createmesh antes; processos e
                          malha/A × raw).
  preprocess_timed      — .ans -> entradas de cada layout (vertex / bip / elem / split), por etapa.
  encode_timed          — numpy -> GPU + z-score (inference._enc), por arch.
  time_per_sample       — t_inf por amostra (aquecimento no início, n repetições por amostra).
  time_batch            — lote b (varredura até OOM; fora da tabela).

Limite de memória da GPU no Windows: sem limite o driver WDDM transborda para a RAM em vez
de dar OOM (MEM_FRACTION em scripts/postprocess.py).
"""
# [REMOVIDO] docstring antiga (só o mi_analyze no cronômetro do FEMM, 5 archs sem a Split):
# substituída pelas definições acima depois do teste de remalha (2026-11-12).
import gzip
import os
import platform
import shutil
import subprocess
import threading
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import psutil
import torch

from src.data_gen.parsers.ans_parsing import (
    _MAGNET_ID, _MU_BY_ID, _build_bidirectional_edge_attrs, _build_edges, _element_areas,
    _grid_polar_xy, _node_current_density, _node_magnet_polarity, _node_material_priority,
    _node_dual_area, _parse_label_currents, _parse_label_materials, _parse_solution,
    _wrap_edge_pairs, read_ans,
)
from src.data_gen.parsers.elem_structure import _build_dual_edges
from src.data_gen.parsers.bipartite_structure import _build_trifinder, _grid_const_per_element
from src.data_gen.parsers.split_layout import _split_vertices
from src.neural_op.archs.interp import interpolate_grid_to_nodes
from src.postprocess.inference import DEVICE, _dec, _enc

# [REMOVIDO] etapas antigas: 'read_ans' (incluía a coluna A da solução) e 'bip_graph' com o
# elem_x junto (a Split usa o elem_x mas não as arestas do bipartido). Substituídas por
# 'read_mesh' (+ 'read_solution' fora do t_pre) e 'elem_feats' + 'bip_graph'.
# ARCH_STAGES = {
#     'FNO2d':                 ('read_ans', 'materials', 'x_hw', 'node_pos'),
#     'FNO_GNN':               ('read_ans', 'materials', 'x_hw', 'node_pos', 'graph_topology', 'v1_graph'),
#     'GNN_PostBase':          ('read_ans', 'materials', 'x_hw', 'node_pos', 'graph_topology', 'v1_graph'),
#     'FNO_BipartiteGNN':      ('read_ans', 'materials', 'x_hw', 'node_pos', 'graph_topology', 'bip_graph'),
#     'FNO_BipartiteGNN_Elem': ('read_ans', 'materials', 'x_hw', 'node_pos', 'graph_topology', 'bip_graph',
#                               'elem_graph'),
# }
# LAYOUT = {'FNO2d': 'vertex', 'FNO_GNN': 'vertex', 'GNN_PostBase': 'vertex', 'FNO_BipartiteGNN': 'bip',
#           'FNO_BipartiteGNN_Elem': 'elem'}

# etapas do pré-processamento usadas por cada arch (a soma + encode é o t_pre da arch).
# 'read_solution' (coluna A) é medida mas não entra em nenhuma arch.
_BASE = ('read_mesh', 'materials', 'x_hw', 'node_pos')
ARCH_STAGES = {
    'FNO2d':                 _BASE,                     # node_pos: r/c dos nós p/ grade -> nós
    'FNO_GNN':               _BASE + ('graph_topology', 'v1_graph'),
    'GNN_PostBase':          _BASE + ('graph_topology', 'v1_graph'),
    'FNO_BipartiteGNN':      _BASE + ('graph_topology', 'elem_feats', 'bip_graph'),
    'FNO_BipartiteGNN_Elem': _BASE + ('graph_topology', 'elem_feats', 'bip_graph', 'elem_graph'),
    'FNO_SplitGNN':          _BASE + ('elem_feats', 'split_graph'),
}
LAYOUT = {'FNO2d': 'vertex', 'FNO_GNN': 'vertex', 'GNN_PostBase': 'vertex', 'FNO_BipartiteGNN': 'bip',
          'FNO_BipartiteGNN_Elem': 'elem', 'FNO_SplitGNN': 'split'}

POLL_S = 0.004          # espera entre leituras dos processos do FEMM (s); o intervalo real é medido
FEMM_CHILDREN = ('triangle.exe', 'fkn.exe')


def _now():
    return time.perf_counter()


def iqr_stats(x):
    """Mediana, quartis e IQR (vão para a tabela/texto); média e desvio padrão (registro)."""
    x = np.asarray(x, dtype=np.float64)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return dict(median=float(med), q1=float(q1), q3=float(q3), iqr=float(q3 - q1),
                mean=float(x.mean()), std=float(x.std(ddof=1)) if x.size > 1 else 0.0,
                min=float(x.min()), max=float(x.max()), n=int(x.size))


# --------------------------------------------------------------------------- #
# hardware / software
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
        psutil=ver('psutil'), femm=_femm_exe_version())


# --------------------------------------------------------------------------- #
# afinidade de CPU (CPUs híbridas: núcleos P × E)
# --------------------------------------------------------------------------- #
def performance_cpus():
    """CPUs lógicas de maior EfficiencyClass (núcleos P numa CPU híbrida Intel), pelo
    GetSystemCpuSetInformation do Windows (grupo 0). Sem a API ou CPU não híbrida -> todas."""
    try:
        import ctypes
        import ctypes.wintypes as wt
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        n = wt.ULONG(0)
        k.GetSystemCpuSetInformation(None, 0, ctypes.byref(n), None, 0)
        buf = (ctypes.c_ubyte * n.value)()
        if not k.GetSystemCpuSetInformation(buf, n, ctypes.byref(n), None, 0):
            raise OSError(ctypes.get_last_error())
        raw, off, ent = bytes(buf), 0, []
        while off < n.value:       # SYSTEM_CPU_SET_INFORMATION: Size, Type, Id, Group, LP, Core, LLC, NUMA, EffClass
            size = int.from_bytes(raw[off:off + 4], 'little')
            grp = int.from_bytes(raw[off + 12:off + 14], 'little')
            ent.append((grp, raw[off + 14], raw[off + 18]))
            off += size
        top = max(e for _, _, e in ent)
        return sorted(lp for g, lp, e in ent if g == 0 and e == top)
    except Exception:           # noqa: BLE001
        return list(range(os.cpu_count()))


def set_affinity(cpus, pid=None):
    """Afinidade do processo (pid None = este). Filhos criados depois herdam (Windows)."""
    psutil.Process(pid).cpu_affinity(list(cpus))


# --------------------------------------------------------------------------- #
# FEMM: processos filhos do femm.exe (triangle.exe = malha, fkn.exe = solver)
# --------------------------------------------------------------------------- #
def _femm_pid():
    """PID do femm.exe aberto pelo openfemm (o mais recente)."""
    cand = [p for p in psutil.process_iter(['name', 'create_time'])
            if (p.info['name'] or '').lower() == 'femm.exe']
    if not cand:
        raise RuntimeError('femm.exe não encontrado (openfemm antes de _femm_pid)')
    return max(cand, key=lambda p: p.info['create_time']).pid


class ChildWatch(threading.Thread):
    """Polling dos filhos do femm.exe a cada POLL_S. Por PID: 1ª e última vez visto (perf_counter).
    A duração medida (última − 1ª) subestima a real em até ~1 intervalo em cada ponta; os
    intervalos reais entre leituras ficam em self.ticks."""
    def __init__(self, parent_pid, poll_s=POLL_S):
        super().__init__(daemon=True)
        self.parent = psutil.Process(parent_pid)
        self.poll_s = poll_s
        self._halt = threading.Event()
        self.seen = {}      # (pid, 1ª) -> [nome, 1ª, última]
        self.affinity = {}  # (pid, 1ª) -> afinidade de CPU do filho (herdada do femm.exe)
        self.ticks = []

    # [REMOVIDO] run() com self.parent.children() a cada leitura: ~6,5 ms por chamada (varre
    # todos os processos), intervalo real ~11,6 ms > 10 ms. Substituído por psutil.pids()
    # (~0,7 ms) + nome/pai só dos PIDs novos.
    # def run(self):
    #     while not self._halt.is_set():
    #         now = _now()
    #         try:
    #             kids = self.parent.children()
    #         except psutil.Error:
    #             kids = []
    #         for k in kids:
    #             a = self.seen.get(k.pid)
    #             if a is None:
    #                 try:
    #                     name = k.name().lower()
    #                 except psutil.Error:
    #                     continue
    #                 self.seen[k.pid] = [name, now, now]
    #             else:
    #                 a[2] = now
    #         self.ticks.append(now)
    #         self._halt.wait(self.poll_s)

    def run(self):
        """A cada leitura: psutil.pids(); PID ausente na leitura anterior = processo novo
        (nome e pai consultados só nele); processo acompanhado presente = ainda vivo. Chave
        (pid, 1ª leitura): um PID reaproveitado depois de sumir conta como outro processo."""
        prev, live = set(psutil.pids()), {}     # live: pid -> chave em self.seen
        ppid = self.parent.pid
        while not self._halt.is_set():
            now = _now()
            cur = set(psutil.pids())
            for pid in cur - prev:
                try:
                    pr = psutil.Process(pid)
                    name = pr.name().lower()
                    if name in FEMM_CHILDREN and pr.ppid() == ppid:
                        key = (pid, now)
                        self.seen[key] = [name, now, now]
                        live[pid] = key
                        try:
                            self.affinity[key] = pr.cpu_affinity()
                        except psutil.Error:
                            self.affinity[key] = None
                except psutil.Error:
                    continue
            for pid in list(live):
                if pid in cur:
                    self.seen[live[pid]][2] = now
                else:
                    del live[pid]
            prev = cur
            self.ticks.append(now)
            self._halt.wait(self.poll_s)

    def finish(self):
        self._halt.set()
        self.join()

    def procs(self, t0):
        """[{name, pid, start_s, end_s, span_s}] em ordem de início (relativo a t0)."""
        out = [dict(name=n, pid=key[0], start_s=a - t0, end_s=b - t0, span_s=b - a,
                    cpu_affinity=self.affinity.get(key))
               for key, (n, a, b) in self.seen.items() if n in FEMM_CHILDREN]
        return sorted(out, key=lambda d: d['start_s'])

    def poll_intervals_ms(self):
        return 1e3 * np.diff(np.asarray(self.ticks))


def _timed_with_children(fn, pid):
    """Roda fn() (chamada COM bloqueante) com o polling dos filhos do femm.exe."""
    w = ChildWatch(pid)
    w.start()
    s = _now()
    fn()
    t = _now() - s
    w.finish()
    return t, w.procs(s), w.poll_intervals_ms()


def _proc_summary(procs):
    tri = [p['span_s'] for p in procs if p['name'] == 'triangle.exe']
    fkn = [p['span_s'] for p in procs if p['name'] == 'fkn.exe']
    return dict(n_triangle=len(tri), n_fkn=len(fkn), triangle=float(sum(tri)),
                fkn=float(sum(fkn)))


def _draw_fem(row, fem: Path):
    """Desenha o design do CSV e salva o .fem sem malha (sequência de
    femm_solve.save_ans_gzip_sample)."""
    import femm
    from src.data_gen.motor_constants import stack_length_mm
    from src.data_gen.motor_model import BLDC_FEMM_Model_Sym120_Annular
    from src.postprocess.torque import params_of
    p = params_of(row)
    model = BLDC_FEMM_Model_Sym120_Annular(motor_params=p, phase=p['rotor_phase']['value'])
    femm.newdocument(0)
    femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, stack_length_mm(p), 200)
    model.draw_motor()
    femm.mi_zoomnatural()
    femm.mi_saveas(str(fem))


def femm_solve_timed_procs(row, work_dir: Path, cpus=None):
    """Mesma sequência de femm_solve.save_ans_gzip_sample. Cronometra à parte openfemm,
    geometria, mi_createmesh e closefemm (fora da comparação) e o mi_analyze com os filhos
    triangle.exe / fkn.exe. t_FEMM = 'fkn'. Retorna (tempos s, info, caminho do .ans):
    tempos = openfemm, geometry, mesh, mi_analyze, triangle, fkn, closure (= mi_analyze −
    triangle − fkn), closefemm; info = processos, nº de triangle/fkn, intervalos de polling.
    cpus: afinidade aplicada ao femm.exe logo após o openfemm (triangle/fkn herdam)."""
    import femm
    work_dir = Path(work_dir).resolve()
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    tm = {}
    cwd0 = os.getcwd()
    os.chdir(work_dir)
    try:
        s = _now(); femm.openfemm('bHide'); femm.main_resize(1000, 1000); tm['openfemm'] = _now() - s
        pid = _femm_pid()
        if cpus is not None:
            set_affinity(cpus, pid)
        s = _now(); _draw_fem(row, work_dir / 'model.fem'); tm['geometry'] = _now() - s
        s = _now(); femm.mi_createmesh(); tm['mesh'] = _now() - s
        tm['mi_analyze'], procs, ticks = _timed_with_children(femm.mi_analyze, pid)
        ps = _proc_summary(procs)
        tm['triangle'], tm['fkn'] = ps['triangle'], ps['fkn']
        tm['closure'] = tm['mi_analyze'] - tm['triangle'] - tm['fkn']
        s = _now(); femm.closefemm(); tm['closefemm'] = _now() - s
    finally:
        os.chdir(cwd0)
    info = dict(procs=procs, n_triangle=ps['n_triangle'], n_fkn=ps['n_fkn'],
                children_affinity_ok=(None if cpus is None else
                                      all(p['cpu_affinity'] == sorted(cpus) for p in procs)),
                poll_ms=dict(median=float(np.median(ticks)), p99=float(np.percentile(ticks, 99)),
                             max=float(ticks.max()), frac_above_10ms=float((ticks > 10).mean()),
                             n=int(ticks.size))
                if ticks.size else None)
    return tm, info, work_dir / 'model.ans'


def check_analyze_remesh(row, raw_gz, work_dir: Path, reps=3, cpus=None):
    """O mi_analyze refaz a malha? Por repetição, com o .fem salvo sem malha e reaberto:
      A) mi_analyze direto (sem mi_createmesh);
      B) mi_createmesh e, em seguida, mi_analyze.
    Mede os tempos e os filhos do femm.exe (triangle.exe / fkn.exe) em cada chamada, e confere
    a malha e o A das duas soluções entre si e com o raw (check_mesh_vs_raw).
    Remalha <=> triangle.exe roda dentro do mi_analyze do caso B."""
    import femm
    work = Path(work_dir).resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    rec = []
    cwd0 = os.getcwd()
    femm.openfemm('bHide')
    femm.main_resize(1000, 1000)
    pid = _femm_pid()
    if cpus is not None:
        set_affinity(cpus, pid)
    try:
        for k in range(reps):
            r = dict(rep=k)
            for case in ('A', 'B'):
                d = work / f'rep{k}_{case}'
                d.mkdir()
                os.chdir(d)
                fem = d / 'model.fem'
                _draw_fem(row, fem)
                femm.mi_close()
                femm.opendocument(str(fem))            # reabre sem malha em memória
                if case == 'B':
                    r['t_createmesh_s'], pr, _ = _timed_with_children(femm.mi_createmesh, pid)
                    r['createmesh_procs'] = pr
                t, pr, _ = _timed_with_children(femm.mi_analyze, pid)
                femm.mi_close()
                os.chdir(cwd0)
                key = 'sem_malha' if case == 'A' else 'apos_malha'
                r[f't_analyze_{key}_s'] = t
                r[f'analyze_{key}_procs'] = pr
                r[f'analyze_{key}_summary'] = _proc_summary(pr)
                lines, nodes, elems, span = read_mesh_input(d / 'model.ans')
                r[f'_mesh_{case}'] = (np.column_stack([nodes, read_solution_A(lines, span)]), elems)
            (na, ea), (nb, eb) = r.pop('_mesh_A'), r.pop('_mesh_B')
            r['malha_A_igual_B'] = bool(na.shape == nb.shape and np.array_equal(na[:, :2], nb[:, :2])
                                        and np.array_equal(ea, eb))
            r['max_abs_dA_A_B'] = float(np.abs(na[:, 2] - nb[:, 2]).max()) if na.shape == nb.shape else None
            r['vs_raw_A'] = check_mesh_vs_raw(na, ea, raw_gz)
            r['vs_raw_B'] = check_mesh_vs_raw(nb, eb, raw_gz)
            rec.append(r)
            sb = r['analyze_apos_malha_summary']
            print(f"  rep {k}: createmesh {r['t_createmesh_s'] * 1e3:.0f} ms | analyze sem malha "
                  f"{r['t_analyze_sem_malha_s'] * 1e3:.0f} ms | analyze após malha "
                  f"{r['t_analyze_apos_malha_s'] * 1e3:.0f} ms (triangle×{sb['n_triangle']} "
                  f"{sb['triangle'] * 1e3:.0f} ms, fkn {sb['fkn'] * 1e3:.0f} ms) | malha A=B "
                  f"{r['malha_A_igual_B']}, =raw {r['vs_raw_B']['mesh_equal_raw']}", flush=True)
    finally:
        os.chdir(cwd0)
        femm.closefemm()
        shutil.rmtree(work, ignore_errors=True)
    med = lambda f: float(np.median([f(r) for r in rec]) * 1e3)
    remesh = all(r['analyze_apos_malha_summary']['n_triangle'] > 0 for r in rec)
    return dict(
        reps=reps, poll_s=POLL_S, remeshes=bool(remesh),
        mediana_ms=dict(createmesh=med(lambda r: r['t_createmesh_s']),
                        analyze_sem_malha=med(lambda r: r['t_analyze_sem_malha_s']),
                        analyze_apos_malha=med(lambda r: r['t_analyze_apos_malha_s']),
                        triangle_no_analyze_apos_malha=med(lambda r: r['analyze_apos_malha_summary']['triangle']),
                        fkn_no_analyze_apos_malha=med(lambda r: r['analyze_apos_malha_summary']['fkn'])),
        mesh_equal_raw_all=all(r['vs_raw_A']['mesh_equal_raw'] and r['vs_raw_B']['mesh_equal_raw'] for r in rec),
        max_abs_dA_vs_raw=max(max(r['vs_raw_A']['max_abs_dA_vs_raw'] or 0.0, r['vs_raw_B']['max_abs_dA_vs_raw'] or 0.0)
                              for r in rec),
        leitura=('remalha <=> triangle.exe roda dentro do mi_analyze mesmo com mi_createmesh antes; '
                 't_FEMM = só o fkn.exe'),
        repeticoes=rec)


# [REMOVIDO] femm_solve_timed: cronometrava o mi_analyze inteiro como "solve", mas o
# mi_analyze refaz a malha (check_analyze_remesh) -- substituída por femm_solve_timed_procs
# (t_FEMM = fkn.exe). Comentada até a atualização ser confirmada.
# --------------------------------------------------------------------------- #
# FEMM: só o mi_analyze no cronômetro
# --------------------------------------------------------------------------- #
# def femm_solve_timed(row, work_dir: Path):
#     """Mesma sequência de femm_solve.save_ans_gzip_sample; cronometra openfemm, geometria e
#     malha à parte (fora da comparação) e o mi_analyze (solução). Retorna (tempos s, .ans).
#
#     O tempo "solve" é o do mi_analyze; se o mi_analyze refaz a malha, "solve" = malha +
#     solução (se o mi_analyze remalha ou não está em aberto: docs/claude/avaliacao.md §5)."""
#     import femm
#     from src.data_gen.motor_constants import stack_length_mm
#     from src.data_gen.motor_model import BLDC_FEMM_Model_Sym120_Annular
#     from src.postprocess.torque import params_of
#     work_dir = Path(work_dir).resolve()
#     shutil.rmtree(work_dir, ignore_errors=True)
#     work_dir.mkdir(parents=True)
#     p = params_of(row)
#     tm = {}
#     cwd0 = os.getcwd()
#     os.chdir(work_dir)
#     try:
#         model = BLDC_FEMM_Model_Sym120_Annular(motor_params=p, phase=p['rotor_phase']['value'])
#         s = _now(); femm.openfemm('bHide'); femm.main_resize(1000, 1000); tm['openfemm'] = _now() - s
#         s = _now()
#         femm.newdocument(0)
#         femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, stack_length_mm(p), 200)
#         model.draw_motor(); femm.mi_zoomnatural(); femm.mi_saveas(str(work_dir / 'model.fem'))
#         tm['geometry'] = _now() - s
#         s = _now(); femm.mi_createmesh(); tm['mesh'] = _now() - s
#         s = _now(); femm.mi_analyze(); tm['solve'] = _now() - s
#         s = _now(); femm.closefemm(); tm['closefemm'] = _now() - s
#     finally:
#         os.chdir(cwd0)
#     return tm, work_dir / 'model.ans'


def check_mesh_vs_raw(nodes, elems, raw_gz):
    """Malha (coordenadas, conectividade, rótulos) e A do .ans novo × o do raw."""
    raw = read_ans(raw_gz)
    rn, re_ = raw.nodes, raw.elems
    same = rn.shape == nodes.shape and re_.shape == elems.shape
    out = dict(mesh_equal_raw=bool(same and np.array_equal(rn[:, :2], nodes[:, :2])
                                   and np.array_equal(re_, elems)),
               n_nodes=int(nodes.shape[0]), n_elems=int(elems.shape[0]))
    out['max_abs_dA_vs_raw'] = float(np.abs(rn[:, 2] - nodes[:, 2]).max()) if same else None
    return out


# --------------------------------------------------------------------------- #
# leitura do .ans: entrada (malha + materiais) × solução (A)
# --------------------------------------------------------------------------- #
def read_mesh_input(ans_path):
    """Leitura do .ans só para a ENTRADA da rede: linhas (cabeçalho do .fem: materiais,
    rótulos, circuitos) e, do bloco [Solution], x, y dos nós e n1, n2, n3, rótulo dos
    elementos -- os mesmos valores de _parse_solution, sem converter a coluna A (ver
    read_solution_A). Ressalva: no .ans x, y e A estão na mesma linha, então os bytes de A são
    lidos do disco aqui. Aceita .ans ou .ans.gz. Retorna (lines, nodes [S,2], elems [M,4],
    (1ª linha dos nós, nº de nós))."""
    p = str(ans_path)
    if p.endswith('.gz'):
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
    else:
        with open(p) as f:
            lines = f.readlines()
    idx = next(i for i, l in enumerate(lines) if l.strip().startswith('[Solution]'))
    n_nodes = int(lines[idx + 1].strip())
    i0 = idx + 2
    nodes = np.loadtxt(lines[i0: i0 + n_nodes], usecols=(0, 1))
    i2 = i0 + n_nodes
    n_elems = int(lines[i2].strip())
    elems = np.loadtxt(lines[i2 + 1: i2 + 1 + n_elems], usecols=(0, 1, 2, 3)).astype(np.int64)
    return lines, nodes, elems, (i0, n_nodes)


def read_solution_A(lines, span):
    """Coluna A (potencial vetor, solução) dos nós -- fora do t_pre."""
    i0, n = span
    return np.loadtxt(lines[i0: i0 + n], usecols=(2,))


# --------------------------------------------------------------------------- #
# pré-processamento por etapa (funções dos parsers smooth; só as entradas)
# --------------------------------------------------------------------------- #
def preprocess_timed(ans_path, r_in, r_ext, ang_1=0.0, ang_2=120.0, n_r=138, n_a=276):
    """.ans -> entradas de cada layout (vertex / bip / elem / split), cronometrado por etapa.
    Retorna (tempos s {etapa}, arrays {layout: {chave: array}}, (nodes [S,3] x,y,A, elems));
    a etapa 'read_solution' (coluna A) não entra em nenhuma arch."""
    t = {}
    a1, a2 = np.deg2rad(ang_1), np.deg2rad(ang_2)

    # [REMOVIDO] read_ans: _parse_solution convertia também a coluna A (solução)
    # s = _now()
    # lines, nodes, elems = _parse_solution(str(ans_path))
    # t['read_ans'] = _now() - s
    s = _now()
    lines, nodes, elems, span = read_mesh_input(ans_path)     # nodes [S,2]: só x, y
    t['read_mesh'] = _now() - s

    s = _now()
    label_material_id, label_mu, label_M = _parse_label_materials(lines)
    elem_material_id = label_material_id[elems[:, 3]]
    elem_mu, elem_M = label_mu[elems[:, 3]], label_M[elems[:, 3]]
    area = _element_areas(nodes, elems)
    elem_J = _parse_label_currents(lines, elems, area)[elems[:, 3]]
    t['materials'] = _now() - s

    # x_hw (bipartite_structure.parse_bipartite_structure)
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

    # vertex (FNO_GNN / GNN_PostBase; smooth_layouts): material por prioridade
    s = _now()
    node_material_id = _node_material_priority(elems, elem_material_id, nodes.shape[0])
    node_dual_area = _node_dual_area(elems, area, nodes.shape[0])
    node_mu = _MU_BY_ID[node_material_id]
    node_M = np.where(node_material_id == _MAGNET_ID,
                      _node_magnet_polarity(nodes, elems, area, elem_material_id, elem_M),
                      0.0).astype(np.float32)
    node_J = _node_current_density(elems, area, elem_J.astype(np.float32), elem_material_id,
                                   node_material_id)
    ei_vertex, ea_vertex = _build_bidirectional_edge_attrs(nodes, edges_undirected, wrap_1, wrap_2,
                                                   r_base, c_base, node_mu, n_r, n_a)
    nx_vertex = np.stack([node_mu.astype(np.float32), node_M, node_dual_area.astype(np.float32),
                      r_base.astype(np.float32), c_base.astype(np.float32), node_J], axis=1)
    t['v1_graph'] = _now() - s

    # atributos dos elementos (elem_x do bipartite_structure; usados por bip, Elem e Split)
    s = _now()
    n_el = elems.shape[0]
    r_e = np.hypot(centroids[:, 0], centroids[:, 1])
    th_e = np.arctan2(centroids[:, 1], centroids[:, 0])
    elem_x = np.stack([elem_mu, elem_M, area.astype(np.float32),
                       ((r_e - r_in) / (r_ext - r_in)).astype(np.float32),
                       ((th_e - a1) / (a2 - a1)).astype(np.float32), elem_J], axis=1).astype(np.float32)
    t['elem_feats'] = _now() - s

    # bipartite (bipartite_structure.parse_bipartite_structure)
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
    # [REMOVIDO] elem_x dentro do bip_graph -- movido para a etapa 'elem_feats' (acima)
    # n_el = elems.shape[0]
    # r_e = np.hypot(centroids[:, 0], centroids[:, 1])
    # th_e = np.arctan2(centroids[:, 1], centroids[:, 0])
    # elem_x = np.stack([elem_mu, elem_M, area.astype(np.float32),
    #                    ((r_e - r_in) / (r_ext - r_in)).astype(np.float32),
    #                    ((th_e - a1) / (a2 - a1)).astype(np.float32), elem_J], axis=1).astype(np.float32)
    tri_idx = np.repeat(np.arange(n_el, dtype=np.int64), 3)
    vtx_idx = elems[:, :3].reshape(-1).astype(np.int64)
    cea = np.linalg.norm(nodes[vtx_idx, :2] - centroids[tri_idx], axis=1).astype(np.float32)[:, None]
    cei = np.stack([tri_idx, vtx_idx], axis=0)
    t['bip_graph'] = _now() - s

    # Elem (elem_structure.parse_elem_structure): elementos = grafo principal
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

    # Split (split_layout.parse_split_smooth, só as entradas): N_aux = elementos, N_base =
    # vértices fundidos em 0°/120° e divididos por material
    s = _now()
    ex = elem_x.astype(np.float64)
    e_mu, e_M, e_area, e_r, e_c, e_J = (ex[:, k] for k in range(6))
    e_sin, e_cos = np.sin(2 * np.pi * e_c), np.cos(2 * np.pi * e_c)
    nx_sp = np.stack([e_r, e_c, e_sin, e_cos, e_mu, e_M, e_J], axis=1).astype(np.float32)
    elem_cls = label_material_id[elems[:, 3]] * 4 + (np.rint(e_M).astype(np.int64) + 1)
    copy_of_corner, copy_vertex, first_corner = _split_vertices(nodes, elems, elem_cls, ang_1, ang_2)
    n_copies = copy_vertex.shape[0]
    e_flat = np.repeat(np.arange(n_el, dtype=np.int64), 3)
    e_first = e_flat[first_corner]
    v_r, v_c = rb32.astype(np.float64), cb32.astype(np.float64)
    c_r, c_c = v_r[copy_vertex], v_c[copy_vertex]
    c_sin, c_cos = np.sin(2 * np.pi * c_c), np.cos(2 * np.pi * c_c)
    w = np.bincount(copy_of_corner, e_area[e_flat], minlength=n_copies)
    c_J = np.bincount(copy_of_corner, (e_area * e_J)[e_flat], minlength=n_copies) / w
    ex_sp = np.stack([c_r, c_c, c_sin, c_cos, e_mu[e_first], e_M[e_first], c_J],
                     axis=1).astype(np.float32)
    s_int, t_int, s_wrap, t_wrap = _build_dual_edges(nodes, elems, ang_1, ang_2)
    len_int = np.linalg.norm(centroids[t_int] - centroids[s_int], axis=1)
    phi = -np.deg2rad(ang_2 - ang_1)
    cj = centroids[t_wrap]
    cj_rot = np.stack([cj[:, 0] * np.cos(phi) - cj[:, 1] * np.sin(phi),
                       cj[:, 0] * np.sin(phi) + cj[:, 1] * np.cos(phi)], axis=1)
    len_wrap = np.linalg.norm(cj_rot - centroids[s_wrap], axis=1)
    si, sj = np.concatenate([s_int, s_wrap]), np.concatenate([t_int, t_wrap])
    length = np.concatenate([len_int, len_wrap])
    src, dst = np.concatenate([si, sj]), np.concatenate([sj, si])
    ei_sp = np.stack([src, dst], axis=0).astype(np.int64)
    ea_sp = np.stack([e_r[dst] - e_r[src], e_sin[dst] - e_sin[src], e_cos[dst] - e_cos[src],
                      np.concatenate([length, length]),
                      e_mu[dst] - e_mu[src], e_J[dst] - e_J[src], e_M[dst] - e_M[src]],
                     axis=1).astype(np.float32)
    v_orig = elems[:, :3].reshape(-1)
    aux_len = np.linalg.norm(nodes[v_orig, :2] - centroids[e_flat], axis=1)
    cei_sp = np.stack([copy_of_corner, e_flat], axis=0).astype(np.int64)
    cea_sp = np.stack([c_r[copy_of_corner] - e_r[e_flat], c_sin[copy_of_corner] - e_sin[e_flat],
                       c_cos[copy_of_corner] - e_cos[e_flat], aux_len,
                       e_mu[e_flat], e_J[e_flat], e_M[e_flat]], axis=1).astype(np.float32)
    t['split_graph'] = _now() - s

    # solução (A): fora do t_pre, só registrada e usada nas conferências × raw
    s = _now()
    node_A = read_solution_A(lines, span)
    t['read_solution'] = _now() - s

    L_n = np.array([nodes.shape[0]], np.int64)
    L_e = np.array([n_el], np.int64)
    arrays = dict(
        vertex=dict(x_hw=x_hw, node_x=nx_vertex, edge_index=ei_vertex, edge_attr=ea_vertex, L=L_n),
        bip=dict(x_hw=x_hw, node_x=nx_bp, elem_x=elem_x, edge_index=ei_bp, edge_attr=ea_bp,
                 cross_edge_index=cei, cross_edge_attr=cea, L=L_n),
        elem=dict(x_hw=x_hw, node_x=nx_el, elem_x=nx_bp, edge_index=ei_el, edge_attr=ea_el,
                  cross_edge_index=cei_el, cross_edge_attr=cea, L=L_e),
        split=dict(x_hw=x_hw, node_x=nx_sp, elem_x=ex_sp, edge_index=ei_sp, edge_attr=ea_sp,
                   cross_edge_index=cei_sp, cross_edge_attr=cea_sp, L=L_e,
                   elem_L=np.array([n_copies], np.int64)))
    # [REMOVIDO] return t, arrays, (nodes, elems)   -- nodes agora sem A no caminho da entrada
    return t, arrays, (np.column_stack([nodes, node_A]), elems)


def check_preprocess(arrays, ref):
    """Igualdade exata das entradas × parsers (ref = samples.parse_smooth(..., with_elem=True)
    do mesmo .ans, + ref['split'] = split_layout.parse_split_smooth, se presente) ->
    {layout.chave: bool}."""
    out = {}
    bip_keys = ('x_hw', 'node_x', 'elem_x', 'edge_index', 'edge_attr', 'cross_edge_index',
                'cross_edge_attr', 'L')
    layouts = [('vertex', ('x_hw', 'node_x', 'edge_index', 'edge_attr', 'L')),
               ('bip', bip_keys), ('elem', bip_keys)]
    if 'split' in ref:
        layouts.append(('split', bip_keys + ('elem_L',)))
    for lay, keys in layouts:
        for k in keys:
            a, b = np.asarray(arrays[lay][k]), np.asarray(ref[lay][k])
            out[f'{lay}.{k}'] = bool(a.shape == b.shape and np.array_equal(a, b))
    return out


# [REMOVIDO] encode_timed antigo: z-score na CPU e depois .to(DEVICE), enquanto a inferência
# (_prepare) passa à GPU e codifica lá -- substituído por encode_timed abaixo, que usa o
# próprio _prepare (a entrada cronometrada é a mesma do t_inf) e inclui a Split.
# def encode_timed(runs, arrays):
#     """numpy -> GPU + z-score (inference._enc), por arch (1ª run de cada arch). {arch: s}."""
#     out, first = {}, {}
#     for r in runs.values():
#         first.setdefault(r['arch'], r)
#     for arch, r in first.items():
#         d, nrm = arrays[LAYOUT[arch]], r['normalizer']
#         g = lambda k: torch.from_numpy(np.ascontiguousarray(d[k]))
#         torch.cuda.synchronize()
#         s = _now()
#         _enc(nrm, g('x_hw')[None], 'x_hw').to(DEVICE)
#         if arch == 'FNO2d':
#             g('node_x').to(DEVICE)                   # r/c dos nós para a interpolação
#         else:
#             _enc(nrm, g('node_x'), 'node_x').to(DEVICE)
#             for k in ('edge_index', 'L'):
#                 g(k).to(DEVICE)
#             _enc(nrm, g('edge_attr'), 'edge_attr').to(DEVICE)
#             if LAYOUT[arch] != 'vertex':
#                 _enc(nrm, g('elem_x'), 'elem_x').to(DEVICE)
#                 g('cross_edge_index').to(DEVICE)
#                 _enc(nrm, g('cross_edge_attr'), 'cross_edge_attr').to(DEVICE)
#         torch.cuda.synchronize()
#         out[arch] = _now() - s
#     return out


def to_torch(d):
    """Layout numpy de 1 amostra (preprocess_timed) -> tensores CPU (x_hw com o eixo do lote)."""
    out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in d.items()}
    out['x_hw'] = out['x_hw'][None]
    return out


def encode_timed(runs, arrays):
    """Codificação da entrada: numpy -> tensores -> GPU + z-score, pelo MESMO _prepare da
    inferência, com cuda.synchronize antes e depois; por arch (1ª run de cada arch).
    Retorna ({arch: s}, parcelas {nome: s}); parcela 'FNO2d.node_x' = r/c dos nós à GPU (só
    para a interpolação grade -> nós do FNO2d; já incluída no total do FNO2d)."""
    out, parts, first = {}, {}, {}
    for r in runs.values():
        first.setdefault(r['arch'], r)
    for arch, r in first.items():
        d, nrm = arrays[LAYOUT[arch]], r['normalizer']
        torch.cuda.synchronize()
        s = _now()
        inp = _prepare(arch, nrm, to_torch(d))
        torch.cuda.synchronize()
        out[arch] = _now() - s
        if arch == 'FNO2d':
            s = _now()
            torch.from_numpy(np.ascontiguousarray(d['node_x'])).to(DEVICE)
            torch.cuda.synchronize()
            parts['FNO2d.node_x'] = _now() - s
        del inp
    return out, parts


# --------------------------------------------------------------------------- #
# inferência
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
        if layout != 'vertex':
            c = s['cross_edge_index'].clone()
            c[0] += eo
            c[1] += no
            ceis.append(c); cas.append(s['cross_edge_attr']); exs.append(s['elem_x'])
            eo += s['elem_x'].shape[0]
        no += s['node_x'].shape[0]
    out['edge_index'] = torch.cat(eis, 1)
    if layout != 'vertex':
        out.update(cross_edge_index=torch.cat(ceis, 1), cross_edge_attr=torch.cat(cas, 0),
                   elem_x=torch.cat(exs, 0))
    if layout == 'split':           # nº de cópias N_base por amostra
        out['elem_L'] = torch.cat([s['elem_L'] for s in samples])
    return out


def _prepare(arch, nrm, batch):
    """Entrada já codificada e no device (pré-processamento fora da medição)."""
    g = {k: v.to(DEVICE) for k, v in batch.items()}
    x = _enc(nrm, g['x_hw'], 'x_hw')
    if arch == 'FNO2d':
        return dict(x=x, r=g['node_x'][:, 3], c=g['node_x'][:, 4], L=g['L'])
    out = dict(x=x, nx=_enc(nrm, g['node_x'], 'node_x'), ei=g['edge_index'],
               ea=_enc(nrm, g['edge_attr'], 'edge_attr'), L=g['L'])
    if LAYOUT[arch] != 'vertex':
        out.update(ex=_enc(nrm, g['elem_x'], 'elem_x'), ci=g['cross_edge_index'],
                   ca=_enc(nrm, g['cross_edge_attr'], 'cross_edge_attr'))
    if LAYOUT[arch] == 'split':
        out['eL'] = g['elem_L']
    return out


def _forward(arch, model, nrm, inp):
    """Forward + desnormalização (+ grade -> nós no FNO2d, como inference.predict; Split ->
    (N_aux, N_base), como inference.predict_split)."""
    if arch == 'FNO2d':
        out_hw = _dec(nrm, model(inp['x']), 'y_hw')
        return interpolate_grid_to_nodes(out_hw, inp['r'], inp['c'], inp['L'])
    if LAYOUT[arch] == 'split':
        _, ya, yb = model(inp['x'], inp['nx'], inp['ex'], inp['ei'], inp['ea'], inp['ci'], inp['ca'],
                          inp['L'], inp['eL'])
        return _dec(nrm, ya, 'node_y'), _dec(nrm, yb, 'elem_y')
    if LAYOUT[arch] == 'vertex':
        _, yn = model(inp['x'], inp['nx'], inp['ei'], inp['ea'], inp['L'])
    else:
        _, yn = model(inp['x'], inp['nx'], inp['ex'], inp['ei'], inp['ea'], inp['ci'], inp['ca'], inp['L'])
    return _dec(nrm, yn, 'node_y')


def gpu_spinup(seconds):
    """Mantém a GPU ocupada (matmul genérico, sem o modelo) por `seconds` para o clock sair do
    repouso antes dos aquecimentos: sem isso a 1ª amostra do 1º modelo medido sai ~3× mais
    lenta (os 3 aquecimentos somam ~10 ms)."""
    if seconds <= 0:
        return
    a = torch.randn(2048, 2048, device=DEVICE)
    s = _now()
    while _now() - s < seconds:
        a @ a
        torch.cuda.synchronize()
    del a


def _reps_ms(fn, n_reps):
    """n_reps execuções de fn com cuda.synchronize antes e depois de cada uma -> ms."""
    ts = []
    for _ in range(n_reps):
        torch.cuda.synchronize()
        s = _now()
        fn()
        torch.cuda.synchronize()
        ts.append(1e3 * (_now() - s))
    return ts


@torch.inference_mode()
def time_per_sample(arch, run, samples, n_warmup, n_reps):
    """t_inf de lote 1 em cada amostra (dicts de to_torch do layout da arch): n_warmup
    forwards na 1ª amostra (só no início) e n_reps por amostra. Entrada preparada e na GPU
    antes do cronômetro. FNO2d: também as parcelas 'grid' (forward + desnormalização na
    grade) e 'interp' (grade -> nós), com as mesmas repetições.
    Retorna (lista por amostra {parcela: [ms]}, pico de memória GiB)."""
    model, nrm = run['model'], run['normalizer']
    lay = LAYOUT[arch]
    inp = _prepare(arch, nrm, collate([samples[0]], lay))
    for _ in range(n_warmup):
        _forward(arch, model, nrm, inp)
    del inp
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    per = []
    for smp in samples:
        inp = _prepare(arch, nrm, collate([smp], lay))
        torch.cuda.synchronize()
        rec = dict(total=_reps_ms(lambda: _forward(arch, model, nrm, inp), n_reps))
        if arch == 'FNO2d':
            rec['grid'] = _reps_ms(lambda: _dec(nrm, model(inp['x']), 'y_hw'), n_reps)
            out_hw = _dec(nrm, model(inp['x']), 'y_hw')
            rec['interp'] = _reps_ms(
                lambda: interpolate_grid_to_nodes(out_hw, inp['r'], inp['c'], inp['L']), n_reps)
            del out_hw
        per.append(rec)
        del inp
    return per, torch.cuda.max_memory_allocated() / 2 ** 30


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
