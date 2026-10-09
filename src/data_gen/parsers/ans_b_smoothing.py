"""Suavização de B do pós-processador do FEMM, lida direto de um arquivo .ans.

Módulo isolado (só numpy + stdlib, sem `import femm`, sem dependência do resto do
projeto). Porte fiel de xfemm, commit 34ac6cc586a346066984526620dfb67383c8e3c5,
`cfemm/fpproc/fpproc.cpp`:

    GetElementB  (ramo PLANAR)                -> element_b
    GetNodalB    (só Frequency == 0)          -> nodal_b
    GetPointB                                 -> point_b
    Ctr                                       -> ctr
    NumList/ConList (linhas ~1907-1923)       -> build_conlist

Restrições (levantam erro em vez de dar resultado errado): problema planar,
magnetostático (Frequency == 0), sem MagDirFctn em Lua, sem modo incremental.

Saídas principais:
    B1, B2  [M]     B por elemento (P0)                — elm.B1/elm.B2
    b1, b2  [M,3]   B nos 3 vértices de cada elemento  — elm.b1[]/elm.b2[]
                    (P1 descontínuo; o que `mo_smooth('on')` interpola)

Uso:
    mesh = load_ans('sample_000000.ans.gz')
    B1, B2 = element_b(mesh)
    b1, b2 = nodal_b(mesh, B1, B2)
    Bx, By = point_b(mesh, B1, B2, b1, b2, elem, x, y, smooth=True)
"""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field

import numpy as np


_LENGTH_CONV = {  # fpproc.cpp linhas 110-115 / 371-382 (mesma ordem de teste)
    'inches': 0.0254, 'millimeters': 0.001, 'centimeters': 0.01,
    'meters': 1.0, 'mils': 2.54e-05, 'microns': 1.e-06,
}


@dataclass
class AnsMesh:
    length_conv: float
    problem_type: str           # 'planar' | 'axisymmetric'
    frequency: float
    # [BlockProps] (blockproplist)
    mu_x: np.ndarray
    mu_y: np.ndarray
    H_c: np.ndarray
    # [PointProps] (nodeproplist) — só J (I_re/I_im)
    pp_J: np.ndarray            # complex [n_pointprops]
    # [NumPoints] (nodelist) — x, y, BoundaryMarker (0-based, -1 = nenhum)
    pts_x: np.ndarray
    pts_y: np.ndarray
    pts_marker: np.ndarray
    # [NumBlockLabels] (blocklist)
    bl_type: np.ndarray         # BlockType 0-based
    bl_magdir: np.ndarray
    # [Solution]
    x: np.ndarray               # [N] coordenadas nas unidades do arquivo
    y: np.ndarray
    A: np.ndarray               # [N] Wb/m
    p: np.ndarray               # [M,3] int
    lbl: np.ndarray             # [M]   índice em blocklist
    # derivados por elemento
    blk: np.ndarray = field(init=False)     # blocklist[lbl].BlockType
    magdir: np.ndarray = field(init=False)  # blocklist[lbl].MagDir

    def __post_init__(self):
        self.blk = self.bl_type[self.lbl]
        self.magdir = self.bl_magdir[self.lbl]


def _value(line: str) -> str:
    return line.split('=', 1)[1].strip()


def load_ans(path) -> AnsMesh:
    """Lê .ans (ou .ans.gz) — só o que GetElementB/GetNodalB/GetPointB usam."""
    path = str(path)
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()

    length_conv, problem_type, frequency = None, 'planar', 0.0
    mu_x, mu_y, H_c, pp_J = [], [], [], []
    pts, bls = [], []
    cur_block = cur_point = None
    i, n_lines = 0, len(lines)
    while i < n_lines:
        s = lines[i].strip()
        low = s.lower()
        if low.startswith('[lengthunits]'):
            u = _value(s).split()[0].lower()
            length_conv = next((v for k, v in _LENGTH_CONV.items()
                                if u.startswith(k if k != 'centimeters' else 'c')), None)
        elif low.startswith('[problemtype]'):
            problem_type = 'axisymmetric' if _value(s).lower().startswith('axi') else 'planar'
        elif low.startswith('[frequency]'):
            frequency = float(_value(s))
        elif low.startswith('<beginblock>'):
            cur_block = {'mu_x': 1.0, 'mu_y': 1.0, 'h_c': 0.0}
        elif cur_block is not None and low.startswith(('<mu_x>', '<mu_y>', '<h_c>')):
            cur_block[low[1:low.index('>')]] = float(_value(s))
        elif low.startswith('<endblock>'):
            mu_x.append(cur_block['mu_x']); mu_y.append(cur_block['mu_y'])
            H_c.append(cur_block['h_c']); cur_block = None
        elif low.startswith('<beginpoint>'):
            cur_point = {'i_re': 0.0, 'i_im': 0.0}
        elif cur_point is not None and low.startswith(('<i_re>', '<i_im>')):
            cur_point[low[1:low.index('>')]] = float(_value(s))
        elif low.startswith('<endpoint>'):
            pp_J.append(complex(cur_point['i_re'], cur_point['i_im'])); cur_point = None
        elif low.startswith('[numpoints]'):
            k = int(_value(s))
            for ln in lines[i + 1:i + 1 + k]:
                t = ln.split()
                pts.append((float(t[0]), float(t[1]), int(t[2]) - 1))
            i += k
        elif low.startswith('[numblocklabels]'):
            k = int(_value(s))
            for ln in lines[i + 1:i + 1 + k]:
                t = ln.split()
                # x y BlockType MaxArea InCircuit MagDir InGroup Turns flags [MagDirFctn]
                if len(t) > 9 and t[9].strip('"'):
                    raise NotImplementedError('MagDirFctn (Lua) não suportado')
                bls.append((int(t[2]) - 1, float(t[5])))
            i += k
        elif low.startswith('[solution]'):
            break
        i += 1
    if length_conv is None:
        raise ValueError('[LengthUnits] ausente/desconhecido')

    n_nodes = int(lines[i + 1])
    nodes = np.array([ln.split()[:3] for ln in lines[i + 2:i + 2 + n_nodes]], dtype=float)
    j = i + 2 + n_nodes
    n_elem = int(lines[j])
    elems = np.array([ln.split()[:4] for ln in lines[j + 1:j + 1 + n_elem]], dtype=np.int64)

    pts = np.array(pts, dtype=float).reshape(-1, 3)
    bls = np.array(bls, dtype=float).reshape(-1, 2)
    return AnsMesh(
        length_conv=length_conv, problem_type=problem_type, frequency=frequency,
        mu_x=np.array(mu_x), mu_y=np.array(mu_y), H_c=np.array(H_c),
        pp_J=np.array(pp_J, dtype=complex),
        pts_x=pts[:, 0], pts_y=pts[:, 1], pts_marker=pts[:, 2].astype(np.int64),
        bl_type=bls[:, 0].astype(np.int64), bl_magdir=bls[:, 1],
        x=nodes[:, 0], y=nodes[:, 1], A=nodes[:, 2],
        p=elems[:, :3], lbl=elems[:, 3],
    )


def _check_supported(mesh: AnsMesh):
    if mesh.problem_type != 'planar':
        raise NotImplementedError('só problema planar')
    if mesh.frequency != 0:
        raise NotImplementedError('só Frequency == 0 (magnetostático)')


# ---------------------------------------------------------------------------
# Ctr / ConList
# ---------------------------------------------------------------------------

def ctr(mesh: AnsMesh):
    """Ctr(i) para todos os elementos — c = 0 + p0/3 + p1/3 + p2/3 (mesma ordem)."""
    x, y, p = mesh.x, mesh.y, mesh.p
    cx = x[p[:, 0]] / 3. + x[p[:, 1]] / 3. + x[p[:, 2]] / 3.
    cy = y[p[:, 0]] / 3. + y[p[:, 1]] / 3. + y[p[:, 2]] / 3.
    return cx, cy


def build_conlist(mesh: AnsMesh):
    """NumList/ConList (fpproc.cpp ~1907-1923) em formato CSR.

    ConList[k] = con_idx[con_ptr[k]:con_ptr[k+1]], na ordem em que o laço
    `for i in elementos: for j in 0..2` os insere (elemento crescente).
    """
    n_nodes = mesh.x.size
    flat = mesh.p.reshape(-1)                                   # ordem (i, j)
    num = np.bincount(flat, minlength=n_nodes)
    con_ptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(num, out=con_ptr[1:])
    order = np.argsort(flat, kind='stable')                     # preserva ordem (i, j)
    con_idx = (order // 3).astype(np.int64)
    return num, con_ptr, con_idx


# ---------------------------------------------------------------------------
# GetElementB (PLANAR)
# ---------------------------------------------------------------------------

def element_b(mesh: AnsMesh):
    """B constante por elemento (P0), em Tesla."""
    _check_supported(mesh)
    x, y, A, p, L = mesh.x, mesh.y, mesh.A, mesh.p, mesh.length_conv
    n0, n1, n2 = p[:, 0], p[:, 1], p[:, 2]
    b = [y[n1] - y[n2], y[n2] - y[n0], y[n0] - y[n1]]
    c = [x[n2] - x[n1], x[n0] - x[n2], x[n1] - x[n0]]
    da = b[0] * c[1] - b[1] * c[0]
    B1 = np.zeros(p.shape[0])
    B2 = np.zeros(p.shape[0])
    for i in range(3):
        B1 += A[p[:, i]] * c[i] / (da * L)
        B2 -= A[p[:, i]] * b[i] / (da * L)
    return B1, B2


# ---------------------------------------------------------------------------
# GetNodalB (Frequency == 0)
# ---------------------------------------------------------------------------

def _same_material(mesh: AnsMesh, e, n):
    """Critério do `m++` de GetNodalB (Frequency==0), vetorizado em pares (e, n)."""
    bx, bn = mesh.blk[e], mesh.blk[n]
    same_md = mesh.magdir[e] == mesh.magdir[n]
    return ((mesh.lbl[e] == mesh.lbl[n])
            | ((mesh.mu_x[bx] == mesh.mu_x[bn]) & (mesh.mu_y[bx] == mesh.mu_y[bn])
               & (mesh.H_c[bx] == mesh.H_c[bn]) & same_md)
            | ((bx == bn) & same_md))


def _scan_interface(mesh, B1, B2, con, ei, k, direction):
    """Um dos laços 'scan ccw/cw for an interface' de GetNodalB.

    Retorna (kind, payload):
      ('punt', e)            — nxt == -1 (special-case punt), e = elemento atual
      ('iface', (e, pt))     — interface achada entre e e nxt (lbl diferente)
      ('none', None)         — laço esgotou NumList[k] iterações sem achar nada
    """
    p, lbl = mesh.p, mesh.lbl
    e = ei
    for _q in range(len(con)):
        pt = 0
        for j in range(3):
            if p[e, j] == k:
                pt = j
        if direction == 'ccw':
            pt -= 1
            if pt < 0:
                pt = 2
        else:
            pt += 1
            if pt > 2:
                pt = 0
        pt = int(p[e, pt])
        nxt = -1
        for cj in con:
            if cj != e:
                for l in range(3):
                    if p[cj, l] == pt:
                        nxt = int(cj)
        if nxt == -1:
            return 'punt', e
        if lbl[ei] != lbl[nxt]:
            return 'iface', (e, pt)
        e = nxt
    return 'none', None


def _interface_contrib(mesh, B1, B2, e, k, pt):
    """Contribuição do lado de interface (k, pt) — corpo comum aos dois scans."""
    L = mesh.length_conv
    tn = complex(mesh.x[pt] - mesh.x[k], mesh.y[pt] - mesh.y[k])
    atn = abs(tn)
    bn = (mesh.A[pt] - mesh.A[k]) / (atn * L)
    z = 0.5 / atn
    tn = tn / abs(tn)
    bt = B1[e] * tn.real + B2[e] * tn.imag              # kludge with bt
    d1 = z * tn.real * bt
    d2 = z * tn.imag * bt
    d1b = z * tn.imag * bn
    d2b = -z * tn.real * bn
    return z, d1, d2, d1b, d2b, tn


def _nodal_b_vertex(mesh, B1, B2, con, ei, i):
    """Ramo 'else' (m != NumList[k]) de GetNodalB para o vértice i do elemento ei.

    Retorna (b1, b2, rule) com rule em {'interface', 'punt', 'corner'}.
    """
    k = int(mesh.p[ei, i])
    R = 0.0
    v1 = 0j
    v2 = 0j
    b1 = 0.0
    b2 = 0.0
    rule = 'interface'

    # scan ccw
    kind, pay = _scan_interface(mesh, B1, B2, con, ei, k, 'ccw')
    if kind == 'punt':
        b1, b2 = B1[pay], B2[pay]
        v1, v2 = 1 + 0j, 1 + 0j
        rule = 'punt'
    elif kind == 'iface':
        z, d1, d2, d1b, d2b, tn = _interface_contrib(mesh, B1, B2, pay[0], k, pay[1])
        R += z
        b1 += d1; b2 += d2
        b1 += d1b; b2 += d2b
        v1 = tn

    # scan cw
    if v2 == 0:
        kind, pay = _scan_interface(mesh, B1, B2, con, ei, k, 'cw')
        if kind == 'punt':
            b1, b2 = B1[pay], B2[pay]
            v1, v2 = 1 + 0j, 1 + 0j
            rule = 'punt'
        elif kind == 'iface':
            z, d1, d2, d1b, d2b, tn = _interface_contrib(mesh, B1, B2, pay[0], k, pay[1])
            R += z
            b1 += d1; b2 += d2
            b1 += d1b; b2 += d2b
            v2 = tn
        # b1[i]/=R; b2[i]/=R  (dentro do if(v2==0), como no original)
        with np.errstate(divide='ignore', invalid='ignore'):  # R==0 -> inf/nan, como em C++
            b1 = np.float64(b1) / np.float64(R)
            b2 = np.float64(b2) / np.float64(R)

    # corner too sharp?
    flag = False
    if abs(v1) < 0.9 or abs(v2) < 0.9:
        flag = True
    if (-v1.real * v2.real - v1.imag * v2.imag) > 0.985:
        flag = True
    if not flag:
        rule = 'corner'
        bnr = 0.0
        for cj in con:
            if mesh.lbl[ei] == mesh.lbl[cj]:
                btr = np.sqrt(B1[cj] * B1[cj] + B2[cj] * B2[cj])
                if btr > bnr:
                    bnr = btr
        Rm = np.sqrt(B1[ei] * B1[ei] + B2[ei] * B2[ei])
        if Rm != 0:
            b1 = bnr / Rm * B1[ei]
            b2 = bnr / Rm * B2[ei]
        else:
            b1 = 0.0
            b2 = 0.0
    return b1, b2, rule


def nodal_b(mesh: AnsMesh, B1, B2, conlist=None, return_rule=False):
    """elm.b1[3]/elm.b2[3] de todos os elementos (P1 descontínuo).

    return_rule=True devolve também `rule` [M,3] (int8):
        0 = normal (média ponderada por 1/dist ao centróide)
        1 = interface (regra ccw/cw)
        2 = special-case punt (nxt == -1, borda do domínio)
        3 = canto agudo (punt por |B| máximo)
        4 = ponto com corrente pontual (usa B do elemento)
    """
    _check_supported(mesh)
    num, con_ptr, con_idx = conlist if conlist is not None else build_conlist(mesh)
    M = mesh.p.shape[0]
    n_nodes = mesh.x.size

    # --- contagem m para cada par (elemento, vértice), vetorizada -------------
    k_flat = mesh.p.reshape(-1)                                 # [3M]
    e_flat = np.repeat(np.arange(M), 3)
    cnt = num[k_flat]
    e_rep = np.repeat(e_flat, cnt)
    starts = con_ptr[k_flat]
    offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    n_rep = con_idx[np.repeat(starts, cnt) + offs]
    same = _same_material(mesh, e_rep, n_rep)
    m = np.add.reduceat(same.astype(np.int64), np.cumsum(cnt) - cnt) if cnt.size else cnt
    normal = (m == cnt).reshape(M, 3)

    # --- caso normal: depende só do nó k (mesmo laço p/ qualquer elm) --------
    cx, cy = ctr(mesh)
    nk = np.repeat(np.arange(n_nodes), num)                     # nó de cada entrada CSR
    dx = mesh.x[nk] - cx[con_idx]
    dy = mesh.y[nk] - cy[con_idx]
    z = 1. / np.sqrt(dx * dx + dy * dy)
    Rn = np.zeros(n_nodes)
    s1 = np.zeros(n_nodes)
    s2 = np.zeros(n_nodes)
    np.add.at(Rn, nk, z)                                        # sequencial, ordem ConList
    np.add.at(s1, nk, z * B1[con_idx])
    np.add.at(s2, nk, z * B2[con_idx])
    with np.errstate(invalid='ignore', divide='ignore'):
        node_b1 = s1 / Rn
        node_b2 = s2 / Rn

    b1 = node_b1[mesh.p].copy()
    b2 = node_b2[mesh.p].copy()
    rule = np.zeros((M, 3), dtype=np.int8)

    # --- interface / punt / canto: laço fiel ---------------------------------
    rule_code = {'interface': 1, 'punt': 2, 'corner': 3}
    for ei, i in zip(*np.nonzero(~normal)):
        k = mesh.p[ei, i]
        con = con_idx[con_ptr[k]:con_ptr[k + 1]]
        vb1, vb2, r = _nodal_b_vertex(mesh, B1, B2, con, int(ei), int(i))
        b1[ei, i], b2[ei, i] = vb1, vb2
        rule[ei, i] = rule_code[r]

    # --- ponto com corrente pontual: usa B do elemento -----------------------
    if mesh.pp_J.size != 0:
        has_J = np.zeros(mesh.pts_x.size, dtype=bool)
        mk = mesh.pts_marker
        ok = mk >= 0
        has_J[ok] = mesh.pp_J[mk[ok]] != 0
        for jx, jy in zip(mesh.pts_x[has_J], mesh.pts_y[has_J]):
            hit = np.abs((mesh.x[mesh.p] - jx) + 1j * (mesh.y[mesh.p] - jy)) < 1.e-08
            for ei, i in zip(*np.nonzero(hit)):
                b1[ei, i], b2[ei, i] = B1[ei], B2[ei]
                rule[ei, i] = 4

    return (b1, b2, rule) if return_rule else (b1, b2)


# ---------------------------------------------------------------------------
# GetPointB
# ---------------------------------------------------------------------------

def point_b(mesh: AnsMesh, B1, B2, b1, b2, elem, x, y, smooth=True):
    """B no ponto (x, y) [unidades do arquivo] do elemento `elem` (vetorizado).

    smooth=False -> P0 (elm.B1/B2); smooth=True -> interpolação linear de b1/b2.
    """
    elem = np.asarray(elem)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not smooth:
        return B1[elem], B2[elem]
    X, Y = mesh.x, mesh.y
    n = mesh.p[elem]
    n0, n1, n2 = n[..., 0], n[..., 1], n[..., 2]
    a = [X[n1] * Y[n2] - X[n2] * Y[n1],
         X[n2] * Y[n0] - X[n0] * Y[n2],
         X[n0] * Y[n1] - X[n1] * Y[n0]]
    b = [Y[n1] - Y[n2], Y[n2] - Y[n0], Y[n0] - Y[n1]]
    c = [X[n2] - X[n1], X[n0] - X[n2], X[n1] - X[n0]]
    da = b[0] * c[1] - b[1] * c[0]
    o1 = np.zeros(np.shape(x))
    o2 = np.zeros(np.shape(x))
    for i in range(3):
        o1 = o1 + (b1[elem, i] * (a[i] + b[i] * x + c[i] * y) / da)
        o2 = o2 + (b2[elem, i] * (a[i] + b[i] * x + c[i] * y) / da)
    return o1, o2
