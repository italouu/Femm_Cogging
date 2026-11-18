"""
.ans do FEMM: restrições de contorno da solução.

  ans_constraints(lines, nodes) : nós Dirichlet (A prescrito = 0) e pares periódicos /
                                  antiperiódicos, lidos do próprio .ans -> classes de nós
                                  amarrados (usado por weighted_mst.t22_python(periodic=True)
                                  para amarrar a função peso g nos nós periódicos).

Formato (Format 4.0, magnetostático, planar), seções usadas:
  [BdryProps] = n   blocos <BeginBdry> ... <BdryType> ... <A_0>,<A_1>,<A_2>,<Phi> ... <EndBdry>
                    (BdryType 0 = A prescrito; 4 = periódico; 5 = antiperiódico; 6/7 = air gap)
  [NumPoints] = n   x y ...
  [NumSegments] = n p0 p1 MaxSideLength BoundaryMarker ...   (marcador 1-based; 0 = nenhum)
  [NumArcSegments] = n  p0 p1 ArcLength[°] MaxSideAngle BoundaryMarker ...
  [Solution]        N; N linhas "x y A ..."; M; M linhas de elemento; n_rótulos; n_rótulos
                    linhas; n_pbc; n_pbc linhas "n0 n1 tipo" (0 periódico, 1 antiperiódico);
                    n_airgap (0 aqui).
"""
import numpy as np


# --------------------------------------------------------------------------- leitura
def _strip(lines):
    return [l.rstrip('\r\n') for l in lines]


def _bdry_props(L):
    """[(nome, tipo, A0, A1, A2, Phi)] na ordem de [BdryProps]."""
    i = next(k for k, l in enumerate(L) if l.startswith('[BdryProps]'))
    n, out, cur = int(L[i].split('=')[1]), [], None
    for l in L[i + 1:]:
        s = l.strip()
        if s.startswith('<BeginBdry>'):
            cur = {}
        elif s.startswith('<EndBdry>'):
            out.append(cur)
            if len(out) == n:
                break
        elif cur is not None and s.startswith('<') and '=' in s:
            k, v = s.split('=', 1)
            cur[k.strip()[1:-1]] = v.strip().strip('"')
    return out


def _section(L, tag):
    i = next(k for k, l in enumerate(L) if l.startswith(tag))
    n = int(L[i].split('=')[1])
    return [l.split() for l in L[i + 1:i + 1 + n]]


def solution_layout(L):
    """Índices das partes de [Solution]: (i_nodes, N, i_elems, M, i_pbc, n_pbc)."""
    i = L.index('[Solution]')
    N = int(L[i + 1]); i_nodes = i + 2
    j = i_nodes + N; M = int(L[j]); i_elems = j + 1
    k = i_elems + M; nb = int(L[k])
    p = k + 1 + nb; n_pbc = int(L[p])
    return i_nodes, N, i_elems, M, p + 1, n_pbc


def _arc_center(p0, p1, ang_deg):
    """Centro do arco de p0 a p1, anti-horário, ângulo ang_deg (convenção do FEMM)."""
    a = np.deg2rad(ang_deg)
    c = (p0 + p1) / 2; d = p1 - p0; L = np.hypot(*d)
    h = (L / 2) / np.tan(a / 2)                 # distância do meio da corda ao centro
    nrm = np.array([-d[1], d[0]]) / L           # normal à esquerda (anti-horário)
    return c + h * nrm, (L / 2) / np.sin(a / 2)


def ans_constraints(lines, nodes, tol_rel=1e-9):
    """Restrições do .ans -> dict:
      dirichlet : índices dos nós com A prescrito (só A = 0 é suportado; outro valor = erro)
      pbc       : [n_pbc, 3] (n0, n1, tipo 0 periódico / 1 antiperiódico) da [Solution]
      root, sign: classe de cada nó (união dos pares) e sinal relativo à raiz (A_i = sign·A_root)
      zero      : máscara [N] dos nós cuja classe contém um nó Dirichlet (A = 0 na classe toda)
      bdry      : nomes/tipos das fronteiras (para o relatório)."""
    L = _strip(lines)
    bp = _bdry_props(L)
    dir_ids = []
    for m, b in enumerate(bp):
        if int(b['BdryType']) == 0:
            vals = [float(b.get(k, 0)) for k in ('A_0', 'A_1', 'A_2', 'Phi')]
            if any(v != 0 for v in vals):
                raise NotImplementedError(f'fronteira {b["BdryName"]!r}: A prescrito ≠ 0 ({vals})')
            dir_ids.append(m + 1)               # marcador 1-based
        elif int(b['BdryType']) not in (4, 5):
            raise NotImplementedError(f'fronteira {b["BdryName"]!r}: BdryType {b["BdryType"]} não suportado')
    pts = np.array([[float(v) for v in f[:2]] for f in _section(L, '[NumPoints]')])
    xy = np.asarray(nodes)[:, :2].astype(np.float64)
    scale = np.abs(xy).max()
    on = np.zeros(len(xy), bool)
    for f in _section(L, '[NumSegments]'):
        if int(f[3]) in dir_ids:
            p0, p1 = pts[int(f[0])], pts[int(f[1])]; d = p1 - p0; l2 = d @ d
            t = ((xy - p0) @ d) / l2
            dist = np.abs((xy[:, 0] - p0[0]) * d[1] - (xy[:, 1] - p0[1]) * d[0]) / np.sqrt(l2)
            on |= (dist < tol_rel * scale) & (t > -tol_rel) & (t < 1 + tol_rel)
    for f in _section(L, '[NumArcSegments]'):
        if int(f[4]) in dir_ids:
            p0, p1, ang = pts[int(f[0])], pts[int(f[1])], float(f[2])
            c, R = _arc_center(p0, p1, ang)
            v = xy - c
            dr = np.abs(np.hypot(v[:, 0], v[:, 1]) - R)
            a0 = np.arctan2(*(p0 - c)[::-1])
            da = (np.arctan2(v[:, 1], v[:, 0]) - a0) % (2 * np.pi)
            on |= (dr < tol_rel * scale) & ((da <= np.deg2rad(ang) + 1e-9) | (da >= 2 * np.pi - 1e-9))
    i_nodes, N, _, _, i_pbc, n_pbc = solution_layout(L)
    pbc = np.array([[int(v) for v in L[i_pbc + q].split()[:3]] for q in range(n_pbc)], int).reshape(-1, 3)
    # união com sinal: A_i = sign_i · A_root(i)
    parent = np.arange(N); sgn = np.ones(N)

    def find(i):
        s = 1.0
        while parent[i] != i:
            s *= sgn[i]; i = parent[i]
        return i, s
    for n0, n1, t in pbc:
        if t not in (0, 1):
            raise NotImplementedError(f'par periódico de tipo {t}')
        r0, s0 = find(n0); r1, s1 = find(n1); s = 1.0 if t == 0 else -1.0     # A_n1 = s·A_n0
        if r0 == r1:
            if not np.isclose(s1, s * s0):
                raise ValueError(f'pares periódicos inconsistentes em ({n0}, {n1})')
            continue
        parent[r1] = r0; sgn[r1] = s * s0 * s1   # A_r1 = s1·A_n1 = s1·s·s0·A_r0
    root = np.empty(N, int); sign = np.empty(N)
    for i in range(N):
        root[i], sign[i] = find(i)
    zero_root = np.zeros(N, bool); zero_root[root[on]] = True
    return dict(dirichlet=np.flatnonzero(on), pbc=pbc, root=root, sign=sign, zero=zero_root[root],
                bdry=[(b['BdryName'], int(b['BdryType'])) for b in bp])
