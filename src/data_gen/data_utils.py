import os, shutil, math, gc
import pandas as pd
import numpy as np
from pathlib import Path
from src.configs.datagen import DatagenConfig
DATASET = DatagenConfig().dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR   = _PROJECT_ROOT / "data" / "raw" / DATASET
_TEMP_DIR  = _PROJECT_ROOT / "data" / "temp"
_TORCH_DIR = _PROJECT_ROOT / "data" / "torch"

def match_data_to_depth(depth_1, data_1, new_depth):
    """
    Map data from (depth_1, data_1) to new_depth.

    Encoding:
    ---------
    - depth_* is a 1D array of leaf depths.
    - Refining a leaf at depth d replaces that entry by 4 entries with value d+1.
    - new_depth is a refinement of depth_1 (never coarser).

    For regions where new_depth is finer, the original data value is copied
    to all new leaves in that region.

    Returns
    -------
    new_data : np.ndarray
        Data array aligned with new_depth.
    """
    d_old = np.asarray(depth_1, dtype=np.int64).ravel()
    d_new = np.asarray(new_depth, dtype=np.int64).ravel()
    data_old = np.asarray(data_1)
    n_old, n_new = len(d_old), len(d_new)
    out = []

    def fill_region(i_new: int, level: int, val) -> int:
        """
        Walk one region in new_depth starting at i_new and level,
        appending 'val' for each leaf under that region.
        Returns the new index in d_new after this region.
        """
        if i_new >= n_new:
            raise ValueError("Unexpected end of new_depth in fill_region.")
        v = int(d_new[i_new])
        if v < level:
            raise ValueError("Invalid encoding in fill_region.")
        if v == level:
            # leaf at this level
            out.append(val)
            return i_new + 1
        # v > level: region refined -> 4 children at level+1
        for _ in range(4):
            i_new = fill_region(i_new, level + 1, val)
        return i_new

    def map_region(i_old: int, i_new: int, level: int) -> tuple[int, int]:
        """
        Map one region (same physical region) between depth_1 and new_depth.
        Returns updated indices (i_old, i_new).
        """
        if i_old >= n_old or i_new >= n_new:
            raise ValueError("Unexpected end in map_region.")
        v_old = int(d_old[i_old])
        v_new = int(d_new[i_new])
        if v_old < level or v_new < level:
            raise ValueError("Invalid encoding in map_region.")

        old_leaf = (v_old == level)
        new_leaf = (v_new == level)

        # Case 1: both leaves -> single value
        if old_leaf and new_leaf:
            out.append(data_old[i_old])
            return i_old + 1, i_new + 1

        # Case 2: old is leaf, new is refined -> copy value to all new leaves
        if old_leaf and not new_leaf:
            val = data_old[i_old]
            i_new2 = fill_region(i_new, level, val)
            return i_old + 1, i_new2

        # Case 3: old refined, new leaf -> not allowed (new would be coarser)
        if not old_leaf and new_leaf:
            raise ValueError("new_depth is coarser than depth_1 in some region.")

        # Case 4: both refined -> recurse on 4 children
        for _ in range(4):
            i_old, i_new = map_region(i_old, i_new, level + 1)
        return i_old, i_new

    i_old = i_new = 0
    while i_old < n_old and i_new < n_new:
        i_old, i_new = map_region(i_old, i_new, level=0)

    if i_old != n_old or i_new != n_new:
        raise ValueError(
            f"Arrays not fully consumed: i_old={i_old}/{n_old}, i_new={i_new}/{n_new}"
        )

    return np.asarray(out, dtype=data_old.dtype)

def combine_grids(depth_1: np.ndarray, depth_2: np.ndarray) -> np.ndarray:
    """
    Combine the structure of two quadtrees into the minimal common grid.

    Encoding convention:
    --------------------
    - depth_* is a 1D array of leaf depths.
    - Refining a leaf at depth d replaces that element by 4 elements with value d+1.
    - Both trees share the same base partition (same sequence of base cells).

    Parameters
    ----------
    depth_1, depth_2 : np.ndarray
        Quadtree encodings as described above.

    Returns
    -------
    combined_depth : np.ndarray
        Depth encoding of the minimal common grid (union of refinements).
    """
    d1 = np.asarray(depth_1, dtype=np.int64).ravel()
    d2 = np.asarray(depth_2, dtype=np.int64).ravel()
    n1, n2 = len(d1), len(d2)
    out: list[int] = []

    def combine_region(i1: int, i2: int, level: int) -> tuple[int, int]:
        """
        Aux function.
        Combines one region (for both trees) at a given level,
        starting at indices i1, i2 in d1, d2.

        Appends the encoding of the combined region into `out`
        and returns the new indices (i1, i2) after consuming that region.
        """
        def copy_region(depth: np.ndarray, i: int, level: int) -> int:
            """
            Copy the encoding of one region from `depth` starting at index i
            for the given level, appending to `out`. Returns new index.
            """
            if i >= len(depth):
                raise ValueError("Unexpected end of depth array in copy_region.")
            v = int(depth[i])
            if v < level:
                raise ValueError(f"Invalid encoding: value {v} < level {level}.")
            if v == level:
                # Leaf at this level.
                out.append(level)
                return i + 1
            # v > level: this region is refined -> 4 children at level+1
            for _ in range(4):
                i = copy_region(depth, i, level + 1)
            return i

        # ---- combine_region logic ----
        if i1 >= n1 or i2 >= n2:
            raise ValueError("Unexpected end of depth arrays in combine_region.")

        v1 = int(d1[i1])
        v2 = int(d2[i2])
        if v1 < level or v2 < level:
            raise ValueError(
                f"Invalid encoding at level {level}: v1={v1}, v2={v2}"
            )

        # Case 1: both are leaves at this level
        if v1 == level and v2 == level:
            out.append(level)
            return i1 + 1, i2 + 1

        # Case 2: d1 leaf, d2 refined -> copy refinement of d2
        if v1 == level and v2 > level:
            i2_new = copy_region(d2, i2, level)
            return i1 + 1, i2_new

        # Case 3: d2 leaf, d1 refined -> copy refinement of d1
        if v2 == level and v1 > level:
            i1_new = copy_region(d1, i1, level)
            return i1_new, i2 + 1

        # Case 4: both refined -> combine 4 children
        for _ in range(4):
            i1, i2 = combine_region(i1, i2, level + 1)
        return i1, i2

    # Walk all base cells (each call to combine_region handles one base cell)
    i1 = i2 = 0
    while i1 < n1 and i2 < n2:
        i1, i2 = combine_region(i1, i2, level=0)

    # Both encodings must be fully consumed
    if i1 != n1 or i2 != n2:
        raise ValueError(
            f"Depth arrays not fully consumed: i1={i1}/{n1}, i2={i2}/{n2}"
        )

    return np.asarray(out, dtype=np.int64)

def _consume_region(depth: np.ndarray, i: int, level: int) -> int:
    """Avança o cursor sobre todas as folhas de uma região no nível dado."""
    if int(depth[i]) == level:
        return i + 1
    for _ in range(4):
        i = _consume_region(depth, i, level + 1)
    return i

def compute_cells(depth: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Para cada folha no stream de depth unificado, retorna o índice (0..H*W-1)
    da célula base correspondente (ordem row-major: ir*W + ia).

    Pressupõe que o stream foi gerado por DFS em ordem row-major sobre a
    grade base H×W — mesma ordem de _qtree_dfs em BLDC_Process.
    """
    depth = np.asarray(depth, dtype=np.int64).ravel()
    cells = np.empty(len(depth), dtype=np.int64)
    i = 0
    for base_idx in range(H * W):
        i_start = i
        i = _consume_region(depth, i, level=0)
        cells[i_start:i] = base_idx
    if i != len(depth):
        raise ValueError(f"depth não consumido completamente: {i}/{len(depth)}")
    return cells

# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def compute_leaf_boxes(depth: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Para cada folha no stream depth (pré-ordem DFS, row-major sobre H×W),
    retorna sua caixa física (r0, c0, r1, c1) em coordenadas full-res
    inteiras (grade H·2^D_max × W·2^D_max).

    depth : [S] int
    Returns: [S, 4] int64
    """
    depth  = np.asarray(depth, dtype=np.int64).ravel()
    S      = len(depth)
    D_max  = int(depth.max()) if S > 0 else 0
    scale  = 1 << D_max

    boxes  = np.empty((S, 4), dtype=np.int64)
    cursor = [0]

    def _fill(r0: int, c0: int, side: int, level: int):
        i = cursor[0]
        if depth[i] == level:
            boxes[i] = (r0, c0, r0 + side, c0 + side)
            cursor[0] += 1
        else:
            half = side >> 1
            _fill(r0,        c0,        half, level + 1)  # TL
            _fill(r0,        c0 + half, half, level + 1)  # TR
            _fill(r0 + half, c0,        half, level + 1)  # BL
            _fill(r0 + half, c0 + half, half, level + 1)  # BR

    for r in range(H):
        for c in range(W):
            _fill(r * scale, c * scale, scale, 0)

    return boxes


def build_graph_edges(
    boxes: np.ndarray,
    H: int,
    W: int,
    D_max: int,
) -> tuple:
    """
    Constrói edge_index e edge_attr para adjacência física entre folhas.

    Regra: dois nós conectados ↔ regiões compartilham fronteira de comprimento
    positivo (segmento 1D). Contato apenas em ponto não gera aresta.
    Dimensão angular (colunas) é periódica. Radial (linhas) não.

    boxes  : [S, 4] int — (r0, c0, r1, c1) em coordenadas full-res
    H, W   : grade base
    D_max  : depth.max()

    Returns
    -------
    edge_index : [2, E] torch.long    bidirecional (i→j e j→i)
    edge_attr  : [E, 4] torch.float32 [Δr, Δc, shared_length, center_dist]
                 Δr, Δc e shared_length em unidades de célula base.
    """
    from collections import defaultdict

    W_full = W * (1 << D_max)
    scale  = float(1 << D_max)

    by_c0 = defaultdict(list)
    by_r0 = defaultdict(list)
    for j, (r0, c0, r1, c1) in enumerate(boxes):
        by_c0[int(c0)].append(j)
        by_r0[int(r0)].append(j)

    srcs: list = []
    dsts: list = []
    attrs: list = []

    def _add(i: int, j: int, dr_phys: float, dc_phys: float, shared_phys: float):
        dr   = dr_phys / scale
        dc   = dc_phys / scale
        sh   = shared_phys / scale
        dist = float(np.sqrt(dr * dr + dc * dc))
        srcs.append(i);  dsts.append(j);  attrs.append([ dr,  dc, sh, dist])
        srcs.append(j);  dsts.append(i);  attrs.append([-dr, -dc, sh, dist])

    for i, (r0_i, c0_i, r1_i, c1_i) in enumerate(boxes):
        side_i = int(r1_i - r0_i)
        cr_i   = (r0_i + r1_i) * 0.5
        cc_i   = (c0_i + c1_i) * 0.5

        # --- vizinhos à direita (wrap angular) ---
        c_right = int(c1_i) % W_full
        for j in by_c0.get(c_right, []):
            r0_j, c0_j, r1_j, c1_j = boxes[j]
            if min(r1_i, r1_j) - max(r0_i, r0_j) <= 0:
                continue
            cr_j = (r0_j + r1_j) * 0.5
            cc_j = (c0_j + c1_j) * 0.5
            dr_phys = cr_j - cr_i
            # vizinho wrapeado está em c0_j=0 mas fisicamente à direita de c1_i=W_full
            dc_phys = (cc_j + W_full - cc_i) if (c0_j == 0 and c1_i == W_full) else (cc_j - cc_i)
            _add(i, j, dr_phys, dc_phys, float(min(side_i, int(r1_j - r0_j))))

        # --- vizinhos abaixo (sem wrap radial) ---
        for j in by_r0.get(int(r1_i), []):
            r0_j, c0_j, r1_j, c1_j = boxes[j]
            if min(c1_i, c1_j) - max(c0_i, c0_j) <= 0:
                continue
            cr_j = (r0_j + r1_j) * 0.5
            cc_j = (c0_j + c1_j) * 0.5
            dr_phys = cr_j - cr_i
            dc_phys = cc_j - cc_i
            _add(i, j, dr_phys, dc_phys, float(min(side_i, int(r1_j - r0_j))))

    if not srcs:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, 4), dtype=np.float32))

    return (np.array([srcs, dsts], dtype=np.int64),
            np.array(attrs,        dtype=np.float32))


def qtree_to_lowres_hw(
    stream: np.ndarray,
    cells:  np.ndarray,
    depth:  np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Projeta stream quadtree na grade base H×W por média ponderada por área.

    Cada folha contribui com peso = (1/2^depth)^2 = 4^(-depth).
    Como o quadtree cobre exatamente 1 célula base por região,
    w_sum[cell] = 1.0 para toda célula com folhas.

    stream : [C, S] float  valores por canal e por folha
    cells  : [S]    int    índice de célula base (0..H*W-1)
    depth  : [S]    int    nível de refinamento

    Returns: [C, H, W] float32
    """
    C = stream.shape[0]
    S = stream.shape[1]

    areas = (4.0 ** (-depth.astype(np.float64)))  # [S]

    out   = np.zeros((C, H * W), dtype=np.float64)
    w_sum = np.zeros(H * W,      dtype=np.float64)

    for c in range(C):
        np.add.at(out[c], cells, stream[c].astype(np.float64) * areas)
    np.add.at(w_sum, cells, areas)

    mask = w_sum > 0
    out[:, mask] /= w_sum[mask]

    return out.reshape(C, H, W).astype(np.float32)


# ---------------------------------------------------------------------------
# Parser do motor — funções modulares reutilizáveis
# ---------------------------------------------------------------------------

# Mapeamento canônico mu_r → material_id (baseado em BLDC_Process.PERMEABILITY)
# Reutilizável por qualquer parser que leia Mu_r_qt_ do motor.
_MU_THRESHOLDS = [
    (10.0,  None, 0),    # iron_1008: mu_r = 5000 → id 0
    (1.01,  10.0, 2),    # magnet (N35p/N35n): mu_r = 1.05 → id 2
    (None,  0.9995, 3),  # copper: mu_r = 0.999 → id 3
]
# vacuum: mu_r = 1.0 → id 1 (default)

def derive_material_id(mu_r_arr: np.ndarray) -> np.ndarray:
    """Mapeia mu_r → material_id por limiar.

    Baseado nos valores exatos de BLDC_Process.PERMEABILITY:
        iron=0 (mu_r=5000), vacuum=1 (mu_r=1.0),
        magnet=2 (mu_r=1.05), copper=3 (mu_r=0.999)

    Genérico: reutilizável por qualquer parser que leia Mu_r_qt_ do motor.
    """
    mu  = np.asarray(mu_r_arr, dtype=np.float32)
    mat = np.ones(len(mu), dtype=np.int32)         # default: vacuum=1
    mat[mu > 10.0]                           = 0   # iron   (5000)
    mat[(mu > 1.01) & (mu < 10.0)]          = 2   # magnet (1.05)
    mat[mu < 0.9995]                         = 3   # copper (0.999)
    return mat


def build_node_x_motor(
    mu_r:     np.ndarray,           # [S]
    M:        np.ndarray,           # [S]
    frac_dom: np.ndarray,           # [S]
    depth:    np.ndarray,           # [S]
    cells:    np.ndarray,           # [S]
    H:        int,
    W:        int,
    cols:     list | None = None,   # índices das colunas a incluir; None = todas (0..8)
) -> np.ndarray:
    """Constrói node_x [S, len(cols)] para o parser do motor.

    Apenas as features listadas em `cols` são computadas — features caras
    (ex: normal_x/normal_y, futuro) só são calculadas se explicitamente pedidas.

    Referência de colunas (CLAUDE.md):
        0  material_id  — derivado de mu_r (ferro=0, ar=1, ima=2, bobina=3)
        1  mu_r         — permeabilidade (point query no centro)
        2  M            — magnetização   (point query no centro)
        3  cell_area    — (1/2^depth)²   área relativa à célula base
        4  r_base       — coord radial normalizada do centro da célula base [0,1]
        5  c_base       — coord angular normalizada do centro da célula base [0,1]
        6  frac_dom     — fração de área do material dominante no bounding box
        7  normal_x     — placeholder 0 (pós-processamento futuro; será custoso)
        8  normal_y     — placeholder 0 (pós-processamento futuro; será custoso)
    """
    if cols is None:
        cols = list(range(9))

    S      = len(depth)
    cache: dict = {}

    def _get(idx: int) -> np.ndarray:
        if idx in cache:
            return cache[idx]
        if   idx == 0: v = derive_material_id(mu_r).astype(np.float32)
        elif idx == 1: v = mu_r.astype(np.float32)
        elif idx == 2: v = M.astype(np.float32)
        elif idx == 3: v = (4.0 ** (-depth.astype(np.float64))).astype(np.float32)
        elif idx == 4: v = ((cells // W) + 0.5).astype(np.float32) / H
        elif idx == 5: v = ((cells %  W) + 0.5).astype(np.float32) / W
        elif idx == 6: v = frac_dom.astype(np.float32)
        elif idx == 7: v = np.zeros(S, dtype=np.float32)  # normal_x — placeholder
        elif idx == 8: v = np.zeros(S, dtype=np.float32)  # normal_y — placeholder
        else: raise ValueError(f"build_node_x_motor: índice inválido {idx}")
        cache[idx] = v
        return v

    return np.stack([_get(i) for i in cols], axis=1)   # [S, len(cols)]


def build_node_y_motor(Bx: np.ndarray, By: np.ndarray) -> np.ndarray:
    """Constrói node_y [S, 2] para o parser do motor.

    Colunas: [Bx, By] — campo radial e angular (T).
    """
    return np.stack([Bx.astype(np.float32), By.astype(np.float32)], axis=1)


def build_graph_edges_motor(
    boxes: np.ndarray,
    H:     int,
    W:     int,
    D_max: int,
    mu_r:  np.ndarray | None = None,   # [S] — necessário apenas se col 4 (delta_mu) pedida
    cols:  list | None       = None,   # subconjunto de [0,1,2,3,4]; None = todos
) -> tuple:
    """Constrói edge_index e edge_attr para o motor, com avaliação lazy por coluna.

    Colunas disponíveis em edge_attr:
        0  delta_r       diferença radial  j−i (unidades de célula base)
        1  delta_c       diferença angular j−i (com wrap; unidades de célula base)
        2  shared_length comprimento da fronteira compartilhada
        3  center_dist   distância euclidiana entre centros
        4  delta_mu      mu_r[i] − mu_r[j] (origem − destino) ← só computado se 4 ∈ cols

    NOTA (2026-07-17): delta_mu é direcional em "origem − destino" (i−j), o
    INVERSO da convenção "destino − origem" (j−i) usada em delta_r/delta_c.
    Escolha deliberada para uso pelo FNO_GNN_v2 (parser FNO_GNN_V2_PARSER):
    positivo quando a aresta sai de alta permeabilidade para baixa (ex:
    ferro→ar), negativo no sentido inverso. Antes de 2026-07-17 a fórmula era
    j−i (mesma convenção das demais colunas), mas nenhum parser registrado
    consumia a coluna 4 até então — sem impacto em runs já treinadas.

    Parâmetros
    ----------
    mu_r : obrigatório apenas quando 4 ∈ cols (ou cols=None).
           Se None e col 4 for pedida, levanta ValueError.
    cols : lista de índices a exportar.  None → todos ([0,1,2,3,4]).

    Retorna
    -------
    edge_index : [2, E] int64
    edge_attr  : [E, len(cols)] float32
    """
    _cols = list(range(5)) if cols is None else cols

    # colunas 0-3 vêm do builder genérico (custo inevitável — geometria de adjacência)
    edge_index, edge_attr_base = build_graph_edges(boxes, H, W, D_max)
    # edge_attr_base : [E, 4]

    need_delta_mu = (4 in _cols)

    if edge_index.shape[1] > 0:
        if need_delta_mu:
            if mu_r is None:
                raise ValueError(
                    "build_graph_edges_motor: col 4 (delta_mu) pedida mas mu_r=None. "
                    "Passe mu_r=[S] ou remova 4 de edge_attr_cols no parser."
                )
            srcs     = edge_index[0]
            dsts     = edge_index[1]
            # [REMOVIDO] delta_mu = (mu_r[dsts] - mu_r[srcs])  # j-i, mesma convenção
            # de delta_r/delta_c. Trocado por i-j (origem-destino) a pedido do
            # usuário (2026-07-17): positivo = aresta sai de alta permeabilidade
            # para baixa. Ver nota no docstring desta função.
            delta_mu = (mu_r[srcs] - mu_r[dsts]).reshape(-1, 1).astype(np.float32)
            full_attr = np.concatenate([edge_attr_base, delta_mu], axis=1)  # [E, 5]
        else:
            full_attr = edge_attr_base  # [E, 4] — delta_mu nunca alocado

        # fatia para as colunas pedidas (cols 0-3 sempre presentes em full_attr)
        geo_cols  = [c for c in _cols if c <= 3]
        has_4     = need_delta_mu
        col_idx   = geo_cols + ([4] if has_4 else [])
        edge_attr = full_attr[:, col_idx]
    else:
        edge_attr = np.zeros((0, len(_cols)), dtype=np.float32)

    return edge_index, edge_attr


# ---------------------------------------------------------------------------

def quadtree_to_full_res(data: np.ndarray,
                        depth: np.ndarray,
                        dim: tuple[int, int]) -> np.ndarray:
    """
    Reconstruct full-resolution image from a quadtree stream (NumPy).

    data  : [N_leaves] 1D array of leaf values (scalars), pre-order per cell
    depth : [N_leaves] 1D int array of leaf depths, same order as data
    dim   : (H, W) base grid size

    Output shape: (H*2^D, W*2^D), where D = depth.max()
    """
    H, W = int(dim[0]), int(dim[1])
    depth = np.asarray(depth, dtype=np.int64)
    data  = np.asarray(data)
    D = int(depth.max())
    S = 1 << D  # 2**D

    out = np.zeros((H * S, W * S), dtype=data.dtype)
    i = 0  # cursor over (data, depth)

    def fill_region(r0: int, c0: int, side: int, level: int):
        nonlocal i
        d = int(depth[i])
        v = data[i]
        if d == level:
            out[r0:r0 + side, c0:c0 + side] = v
            i += 1
        else:
            half = side // 2
            # TL, TR, BL, BR (pre-order)
            fill_region(r0,         c0,         half, level + 1)
            fill_region(r0,         c0 + half,  half, level + 1)
            fill_region(r0 + half,  c0,         half, level + 1)
            fill_region(r0 + half,  c0 + half,  half, level + 1)

    for r in range(H):
        for c in range(W):
            fill_region(r * S, c * S, S, level=0)

    return out

def generate_shapely_visualize(motor_params_list,
                       code_list, 
                       n_r, 
                       n_a,
                       ang_1 = 0, 
                       ang_2 = 120,
                       n_phases = 1):

    from src.data_gen.motor_model import BLDC_Process, BLDC_Shapely_Model
    import random
    import matplotlib.pyplot as plt
    
    #### phase ####
    ini = 0
    end = 360 / int(motor_params_list['number_rotor_poles']['value'][0]) * 1

    for code in code_list:
        
        phases = [random.uniform(ini, end) for _ in range(n_phases)]
        
        for i, phase in enumerate(phases):

            motor_params = BLDC_Process.extract_params_at_index(motor_params=motor_params_list,code=code)

            ######### shapely ###########
            
            model = BLDC_Shapely_Model(motor_params=motor_params, phase=phase)
            model.draw_motor()

            fig, ax = model.plot()
            plt.show()

# [REMOVIDO] generate_one_batch (grid) e generate_one_batch_qtree unificados em generate_one_batch(mode=...)

def generate_one_batch(mode,
                       motor_params_list,
                       code_list,
                       n_r,
                       n_a,
                       ang_1=0,
                       ang_2=120,
                       n_phases=1,
                       max_depth=1):
    """
    Gera e salva dados para uma lista de códigos de motor.

    mode : 'grid'  → grade regular (Bx/By + Material + Mu_avg + Magnetization)
           'qtree' → quadtree adaptativo (Bx/By/depth + Mag_M/depth + Mu_avg/depth)
    max_depth : profundidade máxima de refinamento (ignorado se mode='grid')
    """
    import femm
    from src.data_gen.motor_model import BLDC_Process, BLDC_FEMM_Model, BLDC_Shapely_Model
    import random

    if mode not in ('grid', 'qtree'):
        raise ValueError(f"mode deve ser 'grid' ou 'qtree', recebido: {mode!r}")

    # [REMOVIDO] ini, end, phases loop — phase do rotor fixado em 0
    # Motivo: ângulo do rotor não é parâmetro de design; variação aleatória causava
    # offset de geometria identificado como fonte de erro nas predições.
    # ini = 0
    # end = 2 * 360 / int(motor_params_list['number_rotor_poles']['value'][0])

    for code in code_list:

        # [REMOVIDO] phases = [random.uniform(ini, end) for _ in range(n_phases)]
        # [REMOVIDO] for i, phase in enumerate(phases):
        if 'rotor_phase' in motor_params_list:
            phase = float(motor_params_list['rotor_phase']['value'][code])
        else:
            phase = 0   # rotor sempre na posição canônica
        i = 0

        if True:    # bloco preservado para manter indentação do código abaixo
            tmp_dir = str(_TEMP_DIR / f"tmp_{code}")
            os.makedirs(tmp_dir, exist_ok=True)
            fem_file = os.path.join(tmp_dir, f"model_{code}.fem")
            cwd0 = os.getcwd()
            os.chdir(tmp_dir)

            motor_params = BLDC_Process.extract_params_at_index(motor_params=motor_params_list, code=code)
            lcl_code = f"{code}_p{i}"

            ######### Shapely ###########

            model_shapely = BLDC_Shapely_Model(motor_params=motor_params, phase=phase)
            model_shapely.draw_motor()

            if mode == 'grid':
                model_shapely.save_material_grid(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=lcl_code)
                depth_unified = None
            else:
                stator_od = motor_params['stator_outer_diameter']['value']
                rotor_id  = motor_params['rotor_inner_diameter']['value']
                # [REMOVIDO] save_magnetization_qtree (DFS independente) + combine_grids
                # Novo fluxo: um único DFS em save_material_mu_qtree gera depth_qt_;
                # M e B são amostrados nessa estrutura (sem DFS próprio).

                leaves = model_shapely.save_material_mu_qtree(
                    ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a,
                    max_depth=max_depth, code=lcl_code)

                model_shapely.save_magnetization_from_depth(
                    leaves, ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=lcl_code)

                depth_unified = np.array([c["d"] for c in leaves], dtype=np.int64)

            # Grade base — point queries diretos nos centros H×W (ambos os modos)
            model_shapely.save_grid_mu(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=lcl_code)

            ######### FEMM ###########

            model_femm = BLDC_FEMM_Model(motor_params=motor_params, phase=phase)

            femm.openfemm('bHide')
            femm.main_resize(1000, 1000)
            femm.newdocument(0)
            femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, 0, 200)

            model_femm.draw_motor()

            femm.mi_refreshview()
            femm.mi_zoomnatural()
            femm.mi_saveas(fem_file)
            femm.mi_createmesh()
            femm.mi_analyze()
            femm.mi_loadsolution()

            if mode == 'grid':
                pass   # grade base coberta por save_B_grid_pts abaixo
            else:
                model_femm.save_B_grid_qtree_from_depth(depth_unified, ang_1=ang_1, ang_2=ang_2,
                                                        n_r=n_r, n_a=n_a, code=lcl_code)

            # Grade base — point queries FEMM nos centros H×W (ambos os modos)
            model_femm.save_B_grid_pts(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=lcl_code)

            femm.closefemm()

            os.chdir(cwd0)
            shutil.rmtree(tmp_dir, ignore_errors=True)

_PREFIXES = {
    # [REMOVIDO] "Magnetization_", "Mu_avg_" — usavam média por disco; abolidas
    'grid':  ["Material_",
              "Mu_r_grid_", "Mag_M_grid_", "Mag_Bx_grid_", "Mag_By_grid_", "Mag_A_grid_"],
    # [REMOVIDO] Mu_depth_qt_, Mag_M_depth_qt_, Mag_B_depth_qt_ — unificados em depth_qt_
    'qtree': ["Mag_Bx_qt_", "Mag_By_qt_", "Mag_A_qt_",
              "Mag_M_qt_",
              "Mu_r_qt_", "depth_qt_", "Frac_dom_qt_",
              # grade base — point queries diretos H×W, sem averaging qtree
              "Mu_r_grid_", "Mag_M_grid_", "Mag_Bx_grid_", "Mag_By_grid_", "Mag_A_grid_"],
}

def check_data(mode, n_phases=1):
    """
    Retorna lista de índices de modelos com arquivos faltando.

    mode : 'grid' | 'qtree'
    """
    if mode not in _PREFIXES:
        raise ValueError(f"mode deve ser 'grid' ou 'qtree', recebido: {mode!r}")

    prefixes = _PREFIXES[mode]
    models = pd.read_csv(_RAW_DIR / "valid_designs.csv", header=None).values
    n_models = len(models) - 1
    ext = ".csv"

    missing_models = set()

    for m in range(n_models):
        for p in range(n_phases):
            for pref in prefixes:
                fname = f"{pref}{m}_p{p}{ext}"
                if not (_RAW_DIR / fname).exists():
                    missing_models.add(m)
                    break
            if m in missing_models:
                break

    return list(missing_models)

# ---------------------------------------------------------------------------
# Mapeamento padrão de canais para o dataset quadtree deste projeto.
# Cada entrada: (prefixo_dado, prefixo_depth)
# ---------------------------------------------------------------------------
_X_CHANNELS = [
    # [REMOVIDO] prefixos de depth individuais — todos usam o depth_qt_ unificado
    ('Mu_r_qt_',  'depth_qt_'),
    ('Mag_M_qt_', 'depth_qt_'),
]
_Y_CHANNELS = [
    ('Mag_Bx_qt_', 'depth_qt_'),
    ('Mag_By_qt_', 'depth_qt_'),
]

def _read_grid_hw(raw_dir: Path, code: str, H: int, W: int):
    """Lê os 4 CSVs de point query (grade H×W) e retorna (x_hw, y_hw) float32.

    x_hw [2, H, W] — canal 0: Mu_r, canal 1: M
    y_hw [2, H, W] — canal 0: Bx,   canal 1: By
    """
    def _r(prefix):
        return (pd.read_csv(raw_dir / f"{prefix}{code}.csv", header=None)
                .values.ravel().astype(np.float32).reshape(H, W))

    x_hw = np.stack([_r('Mu_r_grid_'),   _r('Mag_M_grid_')],   axis=0)
    y_hw = np.stack([_r('Mag_Bx_grid_'), _r('Mag_By_grid_')],  axis=0)
    return x_hw, y_hw


class QtreeSampleUnifier:
    """Parser de amostras quadtree do motor BLDC — lê CSVs raw e monta tensores.

    Lê os CSVs gerados por generate_one_batch (mode='qtree') e retorna um dict
    por amostra (model_idx, phase) pronto para save_qtree_chunks.

    Fluxo por amostra:
      1. Lê depth_qt_ (depth unificado único — gerado por save_material_mu_qtree)
      2. Lê Mu_r_qt_, Mag_M_qt_, Frac_dom_qt_, Mag_Bx_qt_, Mag_By_qt_
      3. build_node_x_motor  → node_x [S, len(node_x_cols)] — sem parser: [S, 9]; FNO_GNN_PARSER: [S, 5]
      4. build_node_y_motor  → node_y [S, 2]
      5. build_graph_edges_motor → edge_index [2,E] + edge_attr [E, len(edge_attr_cols)] — FNO_GNN_PARSER: [E, 4]
      6. qtree_to_lowres_hw  → x_hw [2,H,W]  y_hw [2,H,W]

    Cada item retorna dict com chaves:
      x          [1, 2, S]   stream de entrada  (mu_r, M)
      y          [1, 2, S]   stream de saída    (Bx, By)
      depth      [S]         depth unificado
      cells      [S]         índice de célula base (0..H*W-1)
      L          [1]         número de nós
      dim        (H, W)
      node_x     [S, N_x]    features de nó — sem parser: [S,9]; FNO_GNN_PARSER: [S,5] (mu_r,M,cell_area,r_base,c_base)
      node_y     [S, 2]      targets de nó
      edge_index [2, E]
      edge_attr  [E, N_ea]   features de aresta — sem parser: [E,5]; FNO_GNN_PARSER: [E,4] (Δr,Δc,shared_length,center_dist)
      x_hw       [2, H, W]   grade Mu_r/M via média ponderada por área (qtree_to_lowres_hw)
      y_hw       [2, H, W]   grade Bx/By  via média ponderada por área
      x_hw_grid  [2, H, W]   grade Mu_r/M via point query no centro de cada célula base (sem averaging)
      y_hw_grid  [2, H, W]   grade Bx/By  via point query (fronteiras de material nítidas)

    """
    def __init__(self, n_r: int, n_a: int, n_phases: int = 1, raw_dir=None,
                 parser_cfg=None):
        """
        parser_cfg : MotorQtreeParserConfig opcional.
            Se fornecido, __getitem__ pula computações desnecessárias
            (ex: grafo não é construído se parser_cfg.build_graph=False)
            e aplica a seleção de features antes de retornar.
            Se None, retorna o dict completo (node_x [S,9], grafo, etc.).
        """
        self.n_r        = n_r
        self.n_a        = n_a
        self.raw_dir    = Path(raw_dir) if raw_dir is not None else _RAW_DIR
        self.parser_cfg = parser_cfg

        models   = pd.read_csv(self.raw_dir / 'valid_designs.csv', header=None).values
        n_models = len(models) - 1
        self.samples = [(m, p) for m in range(n_models) for p in range(n_phases)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        model_idx, phase = self.samples[idx]
        code = f"{model_idx}_p{phase}"
        H, W = self.n_r, self.n_a

        def _read(prefix):
            return pd.read_csv(
                self.raw_dir / f"{prefix}{code}.csv", header=None
            ).values.ravel()

        # ── depth unificado (único DFS, gerado por save_material_mu_qtree) ──
        depth = _read('depth_qt_').astype(np.int64)
        cells = compute_cells(depth, H, W)
        S     = len(depth)

        # ── campos por nó ────────────────────────────────────────────────────
        mu_r     = _read('Mu_r_qt_').astype(np.float32)
        M        = _read('Mag_M_qt_').astype(np.float32)
        frac_dom = _read('Frac_dom_qt_').astype(np.float32)
        Bx       = _read('Mag_Bx_qt_').astype(np.float32)
        By       = _read('Mag_By_qt_').astype(np.float32)

        # ── features / targets de nó ─────────────────────────────────────────
        # passa cols ao builder para avaliação lazy (apenas features pedidas pelo parser)
        _node_x_cols = self.parser_cfg.node_x_cols if self.parser_cfg is not None else None
        node_x = build_node_x_motor(mu_r, M, frac_dom, depth, cells, H, W,
                                     cols=_node_x_cols)    # [S, len(cols)] — FNO_GNN_PARSER: [S, 5]
        node_y = build_node_y_motor(Bx, By)                # [S, 2]

        # ── grade baixa resolução ─────────────────────────────────────────────
        x_stream = np.stack([mu_r, M], axis=0)   # [2, S]
        y_stream = np.stack([Bx,   By], axis=0)  # [2, S]
        x_hw = qtree_to_lowres_hw(x_stream, cells, depth, H, W)   # [2, H, W]
        y_hw = qtree_to_lowres_hw(y_stream, cells, depth, H, W)   # [2, H, W]

        # ── grade base sem averaging (point queries nos centros H×W) ─────────
        x_hw_grid, y_hw_grid = _read_grid_hw(self.raw_dir, code, H, W)

        # ── grafo — omitido se parser não precisar ────────────────────────────
        need_graph = (self.parser_cfg is None) or self.parser_cfg.build_graph
        if need_graph:
            _edge_cols = (self.parser_cfg.edge_attr_cols
                          if self.parser_cfg is not None else None)
            # mu_r só passado se col 4 (delta_mu) for necessária
            _need_delta_mu = (_edge_cols is None) or (4 in _edge_cols)
            _mu_r_for_graph = mu_r if _need_delta_mu else None

            D_max = int(depth.max()) if S > 0 else 0
            boxes = compute_leaf_boxes(depth, H, W)
            edge_index, edge_attr = build_graph_edges_motor(
                boxes, H, W, D_max,
                mu_r=_mu_r_for_graph,
                cols=_edge_cols,
            )
        else:
            _n_edge_cols = len(self.parser_cfg.edge_attr_cols) if self.parser_cfg is not None else 5
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr  = np.zeros((0, _n_edge_cols), dtype=np.float32)

        sample = {
            'x':          x_stream[np.newaxis].astype(np.float32),  # [1, 2, S]
            'y':          y_stream[np.newaxis].astype(np.float32),  # [1, 2, S]
            'depth':      depth,                                      # [S]
            'cells':      cells,                                      # [S]
            'L':          np.array([S], dtype=np.int64),             # [1]
            'dim':        (H, W),
            'node_x':     node_x,                                    # [S, len(cols)] — FNO_GNN_PARSER: [S, 5]
            'node_y':     node_y,                                    # [S, 2]
            'edge_index': edge_index,                                # [2, E]
            'edge_attr':  edge_attr,                                 # [E, len(cols)] — FNO_GNN_PARSER: [E, 4]
            'x_hw':       x_hw,                                      # [2, H, W]
            'y_hw':       y_hw,                                      # [2, H, W]
            'x_hw_grid':  x_hw_grid,                                 # [2, H, W]
            'y_hw_grid':  y_hw_grid,                                 # [2, H, W]
        }

        # aplica seleção de features se parser foi fornecido
        # node_x já construído com apenas as colunas pedidas (build_node_x_motor lazy);
        # filtramos aqui node_y / x_hw / y_hw e removemos grafo se necessário.
        if self.parser_cfg is not None:
            cfg = self.parser_cfg
            sample['node_y']    = sample['node_y'][:, cfg.node_y_cols]
            sample['x_hw']      = sample['x_hw'][cfg.x_hw_cols]
            sample['y_hw']      = sample['y_hw'][cfg.y_hw_cols]
            sample['x_hw_grid'] = sample['x_hw_grid'][cfg.x_hw_cols]
            sample['y_hw_grid'] = sample['y_hw_grid'][cfg.y_hw_cols]
            if not cfg.build_graph:
                sample.pop('edge_index', None)
                sample.pop('edge_attr',  None)

        return sample


class GridSampleUnifier:
    """Parser de amostras grid do motor BLDC — lê apenas CSVs de point query H×W.

    Para arquiteturas com mode='grid' (FNO2d, MaskedFNO2d).
    Não requer arquivos qtree.

    Retorna dict com:
      x_hw  [Cx, H, W]  canais selecionados pelo parser (padrão: Mu_r, M)
      y_hw  [Cy, H, W]  canais selecionados pelo parser (padrão: Bx, By)
      dim   (H, W)
    """

    def __init__(self, n_r: int, n_a: int, n_phases: int = 1, raw_dir=None,
                 parser_cfg=None):
        self.n_r        = n_r
        self.n_a        = n_a
        self.raw_dir    = Path(raw_dir) if raw_dir is not None else _RAW_DIR
        self.parser_cfg = parser_cfg

        models   = pd.read_csv(self.raw_dir / 'valid_designs.csv', header=None).values
        n_models = len(models) - 1
        self.samples = [(m, p) for m in range(n_models) for p in range(n_phases)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        model_idx, phase = self.samples[idx]
        code = f"{model_idx}_p{phase}"
        H, W = self.n_r, self.n_a

        x_hw, y_hw = _read_grid_hw(self.raw_dir, code, H, W)

        if self.parser_cfg is not None:
            cfg  = self.parser_cfg
            x_hw = x_hw[cfg.x_hw_cols]
            y_hw = y_hw[cfg.y_hw_cols]

        return {'x_hw': x_hw, 'y_hw': y_hw, 'dim': (H, W)}


def save_qtree_chunks(unifier,
                      chunk_size: int = 32,
                      stream_dir=None,
                      graph_dir=None,
                      parser_cfg=None) -> int:
    # [REMOVIDO] — substituído por build_data_chunks.py (pipeline NPZ).
    # Stream chunks (x, y, depth, cells) e quad_chunks abolidos;
    # o novo pipeline gera data_chunk_*.pt em data/torch/data_chunks/.
    return 0