import femm
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import pandas as pd
import random
from pathlib import Path
from src.configs.datagen import DatagenConfig
_dg = DatagenConfig()
DATASET                = _dg.dataset
DISTRIBUTION           = _dg.distribution
CASCADE_BUFFER         = _dg.cascade_buffer
HOMOGENEITY_THRESHOLD  = _dg.homogeneity_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / DATASET
DATA_DIR.mkdir(parents=True, exist_ok=True)

class BLDC_Process:

    # Propriedades de materiais — fonte única para todas as subclasses
    MAGNETIZATION = {'iron_1008': 0,      'N35p': 1,    'N35n': -1,  'copper': 0,     'vacuum': 0  }
    PERMEABILITY  = {'iron_1008': 5000.0, 'N35p': 1.05, 'N35n': 1.05,'copper': 0.999, 'vacuum': 1.0}
    # [REMOVIDO] MATERIAL_ID = {'iron_1008': 6, 'N35p': 1, 'N35n': 5, 'copper': 3, 'vacuum': 0}
    # N35p e N35n unificados como ima=2; ferro=0, ar=1, bobina=3
    MATERIAL_ID   = {'iron_1008': 0,      'N35p': 2,    'N35n': 2,   'copper': 3,     'vacuum': 1  }

    # Limites de fração de área para critério de refinamento quadtree
    REFINEMENT_LOW  = 0.2
    REFINEMENT_HIGH = 0.8

    def __init__(self, motor_params = {}):
        self.outer_diameter        = motor_params['outer_diameter']['value']
        self.inner_diameter        = motor_params['inner_diameter']['value']
        self.number_rotor_poles    = int(motor_params['number_rotor_poles']['value'])
        self.number_stator_slots   = int(motor_params['number_stator_slots']['value'])
        self.stator_outer_diameter = motor_params['stator_outer_diameter']['value']
        self.stator_inner_diameter = motor_params['stator_inner_diameter']['value']
        self.rotor_outer_diameter  = motor_params['rotor_outer_diameter']['value']
        self.rotor_inner_diameter  = motor_params['rotor_inner_diameter']['value']
        self.stack_length          = motor_params['stack_length']['value']
        self.slot_Hs0              = motor_params['slot_Hs0']['value']
        self.slot_Hs1              = motor_params['slot_Hs1']['value']
        self.slot_Hs2              = motor_params['slot_Hs2']['value']
        self.slot_Bs0              = motor_params['slot_Bs0']['value']
        self.slot_Bs1              = motor_params['slot_Bs1']['value']
        self.slot_Bs2              = motor_params['slot_Bs2']['value']
        self.slot_Rs               = motor_params['slot_Rs']['value']
        self.pole_embrance         = motor_params['pole_embrance']['value']
        self.pole_thickness        = motor_params['pole_thickness']['value']

        self.material_iron ='iron_1008'
        self.material_mag = 'N35'
        self.material_copper= 'copper'
        self.material_gap = 'vacuum'

        self.geometries = {
            'vacuum': [],
            'iron_1008': [],
            'N35p': [],
            'N35n': [],
            'copper': []
        }

        self.slotoppener_ang = np.arccos(
            (2 * (self.stator_outer_diameter / 2 - self.slot_Hs0) ** 2 - self.slot_Bs0 ** 2)
            / (2 * (self.stator_outer_diameter / 2 - self.slot_Hs0) ** 2)
        )

    def _iter_polar_points(self, r_in, r_ext, ang_1, ang_2, n_r, n_a):

        dr  = (r_ext - r_in) / n_r
        da  = (ang_2 - ang_1) / n_a

        r_vals = r_in  + (np.arange(n_r) + 0.5) * dr
        a_vals = ang_1 + (np.arange(n_a) + 0.5) * da

        for r in r_vals:
            for th in a_vals:
                x = r * np.cos(th)
                y = r * np.sin(th)
                yield (x, y, r, th)
    
    @staticmethod
    def save_csv(Fields=[],names=[], data_path=None):
        save_dir = Path(data_path) if data_path else DATA_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        for i in range(len(Fields)):
            df = pd.DataFrame(Fields[i])
            df.to_csv(save_dir / names[i], index=False, header=False)

    # ------------------------------------------------------------------
    # Helpers compartilhados entre FEMM e Shapely
    # ------------------------------------------------------------------

    def _coil_coords(self):
        """Coordenadas locais (não rotacionadas) de uma bobina.
        Retorna (x, yp, yn) onde x == coords radiais (igual para ambos lados)
        e yp / yn são os lados positivo e negativo em y."""
        r_bs0 = (self.stator_outer_diameter / 2 - self.slot_Hs0) * np.cos(self.slotoppener_ang / 2)
        x = np.array([r_bs0,
                      r_bs0 - self.slot_Hs1,
                      r_bs0 - self.slot_Hs1 - self.slot_Hs2,
                      r_bs0 - self.slot_Hs1 - self.slot_Hs2,
                      r_bs0,
                      r_bs0])
        yp = np.array([self.slot_Bs0 / 2,
                       self.slot_Bs1 / 2,
                       self.slot_Bs2 / 2,
                       0, 0,
                       self.slot_Bs0 / 2])
        return x, yp, -yp

    @staticmethod
    def _cell_center_xy(c):
        """Retorna (x, y, rc, tc) do centro de uma célula quadtree."""
        rc = 0.5 * (c["r0"] + c["r1"])
        tc = 0.5 * (c["t0"] + c["t1"])
        return rc * np.cos(tc), rc * np.sin(tc), rc, tc

    @staticmethod
    def _qtree_dfs(leaves, field_keys, compute_fn):
        """DFS genérico de refinamento quadtree por substituição in-place.

        Parameters
        ----------
        leaves      : lista de dicts com chaves r0/r1/t0/t1/d + field_keys
        field_keys  : nomes dos campos de dado em cada célula (ex: ['bx','by'])
        compute_fn  : callable(c, x, y, r) -> bool
                        Preenche c[field] in-place; retorna True se deve refinar.
        """
        k = 0
        while k < len(leaves):
            c = leaves[k]
            x, y, _, _ = BLDC_Process._cell_center_xy(c)
            r = np.hypot(x, y)
            should_refine = compute_fn(c, x, y, r)
            if should_refine:
                rm = 0.5 * (c["r0"] + c["r1"])
                tm = 0.5 * (c["t0"] + c["t1"])
                d2 = c["d"] + 1
                children = [
                    {"r0": c["r0"], "r1": rm,      "t0": c["t0"], "t1": tm,      "d": d2, **{fk: None for fk in field_keys}},
                    {"r0": c["r0"], "r1": rm,      "t0": tm,      "t1": c["t1"], "d": d2, **{fk: None for fk in field_keys}},
                    {"r0": rm,      "r1": c["r1"], "t0": c["t0"], "t1": tm,      "d": d2, **{fk: None for fk in field_keys}},
                    {"r0": rm,      "r1": c["r1"], "t0": tm,      "t1": c["t1"], "d": d2, **{fk: None for fk in field_keys}},
                ]
                leaves[k:k+1] = children  # substitui folha pelos 4 filhos (DFS)
            else:
                k += 1
        return leaves

    @staticmethod
    def _leaves_from_depth(depth, r_edges, th_edges, n_r, n_a, field_keys=()):
        """Reconstrói a lista de folhas (dicts r0/r1/t0/t1/d) a partir de um depth array.

        Percurso idêntico ao de _qtree_dfs: row-major sobre células base,
        filhos em ordem TL/TR/BL/BR. Um valor depth[i]==level indica folha;
        depth[i]>level indica região refinada em 4 filhos no nível seguinte.

        Parameters
        ----------
        depth     : 1D int array — saída de _qtree_dfs ou combine_grids
        r_edges   : array [n_r+1] — bordas radiais da grade base
        th_edges  : array [n_a+1] — bordas angulares da grade base
        n_r, n_a  : dimensões da grade base
        field_keys: campos adicionais inicializados como None em cada folha
        """
        depth  = np.asarray(depth, dtype=np.int64).ravel()
        cursor = [0]
        result = []

        def _expand(r0, r1, t0, t1, level):
            d = int(depth[cursor[0]])
            if d == level:
                result.append({"r0": r0, "r1": r1, "t0": t0, "t1": t1, "d": level,
                               **{fk: None for fk in field_keys}})
                cursor[0] += 1
            else:
                rm = 0.5 * (r0 + r1)
                tm = 0.5 * (t0 + t1)
                _expand(r0, rm, t0, tm, level + 1)  # TL
                _expand(r0, rm, tm, t1, level + 1)  # TR
                _expand(rm, r1, t0, tm, level + 1)  # BL
                _expand(rm, r1, tm, t1, level + 1)  # BR

        for ir in range(n_r):
            for ia in range(n_a):
                _expand(r_edges[ir], r_edges[ir + 1], th_edges[ia], th_edges[ia + 1], level=0)

        return result

    @staticmethod
    def _build_cell_rectangle(r0, r1, t0, t1):
        """Bounding box Cartesiano dos 4 cantos polares da célula quadtree.

        Converte os cantos (r0,t0), (r0,t1), (r1,t0), (r1,t1) para Cartesiano
        e retorna o retângulo axis-aligned (min/max de x e y).
        Mais rápido que polígono polar para interseções Shapely.
        """
        xs = [r * np.cos(t) for r in (r0, r1) for t in (t0, t1)]
        ys = [r * np.sin(t) for r in (r0, r1) for t in (t0, t1)]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    @staticmethod
    def _make_material_compute_fn(geometries, max_depth, homogeneity_threshold):
        """Factory: retorna closure compute(c, x, y, r) para _qtree_dfs.

        Substitui _make_compute_fn para o pipeline de features de material
        (mu_r e frac_dom) e de magnetização.

        Refinamento — bounding box Cartesiano da célula:
            frac_dom < homogeneity_threshold AND d < max_depth  →  refina

        Valor — point query no centro:
            Percorre geometrias em ordem e retorna o primeiro material que cobre o ponto.
            Campos preenchidos em c: definidos pelo chamador via field_keys.
            Este método preenche 'mu_r' e 'frac_dom'; save_magnetization_from_depth
            usa point queries diretas (sem DFS) para preencher 'm'.
        """
        permeability = BLDC_Process.PERMEABILITY

        def compute(c, x, y, r):
            # ── bounding box da célula ────────────────────────────────────────
            rect      = BLDC_Process._build_cell_rectangle(c["r0"], c["r1"], c["t0"], c["t1"])
            rect_area = rect.area

            area_dict = {mat: 0.0 for mat in geometries}
            for mat, geoms in geometries.items():
                for geom in geoms:
                    inter = rect.intersection(geom)
                    if not inter.is_empty:
                        area_dict[mat] += inter.area

            sum_area          = sum(area_dict.values())
            area_dict["vacuum"] = max(rect_area - sum_area, 0.0)

            dom_mat  = max(area_dict, key=area_dict.get)
            frac_dom = area_dict[dom_mat] / rect_area if rect_area > 0 else 1.0

            # ── point query no centro para mu_r ──────────────────────────────
            p        = Point(x, y)
            mu_r_val = permeability.get("vacuum", 1.0)
            for mat, geoms in geometries.items():
                for geom in geoms:
                    if geom.covers(p):
                        mu_r_val = permeability.get(mat, mu_r_val)
                        break
                else:
                    continue
                break

            c["mu_r"]     = float(mu_r_val)
            c["frac_dom"] = float(frac_dom)

            return frac_dom < homogeneity_threshold and c["d"] < max_depth

        return compute

    @staticmethod
    def _make_compute_fn(property_dict, field_key, ang_step, res, max_depth,
                         r_interest, r_d, geometries, refinement_low, refinement_high,
                         material_filter=None, ref_factor=1.2):
        """Retorna closure compute(c, x, y, r) para uso em _qtree_dfs.

        property_dict   : dict mat→valor escalar (MAGNETIZATION ou PERMEABILITY)
        field_key       : chave do campo no dict da folha ('m', 'mu', etc.)
        material_filter : set de materiais a considerar; None = todos
        ref_factor      : raio do disco de refinamento em unidades de arco da célula base
                          (>1 → disco vê células vizinhas, criando transições mais suaves)
        """
        def compute(c, x, y, r):
            # [REMOVIDO] disco único — mesmo raio usado para valor e para decisão de refinamento
            # radius    = r * ang_step / 3.0
            # disk      = Point(x, y).buffer(radius, resolution=res)
            # disk_area = disk.area

            # ── disco pequeno: calcula valor do campo (representativo da célula) ──
            radius_val = r * ang_step / 3.0
            disk_val   = Point(x, y).buffer(radius_val, resolution=res)
            disk_area  = disk_val.area

            if material_filter is not None:
                area_dict = {mat: 0.0 for mat in material_filter}
            else:
                area_dict = {mat: 0.0 for mat in geometries.keys()}

            for mat, geoms in geometries.items():
                if material_filter is not None and mat not in material_filter:
                    continue
                for geom in geoms:
                    inter = disk_val.intersection(geom)
                    if not inter.is_empty:
                        area_dict[mat] += inter.area

            sum_area = sum(area_dict.values())
            area_dict["vacuum"] = max(disk_area - sum_area, 0.0)

            c[field_key] = float(
                sum(property_dict.get(m, 0.0) * a for m, a in area_dict.items()) / disk_area
            )

            if c["d"] >= max_depth:
                return False

            # ── disco grande: detecta fronteiras em células vizinhas ──────────────
            radius_ref    = r * ang_step * ref_factor
            disk_ref      = Point(x, y).buffer(radius_ref, resolution=res)
            disk_ref_area = disk_ref.area

            if material_filter is not None:
                ref_area_dict = {mat: 0.0 for mat in material_filter}
            else:
                ref_area_dict = {mat: 0.0 for mat in geometries.keys()}

            for mat, geoms in geometries.items():
                if material_filter is not None and mat not in material_filter:
                    continue
                for geom in geoms:
                    inter = disk_ref.intersection(geom)
                    if not inter.is_empty:
                        ref_area_dict[mat] += inter.area

            ref_sum = sum(ref_area_dict.values())
            ref_area_dict["vacuum"] = max(disk_ref_area - ref_sum, 0.0)
            # [REMOVIDO] max_frac = max(area_dict.values()) / disk_area if disk_area > 0 else 1.0
            max_frac = max(ref_area_dict.values()) / disk_ref_area if disk_ref_area > 0 else 1.0

            return (refinement_low < max_frac < refinement_high) or \
                   (r_interest is not None and abs(r - r_interest) <= r_d)
        return compute

    @staticmethod
    def _cascade_adjacent(leaves, r_edges, th_edges, n_r, n_a, field_keys, buffer=1):
        """Refina células base adjacentes ao refinamento principal (vizinhança Moore, 8 vizinhos).

        Para cada célula com depth >= 1, os 8 vizinhos imediatos são refinados:
          - pass 0, depth == 1 → vizinhos vão para depth 1 (espessa linhas de refinamento)
          - pass 0, depth >  1 → vizinhos vão para depth - 1
          - passes seguintes, depth == 1 → não propaga (cascade para naturalmente)

        Aplica-se apenas a células não refinadas pelo critério base (base_depth == 0).
        Novas folhas têm field_keys=None; o chamador deve preenchê-las.
        Modifica `leaves` in-place.
        """
        def _base_cell(c):
            rc = 0.5 * (c["r0"] + c["r1"])
            tc = 0.5 * (c["t0"] + c["t1"])
            ir = int(np.searchsorted(r_edges[1:], rc))
            ia = int(np.searchsorted(th_edges[1:], tc))
            return max(0, min(ir, n_r - 1)), ia % n_a

        # Passo 1: profundidade máxima por célula base
        base_depth = np.zeros((n_r, n_a), dtype=np.int64)
        for c in leaves:
            ir, ia = _base_cell(c)
            if c["d"] > base_depth[ir, ia]:
                base_depth[ir, ia] = c["d"]

        # Passo 2: propagar cascade — calcula target por célula base
        target_depth = base_depth.copy()
        for pass_idx in range(buffer):
            new_target = target_depth.copy()
            changed = False
            for ir in range(n_r):
                for ia in range(n_a):
                    d = target_depth[ir, ia]
                    if d == 0:
                        continue
                    if pass_idx == 0 and d == 1:
                        t = 1
                    elif d > 1:
                        t = d - 1
                    else:
                        continue   # d==1, pass>0: target seria 0, não propaga
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            if dr == 0 and dc == 0:
                                continue
                            nr = ir + dr
                            nc = (ia + dc) % n_a
                            if 0 <= nr < n_r and new_target[nr, nc] < t:
                                new_target[nr, nc] = t
                                changed = True
            target_depth = new_target
            if not changed:
                break

        # Passo 3: células cascade = target > 0 e base_depth == 0
        cascade_set = {
            (ir, ia)
            for ir in range(n_r)
            for ia in range(n_a)
            if base_depth[ir, ia] == 0 and target_depth[ir, ia] > 0
        }
        if not cascade_set:
            return

        # Passo 4: substituir folha d=0 de cada célula cascade por folhas uniformes
        def _uniform_split(r0, r1, t0, t1, cur_d, tgt_d):
            if cur_d == tgt_d:
                return [{"r0": r0, "r1": r1, "t0": t0, "t1": t1, "d": cur_d,
                         **{fk: None for fk in field_keys}}]
            rm, tm = 0.5 * (r0 + r1), 0.5 * (t0 + t1)
            out = []
            for cr0, cr1, ct0, ct1 in [(r0,rm,t0,tm),(r0,rm,tm,t1),(rm,r1,t0,tm),(rm,r1,tm,t1)]:
                out.extend(_uniform_split(cr0, cr1, ct0, ct1, cur_d + 1, tgt_d))
            return out

        new_leaves = []
        replaced = set()
        for c in leaves:
            cell = _base_cell(c)
            if cell in cascade_set and cell not in replaced:
                ir, ia = cell
                new_leaves.extend(_uniform_split(
                    r_edges[ir], r_edges[ir + 1],
                    th_edges[ia], th_edges[ia + 1],
                    0, int(target_depth[ir, ia])
                ))
                replaced.add(cell)
            elif cell not in cascade_set:
                new_leaves.append(c)
            # cell in cascade_set e já em replaced: leaf original ignorada (já substituída)

        leaves[:] = new_leaves

    @staticmethod
    def generate_samples(num_samples, seed=42):

        N_POLES   = 42
        PHASE_MAX = 360.0 / (2 * N_POLES)

        space = {
            'number_rotor_poles':     {'unit': '',    'x_min': 42,       'x_max': 42},
            'number_stator_slots':    {'unit': '',    'x_min': 36,       'x_max': 36},
            'outer_diameter':         {'unit': 'mm',  'x_min': 93,       'x_max': 93},
            'inner_diameter':         {'unit': 'mm',  'x_min': 57,       'x_max': 57},
            'stator_outer_diameter':  {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'stator_inner_diameter':  {'unit': 'mm',  'x_min': 60,       'x_max': 60},
            'stack_length':           {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'slot_Hs0':               {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'slot_Hs1':               {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'slot_Hs2':               {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'slot_Bs0':               {'unit': 'mm',  'x_min': 1,        'x_max': 0},
            'slot_Bs1':               {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'slot_Bs2':               {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'slot_Rs':                {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'rotor_outer_diameter':   {'unit': 'mm',  'x_min': 90,       'x_max': 90},
            'rotor_inner_diameter':   {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'pole_embrance':          {'unit': '',    'x_min': 0.6,      'x_max': 0.90},
            'pole_thickness':         {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'rotor_phase':            {'unit': 'deg', 'x_min': 0,        'x_max': PHASE_MAX},
            }
        
        rng = np.random.default_rng(seed)
        samples = {key: {'unit': cfg['unit'], 'value': []} for key, cfg in space.items()}

        def _sample(lo, hi, md):
            if DISTRIBUTION == 'uniform':
                return rng.uniform(low=lo, high=hi)
            return rng.triangular(left=lo, right=hi, mode=md)

        for _ in range(num_samples):
            # length
            min = 5
            max = 10
            mode = min + 0.85 * (max - min)/2
            stack_length = _sample(min, max, mode)

            # stator outer diameter
            min_rotor = 4 # min rotor thickness
            min_gap = 0.5
            max_gap = 2

            max_stator = space['rotor_outer_diameter']['x_min'] - 2*(min_rotor + min_gap)
            min_stator = (space['stator_inner_diameter']['x_min'] + max_stator)/2
            
            min = min_stator
            max = max_stator
            mode = min + 0.8 * (max - min)
            stator_outer_diameter = _sample(min, max, mode)
            
            # rotor inner diameter
            min = min_gap
            max = max_gap
            mode = min + 0.3 * (max - min)
            gap = _sample(min, max, mode)
            
            rotor_inner_diameter = stator_outer_diameter + 2*gap
            rotor_outer_diameter = space['rotor_outer_diameter']['x_min']

            stator_inner_diameter = space['stator_inner_diameter']['x_min']

            # poles    
            min_back_iron = 1
            min_pole_thickness = 1.5

            min = min_pole_thickness
            max = rotor_outer_diameter/2 - rotor_inner_diameter/2 - min_back_iron
            mode = min + 0.8 * (max - min)
            pole_thickness = _sample(min, max, mode)

            min = space['pole_embrance']['x_min']
            max = space['pole_embrance']['x_max']
            mode = min + 0.8 * (max - min)
            pole_embrance = _sample(min, max, mode)

            # slots (height)
            # stator yoke
            n_slots = space['number_stator_slots']['x_min']
            slot_arc = 2*np.pi * stator_outer_diameter/2 / n_slots

            h = (stator_outer_diameter - stator_inner_diameter)/2

            min = 0.10 * h
            max = 0.30 * h
            mode = min + 0.5 * (max - min)
            stator_yoke = _sample(min, max, mode)

            h = h - stator_yoke

            min = h * 0.10
            max = h * 0.20
            slot_Hs0 = rng.uniform(low=min,high=max)
            
            min = h * 0.10
            max = h * 0.20
            slot_Hs1 = rng.uniform(low=min,high=max)
            
            slot_Hs2 = h - slot_Hs0 - slot_Hs1
            
            # slots (width)
            Hs = slot_Hs0
            min = space['slot_Bs0']['x_min']
            max = 0.2 * slot_arc
            mode = min + 0.3 * (max - min)
            slot_Bs0 = _sample(min, max, mode)

            Hs = slot_Hs0 + slot_Hs1
            slot_arc = 2*np.pi * (stator_outer_diameter/2 - Hs) / n_slots      
            min = 0.3 * slot_arc
            max = 0.5 * slot_arc
            mode = min + 0.5 * (max - min)
            slot_Bs1 = _sample(min, max, mode)

            Hs = slot_Hs0 + slot_Hs1 + slot_Hs2
            slot_arc = 2*np.pi * (stator_outer_diameter/2 - Hs) / n_slots  
            min = 0.40 * slot_arc
            max = 0.65 * slot_arc
            mode = min + 0.5 * (max - min)
            slot_Bs2 = _sample(min, max, mode)

            number_stator_slots = space['number_stator_slots']['x_min']
            number_rotor_poles = space['number_rotor_poles']['x_min']
            outer_diameter = space['outer_diameter']['x_min']
            inner_diameter = space['inner_diameter']['x_min']
            slot_Rs = 0

            rotor_phase = rng.uniform(0.0, PHASE_MAX)

            samples['outer_diameter']['value'].append(outer_diameter)
            samples['inner_diameter']['value'].append(inner_diameter)
            samples['number_rotor_poles']['value'].append(number_rotor_poles)
            samples['number_stator_slots']['value'].append(number_stator_slots)
            samples['stator_outer_diameter']['value'].append(stator_outer_diameter)
            samples['stator_inner_diameter']['value'].append(stator_inner_diameter)
            samples['stack_length']['value'].append(stack_length)
            samples['rotor_outer_diameter']['value'].append(rotor_outer_diameter)
            samples['rotor_inner_diameter']['value'].append(rotor_inner_diameter)
            samples['pole_embrance']['value'].append(pole_embrance)
            samples['pole_thickness']['value'].append(pole_thickness)
            samples['slot_Hs0']['value'].append(slot_Hs0)
            samples['slot_Hs1']['value'].append(slot_Hs1)
            samples['slot_Hs2']['value'].append(slot_Hs2)
            samples['slot_Rs']['value'].append(slot_Rs)
            samples['slot_Bs0']['value'].append(slot_Bs0)
            samples['slot_Bs1']['value'].append(slot_Bs1)
            samples['slot_Bs2']['value'].append(slot_Bs2)
            samples['rotor_phase']['value'].append(rotor_phase)

        return samples

    @staticmethod
    def generate_samples_constrained(num_samples, seed=42):
        """Geração de amostras com geometria parcialmente fixada.

        Constantes: stator_inner=60mm, rotor_outer=90mm, Hs0=1mm, Hs1=1.2mm, Hs2=3.7mm,
                    stack_length=1mm. Distribuição: DISTRIBUTION global ('uniform' ou 'triangular').
        Variáveis: stator_outer [73.8, 81]mm, rotor_inner (stator_outer+2*gap, gap∈[0.5,2]mm),
                   pole_thickness, pole_embrance [0.6, 0.90], Bs0, Bs1, Bs2.
        Bs* como frações do arco no respectivo raio (mesmos percentuais de generate_samples).
        """
        STATOR_INNER   = 60.0
        ROTOR_OUTER    = 90.0
        HS0, HS1, HS2  = 1.0, 1.2, 3.7
        HS_TOTAL       = HS0 + HS1 + HS2   # 5.9
        STACK_LENGTH   = 1.0
        N_SLOTS        = 36
        N_POLES        = 42
        OUTER_D        = 93.0
        INNER_D        = 57.0

        MIN_GAP            = 0.5
        MAX_GAP            = 2.0
        MIN_BACK_IRON      = 1.0
        MIN_POLE_THICKNESS = 1.5
        MIN_YOKE           = 1.0

        stator_outer_min = STATOR_INNER + 2.0 * (HS_TOTAL + MIN_YOKE)   # ≈ 73.8
        stator_outer_max = ROTOR_OUTER  - 2.0 * (MIN_GAP + MIN_POLE_THICKNESS + MIN_BACK_IRON)  # = 81.0

        PHASE_MAX = 0 #360.0 / (2 * N_POLES)

        space = {
            'number_rotor_poles':     {'unit': '',    'x_min': 42,       'x_max': 42},
            'number_stator_slots':    {'unit': '',    'x_min': 36,       'x_max': 36},
            'outer_diameter':         {'unit': 'mm',  'x_min': 93,       'x_max': 93},
            'inner_diameter':         {'unit': 'mm',  'x_min': 57,       'x_max': 57},
            'stator_outer_diameter':  {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'stator_inner_diameter':  {'unit': 'mm',  'x_min': 60,       'x_max': 60},
            'stack_length':           {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'slot_Hs0':               {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'slot_Hs1':               {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'slot_Hs2':               {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'slot_Bs0':               {'unit': 'mm',  'x_min': 1,        'x_max': 0},
            'slot_Bs1':               {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'slot_Bs2':               {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'slot_Rs':                {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'rotor_outer_diameter':   {'unit': 'mm',  'x_min': 90,       'x_max': 90},
            'rotor_inner_diameter':   {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'pole_embrance':          {'unit': '',    'x_min': 0.6,      'x_max': 0.90},
            'pole_thickness':         {'unit': 'mm',  'x_min': 0,        'x_max': 0},
            'rotor_phase':            {'unit': 'deg', 'x_min': 0,        'x_max': PHASE_MAX},
            }

        rng = np.random.default_rng(seed)
        samples = {key: {'unit': cfg['unit'], 'value': []} for key, cfg in space.items()}

        def _sample(lo, hi, md=None):
            if DISTRIBUTION == 'uniform':
                return rng.uniform(low=lo, high=hi)
            if md is None:
                md = lo + 0.8 * (hi - lo)
            return rng.triangular(left=lo, right=hi, mode=md)

        for _ in range(num_samples):
            stator_outer = _sample(stator_outer_min, stator_outer_max)

            # gap limitado para garantir espaço mínimo para polo + ferro de retorno
            gap_max = min(MAX_GAP, ROTOR_OUTER / 2.0 - stator_outer / 2.0 - MIN_BACK_IRON - MIN_POLE_THICKNESS)
            gap          = _sample(MIN_GAP, max(MIN_GAP, gap_max))
            rotor_inner  = stator_outer + 2.0 * gap

            pt_max         = ROTOR_OUTER / 2.0 - rotor_inner / 2.0 - MIN_BACK_IRON
            pole_thickness = _sample(MIN_POLE_THICKNESS, max(MIN_POLE_THICKNESS, pt_max))

            pole_embrance  = _sample(0.6, 0.90)

            # larguras do slot como fração do arco no respectivo raio
            arc_Bs0 = 2.0 * np.pi * (stator_outer / 2.0)              / N_SLOTS
            Bs0     = _sample(0.15 * arc_Bs0, 0.30 * arc_Bs0)

            arc_Bs1 = 2.0 * np.pi * (stator_outer / 2.0 - HS0 - HS1) / N_SLOTS
            Bs1     = _sample(0.30 * arc_Bs1, 0.40 * arc_Bs1)

            arc_Bs2 = 2.0 * np.pi * (stator_outer / 2.0 - HS_TOTAL)   / N_SLOTS
            Bs2     = _sample(0.35 * arc_Bs2, 0.50 * arc_Bs2)

            rotor_phase = _sample(0.0, PHASE_MAX)

            samples['outer_diameter']['value'].append(OUTER_D)
            samples['inner_diameter']['value'].append(INNER_D)
            samples['number_rotor_poles']['value'].append(N_POLES)
            samples['number_stator_slots']['value'].append(N_SLOTS)
            samples['stator_outer_diameter']['value'].append(stator_outer)
            samples['stator_inner_diameter']['value'].append(STATOR_INNER)
            samples['stack_length']['value'].append(STACK_LENGTH)
            samples['rotor_outer_diameter']['value'].append(ROTOR_OUTER)
            samples['rotor_inner_diameter']['value'].append(rotor_inner)
            samples['pole_embrance']['value'].append(pole_embrance)
            samples['pole_thickness']['value'].append(pole_thickness)
            samples['slot_Hs0']['value'].append(HS0)
            samples['slot_Hs1']['value'].append(HS1)
            samples['slot_Hs2']['value'].append(HS2)
            samples['slot_Rs']['value'].append(0.0)
            samples['slot_Bs0']['value'].append(Bs0)
            samples['slot_Bs1']['value'].append(Bs1)
            samples['slot_Bs2']['value'].append(Bs2)
            samples['rotor_phase']['value'].append(rotor_phase)

        return samples

    @staticmethod
    def generate_samples_fixed_geometry(num_samples, seed=42):
        """Todas as dimensões fixas; apenas o ângulo do rotor varia.

        Constantes: stator_inner=60mm, stator_outer=77.5mm, rotor_inner=80mm,
                    rotor_outer=90mm, Hs0=1mm, Hs1=1.2mm, Hs2=3.7mm,
                    Bs0=0.74mm, Bs1=2.79mm, Bs2=2.35mm,
                    pole_thickness=2mm, pole_embrance=0.90.
        Variável: rotor_phase ∈ [0, 360/N_POLES) graus — um pitch de polo completo.
        """
        STATOR_INNER     = 60.0
        STATOR_OUTER     = 77.5
        ROTOR_INNER      = 80.0
        ROTOR_OUTER      = 90.0
        HS0, HS1, HS2    = 1.0, 1.2, 3.7
        BS0, BS1, BS2    = 0.74, 2.79, 2.35
        POLE_THICKNESS   = 2.0
        POLE_EMBRANCE    = 0.90
        STACK_LENGTH     = 1.0
        N_SLOTS          = 36
        N_POLES          = 42
        OUTER_D          = 93.0
        INNER_D          = 57.0

        PHASE_MAX = 360.0 / (2*N_POLES)   # ≈ 17° — um pitch de polo

        space = {
            'number_rotor_poles':     {'unit': '',   'x_min': N_POLES,      'x_max': N_POLES},
            'number_stator_slots':    {'unit': '',   'x_min': N_SLOTS,      'x_max': N_SLOTS},
            'outer_diameter':         {'unit': 'mm', 'x_min': OUTER_D,      'x_max': OUTER_D},
            'inner_diameter':         {'unit': 'mm', 'x_min': INNER_D,      'x_max': INNER_D},
            'stator_outer_diameter':  {'unit': 'mm', 'x_min': STATOR_OUTER, 'x_max': STATOR_OUTER},
            'stator_inner_diameter':  {'unit': 'mm', 'x_min': STATOR_INNER, 'x_max': STATOR_INNER},
            'stack_length':           {'unit': 'mm', 'x_min': STACK_LENGTH, 'x_max': STACK_LENGTH},
            'slot_Hs0':               {'unit': 'mm', 'x_min': HS0,          'x_max': HS0},
            'slot_Hs1':               {'unit': 'mm', 'x_min': HS1,          'x_max': HS1},
            'slot_Hs2':               {'unit': 'mm', 'x_min': HS2,          'x_max': HS2},
            'slot_Bs0':               {'unit': 'mm', 'x_min': BS0,          'x_max': BS0},
            'slot_Bs1':               {'unit': 'mm', 'x_min': BS1,          'x_max': BS1},
            'slot_Bs2':               {'unit': 'mm', 'x_min': BS2,          'x_max': BS2},
            'slot_Rs':                {'unit': 'mm', 'x_min': 0,            'x_max': 0},
            'rotor_outer_diameter':   {'unit': 'mm', 'x_min': ROTOR_OUTER,  'x_max': ROTOR_OUTER},
            'rotor_inner_diameter':   {'unit': 'mm', 'x_min': ROTOR_INNER,  'x_max': ROTOR_INNER},
            'pole_embrance':          {'unit': '',   'x_min': POLE_EMBRANCE, 'x_max': POLE_EMBRANCE},
            'pole_thickness':         {'unit': 'mm', 'x_min': POLE_THICKNESS,'x_max': POLE_THICKNESS},
            'rotor_phase':            {'unit': 'deg','x_min': 0,            'x_max': PHASE_MAX},
        }

        rng = np.random.default_rng(seed)
        samples = {key: {'unit': cfg['unit'], 'value': []} for key, cfg in space.items()}

        for _ in range(num_samples):
            rotor_phase = rng.uniform(0.0, PHASE_MAX)

            samples['outer_diameter']['value'].append(OUTER_D)
            samples['inner_diameter']['value'].append(INNER_D)
            samples['number_rotor_poles']['value'].append(N_POLES)
            samples['number_stator_slots']['value'].append(N_SLOTS)
            samples['stator_outer_diameter']['value'].append(STATOR_OUTER)
            samples['stator_inner_diameter']['value'].append(STATOR_INNER)
            samples['stack_length']['value'].append(STACK_LENGTH)
            samples['rotor_outer_diameter']['value'].append(ROTOR_OUTER)
            samples['rotor_inner_diameter']['value'].append(ROTOR_INNER)
            samples['pole_embrance']['value'].append(POLE_EMBRANCE)
            samples['pole_thickness']['value'].append(POLE_THICKNESS)
            samples['slot_Hs0']['value'].append(HS0)
            samples['slot_Hs1']['value'].append(HS1)
            samples['slot_Hs2']['value'].append(HS2)
            samples['slot_Rs']['value'].append(0.0)
            samples['slot_Bs0']['value'].append(BS0)
            samples['slot_Bs1']['value'].append(BS1)
            samples['slot_Bs2']['value'].append(BS2)
            samples['rotor_phase']['value'].append(rotor_phase)

        return samples

    @staticmethod
    def export_params(params, filename="valid_designs.csv", data_path=None):
        save_dir = Path(data_path) if data_path else DATA_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        rows = {}
        for key, data in params.items():
            column_name = f"{key} [{data['unit']}]" if data['unit'] else key
            rows[column_name] = data['value']

        df = pd.DataFrame(rows)
        df.to_csv(save_dir / filename, index=False)

    @staticmethod
    def extract_params_at_index(motor_params, code):
        result = {}
        for key, info in motor_params.items():
            value_list = info['value']
            val = value_list[code]
            result[key] = { 'unit': info['unit'], 'value': val}

        return result

class BLDC_FEMM_Model(BLDC_Process):

    N_TURNS = 18  # número de espiras por bobina

    def __init__(self, motor_params, phase):
        super().__init__(motor_params=motor_params)
        self.phase = phase
          
    def draw_motor(self):
        # rotor back iron
        self._create_sec(r_in=self.rotor_inner_diameter/2 + self.pole_thickness,
                   r_ext=self.rotor_outer_diameter/2,
                   ang_in_1=0,
                   ang_in_2=180,
                   seg_1=False,
                   seg_2=False,
                   material=self.material_iron)
        self._create_sec(r_in=self.rotor_inner_diameter/2 + self.pole_thickness,
                   r_ext=self.rotor_outer_diameter/2,
                   ang_in_1=180,
                   ang_in_2=0,
                   seg_1=False,
                   seg_2=False)
        
        # create poles
        step_ang = 360/self.number_rotor_poles
        pole_ang = step_ang * self.pole_embrance
        pole_offset = 0
        for pole in range(self.number_rotor_poles):
            ang_1 = step_ang * pole + pole_offset
            ang_2 = ang_1 + pole_ang
            direction = 0 if pole % 2 == 0 else 180
            self._create_sec(r_in=self.rotor_inner_diameter/2,
                             r_ext=self.rotor_inner_diameter/2 + self.pole_thickness,
                             ang_in_1=ang_1,
                             ang_in_2=ang_2,
                             material=self.material_mag,
                             direction=direction,
                             phase=self.phase)
        
        # create coils
        
        step_ang = 360/self.number_stator_slots
        for coil in range(self.number_stator_slots):
            self._create_coils(ang=step_ang*coil)
        self._coil_props()

        # create stator
        self._create_stator()

        # boundary conditions
        self._set_boundary()

    def _set_boundary(self):
        
        res = 1
        femm.mi_addboundprop("A=0", 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)

        femm.mi_getmaterial(self.material_gap)
        femm.mi_addblocklabel(0,0)
        femm.mi_selectlabel(0,0)
        femm.mi_setblockprop(self.material_gap, 0, 1, "", 0, 0, 0)
        femm.mi_clearselected()

        r = self.outer_diameter/2
        r_in = self.outer_diameter/2 * 0.99
        ang_1 = 0
        ang_2 = np.pi
        theta = np.pi/2
        femm.mi_drawarc(r * np.cos(ang_1), r * np.sin(ang_1),
                    r * np.cos(ang_2), r * np.sin(ang_2),
                    np.rad2deg(np.abs(ang_1-ang_2)),res)
        ang_1 = np.pi
        ang_2 = 0
        femm.mi_drawarc(r * np.cos(ang_1), r * np.sin(ang_1),
                    r * np.cos(ang_2), r * np.sin(ang_2),
                    np.rad2deg(np.abs(ang_1-ang_2)),res)
        
        femm.mi_selectarcsegment( r * np.cos(theta),  r * np.sin(theta))
        femm.mi_selectarcsegment(-r * np.cos(theta), -r * np.sin(theta))
        femm.mi_setarcsegmentprop(1,"A=0", 0, 0)
        femm.mi_clearselected()

        femm.mi_getmaterial(self.material_gap)
        femm.mi_addblocklabel(r_in * np.cos(theta),r_in * np.sin(theta))
        femm.mi_selectlabel(r_in * np.cos(theta),r_in * np.sin(theta))
        femm.mi_setblockprop(self.material_gap, 0, 1, "", 0, 0, 0)
        femm.mi_clearselected()

    def _create_stator(self):
        res = 1

        r_ext = self.stator_outer_diameter/2
        r_in = r_ext - self.slot_Hs0
        step = 2*np.pi/self.number_stator_slots
        tooth_ang = step - self.slotoppener_ang

        for i in range(self.number_stator_slots):
            ang_1 = step * i + self.slotoppener_ang/2
            ang_2 = ang_1 + tooth_ang
            femm.mi_drawarc(r_ext * np.cos(ang_1), r_ext * np.sin(ang_1),
                        r_ext * np.cos(ang_2), r_ext * np.sin(ang_2),
                        np.rad2deg(np.abs(ang_1-ang_2)),res)
            
            femm.mi_drawline(r_ext * np.cos(ang_1), r_ext * np.sin(ang_1),
                        r_in * np.cos(ang_1), r_in * np.sin(ang_1))
            femm.mi_drawline(r_ext * np.cos(ang_2), r_ext * np.sin(ang_2),
                        r_in * np.cos(ang_2), r_in * np.sin(ang_2))
            
        ang_avg = np.pi/2
        r_avg = (self.rotor_inner_diameter+self.stator_outer_diameter)/4
        femm.mi_getmaterial(self.material_gap)
        femm.mi_addblocklabel(r_avg * np.cos(ang_avg),r_avg * np.sin(ang_avg))
        femm.mi_selectlabel(r_avg * np.cos(ang_avg),r_avg * np.sin(ang_avg))
        femm.mi_setblockprop(self.material_gap, 0, 1, "", np.rad2deg(ang_avg), 0, 0)
        femm.mi_clearselected()

        r_in = self.stator_inner_diameter/2
        ang_1 = 0
        ang_2 = np.pi
        femm.mi_drawarc(r_in * np.cos(ang_1), r_in * np.sin(ang_1),
                    r_in * np.cos(ang_2), r_in * np.sin(ang_2),
                    np.rad2deg(np.abs(ang_1-ang_2)),res)
        ang_1 = np.pi
        ang_2 = 0
        femm.mi_drawarc(r_in * np.cos(ang_1), r_in * np.sin(ang_1),
                    r_in * np.cos(ang_2), r_in * np.sin(ang_2),
                    np.rad2deg(np.abs(ang_1-ang_2)),res)
        
        ang_avg = np.pi/2
        r_avg = (self.stator_inner_diameter/2 + self.stator_outer_diameter/2 - (self.slot_Hs0 + self.slot_Hs1 + self.slot_Hs2))/2
        femm.mi_getmaterial(self.material_iron)
        femm.mi_addblocklabel(r_avg * np.cos(ang_avg),r_avg * np.sin(ang_avg))
        femm.mi_selectlabel(r_avg * np.cos(ang_avg),r_avg * np.sin(ang_avg))
        femm.mi_setblockprop(self.material_iron, 0, 1, "", np.rad2deg(ang_avg), 0, 0)
        femm.mi_clearselected()
        
    def _create_coils(self, ang=0):
        ang_rad = np.deg2rad(ang)
        x, yp, yn = self._coil_coords()

        vp = np.column_stack((x, yp))
        vn = np.column_stack((x, yn))

        R = np.array([[np.cos(ang_rad), -np.sin(ang_rad)],
                      [np.sin(ang_rad),  np.cos(ang_rad)]])
        vp = vp @ R.T
        vn = vn @ R.T

        for i in range(len(vp) - 1):
            femm.mi_drawline(vp[i, 0], vp[i, 1], vp[i+1, 0], vp[i+1, 1])
            femm.mi_drawline(vn[i, 0], vn[i, 1], vn[i+1, 0], vn[i+1, 1])

    def _coil_props(self):
        n_turns = self.N_TURNS

        r = self.stator_outer_diameter/2 - (self.slot_Hs0 + self.slot_Hs1 + self.slot_Hs2/2)
        step = 2*np.pi/self.number_stator_slots
        ang_offset = step/2 - self.slotoppener_ang/2

        for i in range(self.number_stator_slots):

            femm.mi_addcircprop('coil_' + str(i) + '_0', 0, 1)
            femm.mi_addcircprop('coil_' + str(i) + '_1', 0, 1)

            ang = step * i + step/2 - ang_offset
            femm.mi_getmaterial(self.material_copper)
            femm.mi_addblocklabel(r*np.cos(ang),r*np.sin(ang))
            femm.mi_selectlabel(r*np.cos(ang),r*np.sin(ang))
            femm.mi_setblockprop(self.material_copper, 0, 1, 'coil_' + str(i) + '_0', 0, 0, n_turns)
            femm.mi_clearselected()

            ang = step * i + step/2 + ang_offset
            femm.mi_getmaterial(self.material_copper)
            femm.mi_addblocklabel(r*np.cos(ang),r*np.sin(ang))
            femm.mi_selectlabel(r*np.cos(ang),r*np.sin(ang))
            femm.mi_setblockprop(self.material_copper, 0, 1, 'coil_' + str(i) + '_1', 0, 0, n_turns)
            femm.mi_clearselected()

    def _create_sec(self, r_in, r_ext, ang_in_1, ang_in_2, ang_ext_1=None, ang_ext_2=None,
                    seg_1=True, seg_2=True, material=None, direction=0, phase=0):

        if ang_ext_1 == None:
            ang_ext_1 = ang_in_1
            ang_ext_2 = ang_in_2

        ang_in_1 = np.deg2rad(ang_in_1 + phase)
        ang_in_2 = np.deg2rad(ang_in_2 + phase)
        ang_ext_1 = np.deg2rad(ang_ext_1 + phase)
        ang_ext_2 = np.deg2rad(ang_ext_2 + phase)

        # max arc sub seg ang
        res = 1

        femm.mi_drawarc(r_in * np.cos(ang_in_1), r_in * np.sin(ang_in_1),
                    r_in * np.cos(ang_in_2), r_in * np.sin(ang_in_2),
                    np.rad2deg(np.abs(ang_in_1-ang_in_2)),res)
        
        femm.mi_drawarc(r_ext * np.cos(ang_ext_1), r_ext * np.sin(ang_ext_1),
                    r_ext * np.cos(ang_ext_2), r_ext * np.sin(ang_ext_2),
                    np.rad2deg(np.abs(ang_ext_1-ang_ext_2)),res)

        if seg_1:
            femm.mi_addsegment(r_in * np.cos(ang_in_1), r_in * np.sin(ang_in_1),
                        r_ext * np.cos(ang_ext_1), r_ext * np.sin(ang_ext_1))
        if seg_2:
            femm.mi_addsegment(r_in * np.cos(ang_in_2), r_in * np.sin(ang_in_2),
                        r_ext * np.cos(ang_ext_2), r_ext * np.sin(ang_ext_2))
        
        if material:
            ang_avg = (ang_in_1+ang_in_2+ang_ext_1+ang_ext_2)/4
            r_avg = (r_in+r_ext)/2
            femm.mi_getmaterial(material)
            femm.mi_addblocklabel(r_avg * np.cos(ang_avg),r_avg * np.sin(ang_avg))
            femm.mi_selectlabel(r_avg * np.cos(ang_avg),r_avg * np.sin(ang_avg))
            femm.mi_setblockprop(material, 0, 1, "", np.rad2deg(ang_avg) + direction, 0, 0)
            femm.mi_clearselected()

    def save_B_grid(self, ang_1, ang_2, n_r, n_a, code = 0):

        r_in = self.inner_diameter/2
        r_ext = self.outer_diameter/2
        ang_1 = np.deg2rad(ang_1)
        ang_2 = np.deg2rad(ang_2)

        Bx = np.empty((n_r, n_a), dtype=float)
        By = np.empty((n_r, n_a), dtype=float)
        bx_list, by_list = [], []

        for x, y, r, th in self._iter_polar_points(r_in = r_in, r_ext = r_ext, 
                                            ang_1 = ang_1, ang_2 = ang_2, 
                                            n_r = n_r, n_a = n_a):
            bx, by = femm.mo_getb(x, y)
            bx_list.append(bx)
            by_list.append(by)
        
        Bx[:] = np.asarray(bx_list, dtype=float).reshape(n_r, n_a)
        By[:] = np.asarray(by_list, dtype=float).reshape(n_r, n_a)

        Fields = [Bx, By]
        names = [f"Mag_Bx_{code}.csv",f"Mag_By_{code}.csv"]

        self.save_csv(Fields=Fields,names=names)

    def save_B_grid_qtree(self, ang_1, ang_2, n_r, n_a, r_interest, max_depth=1, code=0, r_d=0.5):
        """
        Quadtree sampling in (r,theta) cells.
        Critério de refinamento: banda radial |r - r_interest| <= r_d.
        """
        r_in  = self.inner_diameter / 2
        r_ext = self.outer_diameter / 2
        ang_1 = np.deg2rad(ang_1)
        ang_2 = np.deg2rad(ang_2)

        r_edges  = np.linspace(r_in,  r_ext, n_r + 1)
        th_edges = np.linspace(ang_1, ang_2, n_a + 1)

        leaves = []
        for ir in range(n_r):
            for ia in range(n_a):
                leaves.append({
                    "r0": r_edges[ir], "r1": r_edges[ir + 1],
                    "t0": th_edges[ia], "t1": th_edges[ia + 1],
                    "d": 0, "bx": None, "by": None
                })

        def compute(c, x, y, r):
            bx, by = femm.mo_getb(x, y)
            c["bx"], c["by"] = float(bx), float(by)
            return c["d"] < max_depth and abs(r - r_interest) <= r_d

        self._qtree_dfs(leaves, field_keys=["bx", "by"], compute_fn=compute)

        depth = np.array([c["d"]  for c in leaves], dtype=np.int32).reshape(-1, 1)
        Bx    = np.array([c["bx"] for c in leaves], dtype=float).reshape(-1, 1)
        By    = np.array([c["by"] for c in leaves], dtype=float).reshape(-1, 1)

        Fields = [Bx, By, depth]
        names  = [
            f"Mag_Bx_qt_{code}.csv",
            f"Mag_By_qt_{code}.csv",
            f"Mag_B_depth_qt_{code}.csv"
        ]
        self.save_csv(Fields=Fields, names=names)

    def save_B_grid_pts(self, ang_1, ang_2, n_r, n_a, code=0):
        """Salva Bx/By na grade base [n_r, n_a] — point queries nos centros H×W.

        Sem averaging de sub-células qtree. A solução FEMM deve estar carregada.
        Chamado em ambos os modos (grid e qtree).
        CSVs: Mag_Bx_grid_{code}.csv, Mag_By_grid_{code}.csv
        """
        r_in      = self.inner_diameter / 2
        r_ext     = self.outer_diameter  / 2
        ang_1_rad = np.deg2rad(ang_1)
        ang_2_rad = np.deg2rad(ang_2)

        bx_list, by_list = [], []
        for x, y, r, th in self._iter_polar_points(r_in, r_ext, ang_1_rad, ang_2_rad, n_r, n_a):
            bx, by = femm.mo_getb(x, y)
            bx_list.append(bx)
            by_list.append(by)

        Bx = np.asarray(bx_list, dtype=float).reshape(-1, 1)
        By = np.asarray(by_list, dtype=float).reshape(-1, 1)
        self.save_csv(Fields=[Bx, By],
                      names=[f"Mag_Bx_grid_{code}.csv", f"Mag_By_grid_{code}.csv"])

    def save_B_grid_qtree_from_depth(self, depth, ang_1, ang_2, n_r, n_a, code=0):
        """Amostra Bx/By nos centros das folhas definidas por depth.

        Não aplica critério de refinamento próprio — a estrutura vem do Shapely.
        A solução FEMM deve estar carregada (mi_loadsolution já chamado).

        Parameters
        ----------
        depth : 1D int array — estrutura unificada (saída de save_material_mu_qtree / depth_qt_)
        """
        r_in      = self.inner_diameter / 2
        r_ext     = self.outer_diameter  / 2
        ang_1_rad = np.deg2rad(ang_1)
        ang_2_rad = np.deg2rad(ang_2)

        r_edges  = np.linspace(r_in,      r_ext,     n_r + 1)
        th_edges = np.linspace(ang_1_rad, ang_2_rad, n_a + 1)

        leaves = self._leaves_from_depth(
            depth, r_edges, th_edges, n_r, n_a, field_keys=("bx", "by")
        )

        for c in leaves:
            x, y, _, _ = self._cell_center_xy(c)
            bx, by = femm.mo_getb(x, y)
            c["bx"], c["by"] = float(bx), float(by)

        # [REMOVIDO] depth_arr salvo como Mag_B_depth_qt_ — depth unificado já em depth_qt_
        Bx = np.array([c["bx"] for c in leaves], dtype=float).reshape(-1, 1)
        By = np.array([c["by"] for c in leaves], dtype=float).reshape(-1, 1)

        Fields = [Bx, By]
        names  = [
            f"Mag_Bx_qt_{code}.csv",
            f"Mag_By_qt_{code}.csv",
        ]
        self.save_csv(Fields=Fields, names=names)

class BLDC_Shapely_Model(BLDC_Process):
    def __init__(self, motor_params, phase):
        super().__init__(motor_params=motor_params)
        self.phase = phase

    def draw_motor(self):

        # backiron
        center = Point(0,0)
        c1 = center.buffer(self.rotor_outer_diameter/2)
        c2 = center.buffer(self.rotor_inner_diameter/2 + self.pole_thickness)
        self.back_iron = c1.difference(c2)
        self.geometries['iron_1008'].append(self.back_iron)

        #out motor
        c_out = center.buffer(self.outer_diameter/2)
        self.out_motor = c_out.difference(c1)
        # [REMOVIDO] vácuo deve ser implícito (resíduo de área não coberta por
        # ferro/ímã/cobre), igual ao resto do código. Adicionar out_motor aqui
        # fazia compute() em _make_material_compute_fn contar a área de vácuo
        # duas vezes (interseção + resíduo), zerando frac_dom e forçando refino
        # até max_depth no anel entre o rotor e o raio externo de amostragem.
        # self.geometries['vacuum'].append(self.out_motor)

        # poles
        step_ang = 360/self.number_rotor_poles
        pole_ang = step_ang * self.pole_embrance
        pole_offset = 0
        self.poles_p = Polygon() 
        self.poles_n = Polygon() 
        for pole in range(self.number_rotor_poles):
            ang_1 = step_ang * pole + pole_offset
            ang_2 = ang_1 + pole_ang
            direction = 0 if pole % 2 == 0 else 180
            poly = self._create_sec(r_in=self.rotor_inner_diameter/2,
                             r_ext=self.rotor_inner_diameter/2 + self.pole_thickness,
                             ang_in_1=ang_1, ang_in_2=ang_2, phase=self.phase)
            if direction == 0:
                self.poles_p = self.poles_p.union(poly)
            else:
                self.poles_n = self.poles_n.union(poly)

        self.geometries['N35p'].append(self.poles_p)
        self.geometries['N35n'].append(self.poles_n)

        # coils
        step_ang = 360/self.number_stator_slots
        self.coils = Polygon()
        for coil in range(self.number_stator_slots):
            poly = self._create_coils(ang=step_ang*coil)
            self.coils = self.coils.union(poly)
        self.geometries['copper'].append(self.coils)

        # stator
        self.stator = self._create_stator()
        self.geometries['iron_1008'].append(self.stator)

        #in motor
        self.in_motor = center.buffer(self.inner_diameter/2)
        # [REMOVIDO] mesmo motivo do out_motor acima — vácuo implícito, não
        # cadastrado em geometries['vacuum']. Na prática in_motor nunca
        # intersecta o domínio amostrado (raio igual a r_in da amostragem),
        # mas mantemos o mesmo padrão por consistência.
        # self.geometries['vacuum'].append(self.in_motor)

    def _create_air_gap(self):
        center = Point(0,0)
        c1 = center.buffer(self.rotor_outer_diameter/2 * 1.1)
        c2 = center.buffer(self.stator_inner_diameter/2 * 0.9)
        ring = c1.difference(c2)

        rotor  = unary_union([self.back_iron, self.poles_n, self.poles_p])
        stator = unary_union([self.stator,self.coils])

        motor  = unary_union([rotor, stator])

        air_gap = ring.difference(motor)

        return air_gap
    
    def _create_stator(self):
        res = 1
        n = 360/res

        r_ext = self.stator_outer_diameter/2
        r_in = r_ext - self.slot_Hs0
        step = 2*np.pi/self.number_stator_slots
        tooth_ang = step - self.slotoppener_ang

        coords = []
        for i in range(self.number_stator_slots):
            ang_1 = step * i + self.slotoppener_ang / 2.0
            ang_2 = ang_1 + tooth_ang
            ang_3 = ang_2 + self.slotoppener_ang

            p1_in  = np.array([r_in * np.cos(ang_1), r_in * np.sin(ang_1)])

            angs = np.linspace(ang_1, ang_2, max(3, int(n)))
            outer = np.column_stack((r_ext * np.cos(angs), r_ext * np.sin(angs)))

            p2_in  = np.array([r_in * np.cos(ang_2), r_in * np.sin(ang_2)])

            p3_in  = np.array([r_in * np.cos(ang_3), r_in * np.sin(ang_3)])

            coords += [p1_in, *outer, p2_in, p3_in]

        stator = Polygon(np.vstack(coords))

        center = Point(0,0)
        stator = stator.difference(self.coils)
        c_in = center.buffer(self.stator_inner_diameter/2)
        stator = stator.difference(c_in)

        return stator

    def _create_coils(self, ang=0):
        ang_rad = np.deg2rad(ang)
        x, yp, yn = self._coil_coords()

        vp = np.column_stack((x, yp))
        vn = np.column_stack((x, yn))

        R = np.array([[np.cos(ang_rad), -np.sin(ang_rad)],
                      [np.sin(ang_rad),  np.cos(ang_rad)]])
        vp = vp @ R.T
        vn = vn @ R.T

        poly_p = Polygon(np.vstack([vp, vp[0]]))
        poly_n = Polygon(np.vstack([vn, vn[0]]))
        return poly_p.union(poly_n)

    def _create_sec(self, r_in, r_ext, ang_in_1, ang_in_2, ang_ext_1=None, ang_ext_2=None,
                    seg_1=True, seg_2=True, phase = 0):
        
        res = 1
        n = int(np.rint(np.abs(ang_in_1 - ang_in_2)/res))

        if ang_ext_1 is None:
            ang_ext_1 = ang_in_1
            ang_ext_2 = ang_in_2
            
        ang_in_1, ang_in_2 = np.deg2rad([ang_in_1 + phase, ang_in_2 + phase])
        ang_ext_1, ang_ext_2 = np.deg2rad([ang_ext_1 + phase, ang_ext_2 + phase])

        ang_ext = np.linspace(ang_ext_1, ang_ext_2, n)
        x_ext = r_ext * np.cos(ang_ext)
        y_ext = r_ext * np.sin(ang_ext)

        ang_in = np.linspace(ang_in_2, ang_in_1, n)
        x_in = r_in * np.cos(ang_in)
        y_in = r_in * np.sin(ang_in)

        x = np.concatenate([x_ext, x_in])
        y = np.concatenate([y_ext, y_in])

        return Polygon(np.c_[x, y])

    def plot(self, material_geoms = None):

        if material_geoms is None:
                material_geoms = self.geometries
        face_alpha = 0.9

        color_map = {
            'vacuum': "purple",
            'iron_1008': "green",
            'N35p': "red",
            'N35n': "blue",
            'copper': "orange"
        }

        fig, ax = plt.subplots()

        def _plot_polygon(poly, color):
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=face_alpha, color=color if color else "white")
            for hole in poly.interiors:
                hx, hy = hole.xy
                ax.fill(hx, hy, color="white")

        for material, geoms in material_geoms.items():
            color = color_map.get(material, "black")
            for geom in geoms:
                if geom.is_empty:
                    continue
                if geom.geom_type == "Polygon":
                    _plot_polygon(geom, color)
                elif geom.geom_type == "MultiPolygon":
                    for poly in geom.geoms:
                        _plot_polygon(poly, color)

        ax.set_aspect("equal", adjustable="box")

        return fig, ax

    def save_magnetization_from_depth(self, leaves_or_depth, ang_1, ang_2, n_r, n_a, code=0):
        """Salva magnetização (M) nos centros das folhas definidas pelo depth unificado.

        Análogo a save_B_grid_qtree_from_depth (FEMM), mas para Shapely.
        Não executa DFS próprio — recebe a estrutura já calculada por save_material_mu_qtree.

        Parameters
        ----------
        leaves_or_depth : lista de folhas (retorno de save_material_mu_qtree)
                          OU array 1D int (depth_qt_ lido de CSV).
        ang_1, ang_2    : ângulos em graus (usados só se leaves_or_depth for array).
        n_r, n_a        : dimensões da grade base (usados só se for array).

        CSV gerado: Mag_M_qt_{code}.csv  (sem depth próprio — usa depth_qt_)
        """
        if isinstance(leaves_or_depth, (list,)):
            leaves = leaves_or_depth
        else:
            # reconstrução a partir do array de depth (lido de CSV)
            r_in      = self.inner_diameter / 2
            r_ext     = self.outer_diameter  / 2
            ang_1_rad = np.deg2rad(ang_1)
            ang_2_rad = np.deg2rad(ang_2)
            r_edges   = np.linspace(r_in,      r_ext,     n_r + 1)
            th_edges  = np.linspace(ang_1_rad, ang_2_rad, n_a + 1)
            leaves    = self._leaves_from_depth(
                leaves_or_depth, r_edges, th_edges, n_r, n_a, field_keys=("m",)
            )

        for c in leaves:
            x, y, _, _ = self._cell_center_xy(c)
            p     = Point(x, y)
            m_val = 0.0
            for mat, geoms in self.geometries.items():
                for geom in geoms:
                    if geom.covers(p):
                        m_val = float(self.MAGNETIZATION.get(mat, 0.0))
                        break
                else:
                    continue
                break
            c["m"] = m_val

        Mag = np.array([c["m"] for c in leaves], dtype=float).reshape(-1, 1)
        self.save_csv(Fields=[Mag], names=[f"Mag_M_qt_{code}.csv"])

    # [REMOVIDO] save_material_mu_avg_qtree — substituído por save_material_mu_qtree
    # Motivo: usava dois discos Shapely (disk_val + disk_ref) para refinamento;
    #         gerava Mu_avg_qt_ (média ponderada por área) e não incluía frac_dom.
    # def save_material_mu_avg_qtree(self, ang_1, ang_2, n_r, n_a,
    #                                max_depth=1, code=0, res=20,
    #                                r_interest=None, r_d=0.5): ...

    def save_material_mu_qtree(self, ang_1, ang_2, n_r, n_a,
                               max_depth=1, code=0):
        """Quadtree sampling de permeabilidade (mu_r) e fração dominante (frac_dom).

        Substitui save_material_mu_avg_qtree.
        Análogo a save_B_grid_qtree_from_depth (FEMM): executa o único DFS da amostra
        e salva features de material. save_magnetization_from_depth usa as folhas retornadas.

        Refinamento: bounding box Cartesiano — _make_material_compute_fn.
            Refina se frac_dom < HOMOGENEITY_THRESHOLD e d < max_depth.
        Valor:
            mu_r     = PERMEABILITY[material no centro] (point query)
            frac_dom = fração de área do material dominante no bounding box
        CSVs: Mu_r_qt_{code}.csv, depth_qt_{code}.csv, Frac_dom_qt_{code}.csv
        depth_qt_ é o depth unificado de toda a amostra — usado por
        save_magnetization_from_depth e save_B_grid_qtree_from_depth.
        """
        r_in  = self.inner_diameter / 2
        r_ext = self.outer_diameter  / 2
        ang_1 = np.deg2rad(ang_1)
        ang_2 = np.deg2rad(ang_2)

        r_edges  = np.linspace(r_in,  r_ext, n_r + 1)
        th_edges = np.linspace(ang_1, ang_2, n_a + 1)

        leaves = []
        for ir in range(n_r):
            for ia in range(n_a):
                leaves.append({
                    "r0": r_edges[ir], "r1": r_edges[ir + 1],
                    "t0": th_edges[ia], "t1": th_edges[ia + 1],
                    "d": 0, "mu_r": None, "frac_dom": None
                })

        compute = self._make_material_compute_fn(
            geometries=self.geometries,
            max_depth=max_depth,
            homogeneity_threshold=HOMOGENEITY_THRESHOLD,
        )

        self._qtree_dfs(leaves, field_keys=["mu_r", "frac_dom"], compute_fn=compute)
        if CASCADE_BUFFER is not None:
            self._cascade_adjacent(leaves, r_edges, th_edges, n_r, n_a,
                                   field_keys=["mu_r", "frac_dom"], buffer=CASCADE_BUFFER)

        for c in leaves:
            if c["mu_r"] is None:
                x, y, _, _ = self._cell_center_xy(c)
                compute(c, x, y, np.hypot(x, y))

        depth    = np.array([c["d"]        for c in leaves], dtype=np.int32).reshape(-1, 1)
        Mu_r     = np.array([c["mu_r"]     for c in leaves], dtype=float).reshape(-1, 1)
        Frac_dom = np.array([c["frac_dom"] for c in leaves], dtype=float).reshape(-1, 1)

        Fields = [Mu_r, depth, Frac_dom]
        names  = [
            f"Mu_r_qt_{code}.csv",
            f"depth_qt_{code}.csv",      # depth unificado — compartilhado com M e B
            f"Frac_dom_qt_{code}.csv",
        ]
        self.save_csv(Fields=Fields, names=names)
        return leaves   # retorna folhas para reuso imediato sem re-leitura de CSV

    def save_grid_mu(self, ang_1, ang_2, n_r, n_a, code=0):
        """Salva Mu_r e M na grade base [n_r, n_a] — point queries nos centros H×W.

        Sem averaging de sub-células qtree: cada célula recebe o valor do material
        no seu ponto central exato. Fronteiras de material nítidas.
        Chamado em ambos os modos (grid e qtree).
        CSVs: Mu_r_grid_{code}.csv, Mag_M_grid_{code}.csv
        """
        permeability  = BLDC_Process.PERMEABILITY
        magnetization = BLDC_Process.MAGNETIZATION
        r_in      = self.inner_diameter / 2
        r_ext     = self.outer_diameter  / 2
        ang_1_rad = np.deg2rad(ang_1)
        ang_2_rad = np.deg2rad(ang_2)

        mu_list, m_list = [], []
        for x, y, r, th in self._iter_polar_points(r_in, r_ext, ang_1_rad, ang_2_rad, n_r, n_a):
            p    = Point(x, y)
            mu_r = permeability['vacuum']
            m    = magnetization['vacuum']
            found = False
            for material, geoms in self.geometries.items():
                for geom in geoms:
                    if geom.covers(p):
                        mu_r  = permeability[material]
                        m     = magnetization[material]
                        found = True
                        break
                if found:
                    break
            mu_list.append(mu_r)
            m_list.append(m)

        Mu_r = np.asarray(mu_list, dtype=float).reshape(-1, 1)
        M    = np.asarray(m_list,  dtype=float).reshape(-1, 1)
        self.save_csv(Fields=[Mu_r, M],
                      names=[f"Mu_r_grid_{code}.csv", f"Mag_M_grid_{code}.csv"])

    def save_material_grid(self, ang_1, ang_2, n_r, n_a, code = 0):
        material_table = self.MATERIAL_ID

        r_in = self.inner_diameter/2
        r_ext = self.outer_diameter/2
        ang_1 = np.deg2rad(ang_1)
        ang_2 = np.deg2rad(ang_2)
        
        Field = np.empty((n_r, n_a), dtype=float)
        field_list = []

        for x, y, r, th in self._iter_polar_points(
                r_in=r_in, r_ext=r_ext,
                ang_1=ang_1, ang_2=ang_2,
                n_r=n_r, n_a=n_a):

            p = Point(x, y)
            found = False
            for material, geoms in self.geometries.items():
                for geom in geoms:
                    if geom.covers(p):
                        material_value = material_table[material]
                        found = True
                        break
                if found:
                    break
            if not found:
                material_value = material_table['vacuum']
            field_list.append(material_value)

        Field[:] = np.asarray(field_list, dtype=float).reshape(n_r, n_a)

        Fields = [Field]
        names = [f"Material_{code}.csv"]
        
        self.save_csv(Fields=Fields,names=names)
