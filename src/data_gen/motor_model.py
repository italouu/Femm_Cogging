import femm
import numpy as np
import pandas as pd
from pathlib import Path
from src.configs.datagen import DatagenConfig
from src.data_gen.motor_constants import (
    N_POLES_SECTOR as _MOTOR_N_POLES_SECTOR,
    WINDING_SECTOR as _MOTOR_WINDING_SECTOR,
    STACK_LENGTH_MM as _MOTOR_STACK_LENGTH_MM,
)
_dg = DatagenConfig()
DATASET                = _dg.dataset
DISTRIBUTION           = _dg.distribution

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / DATASET
DATA_DIR.mkdir(parents=True, exist_ok=True)

class BLDC_Process:

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
        # correntes de fase (2026-10-14): amplitude [A] e ângulo elétrico θe [graus] do
        # sistema trifásico equilibrado (ver phase_currents). Ausentes (CSV antigo) -> 0 A.
        self.current_amplitude     = float(motor_params.get('current_amplitude', {'value': 0.0})['value'])
        self.current_angle         = float(motor_params.get('current_angle', {'value': 0.0})['value'])

        self.material_iron ='iron_1008'
        self.material_mag = 'N35'
        self.material_copper= 'copper'
        self.material_gap = 'vacuum'

        self.slotoppener_ang = np.arccos(
            (2 * (self.stator_outer_diameter / 2 - self.slot_Hs0) ** 2 - self.slot_Bs0 ** 2)
            / (2 * (self.stator_outer_diameter / 2 - self.slot_Hs0) ** 2)
        )

    # ------------------------------------------------------------------
    # Geometria da bobina (usada pelo desenho do setor, BLDC_FEMM_Model_Sym120)
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
    def generate_samples(num_samples, seed=42):

        N_POLES   = 42
        PHASE_MAX = 2 * 360.0 / N_POLES

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
            # length -- sorteio MANTIDO só para preservar a sequência do gerador (e, com a
            # mesma semente, as geometrias do raw já gerado); o valor é descartado e o stack
            # length usado é o fixo STACK_LENGTH_MM.
            min = 5
            max = 10
            mode = min + 0.85 * (max - min)/2
            _ = _sample(min, max, mode)
            stack_length = _MOTOR_STACK_LENGTH_MM

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
    def add_phase_currents(samples, i_peak, seed=42):
        """Acrescenta as correntes de fase a `samples` (saída de qualquer generate_samples*),
        in place, e devolve `samples` (2026-10-14).

        Sistema trifásico equilibrado: amplitude I ~ U[0, i_peak] [A] e ângulo elétrico
        θe ~ U[0, 360) [graus] — colunas 'current_amplitude' e 'current_angle'. Garante
        |Ia|, |Ib|, |Ic| <= i_peak (sortear Ia, Ib independentes deixaria Ic chegar a
        2·i_peak). Gerador próprio (seed + rótulo fixo), sorteado depois da geometria:
        com o mesmo seed, geometria e rotor_phase saem idênticos aos de antes.
        """
        n = len(next(iter(samples.values()))['value'])
        rng = np.random.default_rng([seed, 0x1A])
        samples['current_amplitude'] = {'unit': 'A', 'value': list(rng.uniform(0.0, i_peak, n))}
        samples['current_angle'] = {'unit': 'deg', 'value': list(rng.uniform(0.0, 360.0, n))}
        return samples

    @staticmethod
    def phase_currents(amplitude, angle_deg):
        """{'A','B','C'} -> corrente [A]: Ia = I·cos θe, Ib = I·cos(θe − 120°),
        Ic = I·cos(θe + 120°) (Ia + Ib + Ic = 0). `+ 0.0` evita −0.0 (I = 0 dá o .ans
        idêntico ao de antes das correntes)."""
        th = np.deg2rad(angle_deg)
        return {ph: float(amplitude * np.cos(th - shift)) + 0.0
                for ph, shift in (('A', 0.0), ('B', 2 * np.pi / 3), ('C', -2 * np.pi / 3))}

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

# --------------------------------------------------------------------------- #
# Biblioteca de materiais do FEMM (matlib.dat) — reprodutibilidade (2026-10-07)
# --------------------------------------------------------------------------- #
# iron_1008/vacuum/copper foram inseridos à mão no matlib.dat da máquina
# original; numa instalação limpa do FEMM mi_getmaterial não os encontra.
# _femm_library_materials() lê o matlib.dat da instalação registrada no COM
# (a mesma que femm.openfemm() abre) e BLDC_FEMM_Model._get_material decide
# entre mi_getmaterial (biblioteca) e mi_addmaterial (FEMM_MATERIAL_DEFS).

_MATLIB_NUM_FIELDS = ('Mu_x', 'Mu_y', 'H_c', 'H_cAngle', 'J_re', 'J_im', 'Sigma', 'd_lam',
                      'Phi_h', 'Phi_hx', 'Phi_hy', 'LamType', 'LamFill', 'NStrands', 'WireD')
_MATLIB_CACHE = None


def _femm_matlib_path():
    """matlib.dat da instalação do FEMM registrada no COM (femm.ActiveFEMM ->
    LocalServer32 -> <femm>/bin/femm.exe). None se não achar."""
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r'femm.ActiveFEMM\CLSID') as k:
            clsid = winreg.QueryValue(k, None)
    except OSError:
        return None
    for root, sub in ((winreg.HKEY_CLASSES_ROOT, rf'CLSID\{clsid}\LocalServer32'),
                      (winreg.HKEY_LOCAL_MACHINE, rf'SOFTWARE\WOW6432Node\Classes\CLSID\{clsid}\LocalServer32'),
                      (winreg.HKEY_LOCAL_MACHINE, rf'SOFTWARE\Classes\CLSID\{clsid}\LocalServer32')):
        try:
            with winreg.OpenKey(root, sub) as k:
                exe = winreg.QueryValue(k, None).strip().strip('"')
        except OSError:
            continue
        p = Path(exe.split(' /')[0]).parent / 'matlib.dat'
        if p.is_file():
            return p
    return None


def _parse_matlib(path):
    """{nome: {campo: valor, 'bh': ((B,H),...)}} — primeira ocorrência de cada
    nome (pastas ignoradas, só os blocos <BeginBlock>..<EndBlock>)."""
    mats, cur, bh_left = {}, None, 0
    for raw in Path(path).read_text(encoding='latin-1').splitlines():
        line = raw.strip()
        if bh_left:
            b, h = line.split()[:2]
            cur['bh'].append((float(b), float(h)))
            bh_left -= 1
        elif line.startswith('<BeginBlock>'):
            cur = {'bh': []}
        elif cur is None:
            continue
        elif line.startswith('<EndBlock>'):
            name = cur.pop('name', None)
            if name is not None and name not in mats:
                cur['bh'] = tuple(cur['bh'])
                mats[name] = cur
            cur = None
        elif line.startswith('<BlockName>'):
            cur['name'] = line.split('=', 1)[1].strip().strip('"')
        elif line.startswith('<BHPoints>'):
            bh_left = int(line.split('=', 1)[1])
        elif line.startswith('<'):
            key = line[1:line.index('>')]
            if key in _MATLIB_NUM_FIELDS:
                cur[key] = float(line.split('=', 1)[1])
    return mats


def _femm_library_materials():
    """Materiais da biblioteca local do FEMM (cache por processo); {} se não achar."""
    global _MATLIB_CACHE
    if _MATLIB_CACHE is None:
        p = _femm_matlib_path()
        _MATLIB_CACHE = _parse_matlib(p) if p is not None else {}
    return _MATLIB_CACHE


def _same_material(lib, defn, rtol=1e-9):
    """Biblioteca == FEMM_MATERIAL_DEFS (todos os campos numéricos + curva BH)."""
    for f in _MATLIB_NUM_FIELDS:
        if not np.isclose(lib.get(f, 0.0), float(defn[f]), rtol=rtol, atol=0.0):
            return False
    a, b = np.asarray(lib['bh'], float), np.asarray(defn['bh'], float)
    return a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=0.0)


class BLDC_FEMM_Model(BLDC_Process):

    N_TURNS = 18  # número de espiras por bobina

    # True ignora a biblioteca local para os materiais de FEMM_MATERIAL_DEFS e
    # os define explicitamente (teste de reprodutibilidade).
    FORCE_EXPLICIT_MATERIALS = False

    def __init__(self, motor_params, phase):
        super().__init__(motor_params=motor_params)
        self.phase = phase
        self._explicit_materials = set()   # já criados via mi_addmaterial neste documento

    def _get_material(self, name):
        """Substitui femm.mi_getmaterial(name) (2026-10-07).

        - name na biblioteca local do FEMM -> mi_getmaterial (como antes; o .ans
          sai idêntico). Se name também estiver em FEMM_MATERIAL_DEFS, as
          propriedades da biblioteca precisam bater com a definição — senão
          ValueError (evita reproduzir com outra curva BH sem perceber).
        - fora da biblioteca (instalação limpa) ou FORCE_EXPLICIT_MATERIALS ->
          mi_addmaterial + mi_addbhpoint com FEMM_MATERIAL_DEFS, UMA vez por
          documento (o parser resolve material pelo índice em [BlockProps] de
          cada rótulo, classificado por Mu_x/curva BH — não precisa repetir a
          entrada a cada rótulo como o mi_getmaterial faz).
        """
        from src.data_gen.motor_constants import FEMM_MATERIAL_DEFS
        defn = FEMM_MATERIAL_DEFS.get(name)
        # FORCE simula instalação limpa: só os customizados (FEMM_MATERIAL_DEFS)
        # deixam de vir da biblioteca; N35 etc. continuam da biblioteca padrão.
        if self.FORCE_EXPLICIT_MATERIALS and defn is not None:
            lib = None
        else:
            lib = _femm_library_materials().get(name)
        if lib is not None:
            if defn is not None and not _same_material(lib, defn):
                raise ValueError(
                    f"material '{name}' da biblioteca do FEMM ({_femm_matlib_path()}) difere "
                    f"de FEMM_MATERIAL_DEFS (motor_constants.py) — renomeie/remova da "
                    f"biblioteca ou confira a definição")
            femm.mi_getmaterial(name)
            return
        if defn is None:
            raise ValueError(f"material '{name}' não está na biblioteca do FEMM nem em "
                             f"FEMM_MATERIAL_DEFS (motor_constants.py)")
        if name in self._explicit_materials:
            return
        femm.mi_addmaterial(name, defn['Mu_x'], defn['Mu_y'], defn['H_c'], defn['J_re'],
                            defn['Sigma'], defn['d_lam'], defn['Phi_h'], defn['LamFill'],
                            defn['LamType'], defn['Phi_hx'], defn['Phi_hy'],
                            defn['NStrands'], defn['WireD'])
        for b, h in defn['bh']:
            femm.mi_addbhpoint(name, b, h)
        self._explicit_materials.add(name)
          


        
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
            self._get_material(material)
            femm.mi_addblocklabel(r_avg * np.cos(ang_avg),r_avg * np.sin(ang_avg))
            femm.mi_selectlabel(r_avg * np.cos(ang_avg),r_avg * np.sin(ang_avg))
            femm.mi_setblockprop(material, 0, 1, "", np.rad2deg(ang_avg) + direction, 0, 0)
            femm.mi_clearselected()


class BLDC_FEMM_Model_Sym120(BLDC_FEMM_Model):
    """Desenha só o setor 0-120 graus do motor (simetria periódica), em vez do
    círculo completo (360 graus) — contorno periódico (BdryFormat=4) nos dois
    cortes radiais em vez de A=0 em toda a fronteira externa. BLDC_FEMM_Model
    só fornece os materiais e os primitivos de desenho (_create_sec,
    _create_coils); o desenho do motor inteiro (360°) saiu na limpeza de
    2026-10-21 (sem uso).

    Promovido de tests/proto_femm_sym120.py (validado 2026-07-22, promovido
    2026-07-23) — mesma classe, mesmo nome, sem alterações de geometria.

    Por que 120 graus é o período correto (42 polos / 36 ranhuras):
      - passo de polo = 360/42 = 8.571 graus; passo de ranhura = 360/36 = 10 graus
      - período geométrico combinado = 360/gcd(42,36) = 60 graus, mas 60 graus
        contém 7 polos (ímpar) -> troca de polaridade (anti-periódico)
      - 120 graus contém 14 polos (par, sem troca de sinal) e 12 ranhuras
        (12*10=120 exato) -> período periódico (não anti-periódico) mínimo

    O corte em theta=0/120 cai exatamente no CENTRO de uma ranhura (ranhura 0 e
    ranhura "12"=0+120). A geometria da bobina (_coil_coords) já é construída
    como dois lóbulos espelhados (vp/vn) em torno dessa linha central, então
    o corte é feito desenhando só o lóbulo vp da ranhura 0 e só o lóbulo vn
    da ranhura 12 — sem precisar recortar o perfil arredondado da ranhura.

    Os polos NÃO precisam ser cortados: o polo 0 já começa exatamente em
    theta=0 (pole_offset=0 no modelo completo), então 14 polos completos
    fecham exatamente em 120 graus.
    [ATUALIZADO 2026-10-05] o parágrafo acima só vale para phase=0. Até essa
    data draw_motor ignorava self.phase (polos com phase=0 fixo — herança do
    protótipo de 2026-07-22, teste isolado com fase 0), então rotor_phase
    sorteado nunca chegava à simulação. Agora os polos são girados pela fase
    (_pole_pieces): o polo que cruza 120° é partido em dois pedaços, um em cada
    ponta do setor, com a magnetização do polo inteiro; o corte periódico da
    faixa dos ímãs se adapta (modo "poles" em _draw_periodic_cuts). Validado
    em tests/inspect_femm_phase_sym120.py (wrap exato nos cortes, |ΔA|~1e-12).
    """

    N_POLES_SECTOR = _MOTOR_N_POLES_SECTOR   # 120 / (360/42) == 14
    N_SLOTS_SECTOR = 12   # 120 / (360/36)
    _PERIODIC = "periodic_120"

    def _bore_start_radius(self):
        """Raio onde o domínio começa na direção radial interna. 0.0 = vai
        até o centro (modelo completo, só recorta angularmente). Subclasses
        podem sobrescrever para truncar também na direção radial."""
        return 0.0

    def draw_motor(self):
        # --- rotor: back iron num único arco 0-120 (fecha com segmentos radiais) ---
        self._create_sec(r_in=self.rotor_inner_diameter / 2 + self.pole_thickness,
                          r_ext=self.rotor_outer_diameter / 2,
                          ang_in_1=0, ang_in_2=120,
                          material=self.material_iron)

        # --- polos 0..13 girados pela fase do rotor, dentro do setor ---
        # _create_sec usa magdir = centro do PEDAÇO desenhado + direction; o
        # pedaço de um polo partido no corte recebe a magnetização do polo
        # INTEIRO (direction corrigido pela diferença de centros).
        self._pole_edges_at_cut = set()   # cortes (0/120) onde já existe aresta de polo
        for a1, a2, center, direction, _ in self._pole_pieces():
            self._create_sec(r_in=self.rotor_inner_diameter / 2,
                              r_ext=self.rotor_inner_diameter / 2 + self.pole_thickness,
                              ang_in_1=a1, ang_in_2=a2,
                              material=self.material_mag,
                              direction=direction + (center - (a1 + a2) / 2), phase=0)
            for a in (a1, a2):
                if a == 0.0 or a == self._SECTOR_DEG:
                    self._pole_edges_at_cut.add(a)

        # --- bobinas (ranhura 0 só vp, ranhuras 1..11 completas, ranhura 12 só vn) ---
        self._create_coils_sector()
        self._coil_props_sector()

        # --- estator ---
        self._create_stator_sector()

        # --- contorno externo A=0 (um único arco 0-120) ---
        self._set_boundary_sector()

        # --- cortes periódicos: por último, pra poder reaproveitar arestas
        #     já desenhadas (polo 0 e o lóbulo da bobina) ---
        self._draw_periodic_cuts()

    _SECTOR_DEG = 120.0
    _POLE_CUT_TOL_DEG = 0.01   # pedaço de polo mais estreito que isso é colado no corte

    def _pole_pieces(self):
        """Polos do setor girados pela fase do rotor (2026-10-05).

        Polo k em θ = (k·passo + phase) mod 120°, largura passo·pole_embrance,
        polaridade alternada pela paridade de k (14 polos, número par — a
        periodicidade de 120° vale para qualquer fase). Um polo que cruza 120°
        vira dois pedaços, [θ, 120°] e [0°, θ+largura−120°]; pedaços mais
        estreitos que _POLE_CUT_TOL_DEG são colados no corte (evita elemento
        degenerado). Com phase=0 dá exatamente os polos do desenho antigo.

        Retorna [(a1, a2, centro_do_polo, direction, k)] em graus, a1/a2 em
        [0, 120]; centro_do_polo já trazido para o referencial do pedaço (−120°
        no pedaço do início do setor).
        """
        sector, tol = self._SECTOR_DEG, self._POLE_CUT_TOL_DEG
        step = 360 / self.number_rotor_poles
        width = step * self.pole_embrance
        pieces = []
        for k in range(self.N_POLES_SECTOR):
            a1 = (step * k + self.phase) % sector
            a2 = a1 + width
            center = a1 + width / 2
            direction = 0 if k % 2 == 0 else 180
            if sector - a1 < tol:                      # pedaço do fim minúsculo: polo todo no início
                a1, a2, center = 0.0, a2 - sector, center - sector
            elif 0.0 < a1 < tol:                       # começa logo depois de 0°: encosta no corte
                a1 = 0.0                               # (fresta de vácuo de ~µm explodia a malha —
                                                       #  sample 943: 0,0018° -> 78k nós; 2026-10-05)
            if abs(a2 - sector) < tol:                 # termina no corte: encosta exatamente em 120°
                pieces.append((a1, sector, center, direction, k))
            elif a2 < sector:                          # cabe inteiro no setor
                pieces.append((a1, a2, center, direction, k))
            else:                                      # cruza 120°: dois pedaços
                pieces.append((a1, sector, center, direction, k))
                pieces.append((0.0, a2 - sector, center - sector, direction, k))
        return pieces

    # ------------------------------------------------------------------
    # bobinas / ranhuras
    # ------------------------------------------------------------------

    def _create_coil_half(self, ang_deg, side):
        """Desenha só um lóbulo (vp ou vn) da bobina — usado nas 2 ranhuras
        de fronteira do setor (0 e 120), que são cortadas ao meio."""
        ang_rad = np.deg2rad(ang_deg)
        x, yp, yn = self._coil_coords()
        y = yp if side == 'vp' else yn
        v = np.column_stack((x, y))
        R = np.array([[np.cos(ang_rad), -np.sin(ang_rad)],
                      [np.sin(ang_rad), np.cos(ang_rad)]])
        v = v @ R.T
        for i in range(len(v) - 1):
            femm.mi_drawline(v[i, 0], v[i, 1], v[i + 1, 0], v[i + 1, 1])

    def _create_coils_sector(self):
        step_ang = 360 / self.number_stator_slots  # usa o pitch ORIGINAL (36 ranhuras)
        self._create_coil_half(ang_deg=0, side='vp')                # ranhura 0: só vp
        for i in range(1, self.N_SLOTS_SECTOR):                     # ranhuras 1..11: completas
            self._create_coils(ang=step_ang * i)
        self._create_coil_half(ang_deg=120, side='vn')              # ranhura 12(=0+120): só vn

    def _coil_props_sector(self):
        """Propriedades das bobinas do setor (loop até N_SLOTS_SECTOR, não 36).

        Rodar o loop só até 12 (em vez de 36) já produz exatamente a
        combinação certa: coil_i_0 (i=0..11) cobre os lóbulos vp das
        ranhuras 0..11; coil_i_1 (i=0..11) cobre os lóbulos vn das
        ranhuras 1..12 — ou seja, ranhura 0 só ganha label vp e ranhura
        12 só ganha label vn, batendo com _create_coils_sector acima.
        """
        n_turns = self.N_TURNS
        r = self.stator_outer_diameter / 2 - (self.slot_Hs0 + self.slot_Hs1 + self.slot_Hs2 / 2)
        step = 2 * np.pi / self.number_stator_slots
        ang_offset = step / 2 - self.slotoppener_ang / 2
        # correntes de fase (2026-10-14): bobina do dente i = coil_i_0 (+sinal·I_fase) e
        # coil_i_1 (−sinal·I_fase), tabela WINDING_SECTOR (motor_constants.py)
        i_ph = self.phase_currents(self.current_amplitude, self.current_angle)

        for i in range(self.N_SLOTS_SECTOR):
            ph, sign = _MOTOR_WINDING_SECTOR[i]
            femm.mi_addcircprop('coil_' + str(i) + '_0', +sign * i_ph[ph] + 0.0, 1)
            femm.mi_addcircprop('coil_' + str(i) + '_1', -sign * i_ph[ph] + 0.0, 1)

            ang = step * i + step / 2 - ang_offset
            self._get_material(self.material_copper)
            femm.mi_addblocklabel(r * np.cos(ang), r * np.sin(ang))
            femm.mi_selectlabel(r * np.cos(ang), r * np.sin(ang))
            femm.mi_setblockprop(self.material_copper, 0, 1, 'coil_' + str(i) + '_0', 0, 0, n_turns)
            femm.mi_clearselected()

            ang = step * i + step / 2 + ang_offset
            self._get_material(self.material_copper)
            femm.mi_addblocklabel(r * np.cos(ang), r * np.sin(ang))
            femm.mi_selectlabel(r * np.cos(ang), r * np.sin(ang))
            femm.mi_setblockprop(self.material_copper, 0, 1, 'coil_' + str(i) + '_1', 0, 0, n_turns)
            femm.mi_clearselected()

    # ------------------------------------------------------------------
    # estator
    # ------------------------------------------------------------------

    def _create_stator_sector(self):
        res = 1
        r_ext = self.stator_outer_diameter / 2
        r_in = r_ext - self.slot_Hs0
        step = 2 * np.pi / self.number_stator_slots
        tooth_ang = step - self.slotoppener_ang

        for i in range(self.N_SLOTS_SECTOR):
            ang_1 = step * i + self.slotoppener_ang / 2
            ang_2 = ang_1 + tooth_ang
            femm.mi_drawarc(r_ext * np.cos(ang_1), r_ext * np.sin(ang_1),
                             r_ext * np.cos(ang_2), r_ext * np.sin(ang_2),
                             np.rad2deg(np.abs(ang_1 - ang_2)), res)
            femm.mi_drawline(r_ext * np.cos(ang_1), r_ext * np.sin(ang_1),
                              r_in * np.cos(ang_1), r_in * np.sin(ang_1))
            femm.mi_drawline(r_ext * np.cos(ang_2), r_ext * np.sin(ang_2),
                              r_in * np.cos(ang_2), r_in * np.sin(ang_2))

        # label do "gap" (vácuo) entre estator e rotor — 90 graus cai dentro do setor
        ang_avg = np.pi / 2
        r_avg = (self.rotor_inner_diameter + self.stator_outer_diameter) / 4
        self._get_material(self.material_gap)
        femm.mi_addblocklabel(r_avg * np.cos(ang_avg), r_avg * np.sin(ang_avg))
        femm.mi_selectlabel(r_avg * np.cos(ang_avg), r_avg * np.sin(ang_avg))
        femm.mi_setblockprop(self.material_gap, 0, 1, "", np.rad2deg(ang_avg), 0, 0)
        femm.mi_clearselected()

        # círculo interno do estator (furo do eixo) — um único arco 0-120
        r_in2 = self.stator_inner_diameter / 2
        femm.mi_drawarc(r_in2 * np.cos(0), r_in2 * np.sin(0),
                         r_in2 * np.cos(np.deg2rad(120)), r_in2 * np.sin(np.deg2rad(120)),
                         120, res)

        # label de vácuo dentro da cunha do furo (FEMM exige material em toda
        # região fechada — sem label = erro "material not defined", não vira
        # buraco automaticamente); igual ao (0,0) do modelo original
        r_bore_lab = (self._bore_start_radius() + r_in2) / 2
        ang_bore = np.deg2rad(60)
        self._get_material(self.material_gap)
        femm.mi_addblocklabel(r_bore_lab * np.cos(ang_bore), r_bore_lab * np.sin(ang_bore))
        femm.mi_selectlabel(r_bore_lab * np.cos(ang_bore), r_bore_lab * np.sin(ang_bore))
        femm.mi_setblockprop(self.material_gap, 0, 1, "", 0, 0, 0)
        femm.mi_clearselected()

        ang_avg = np.deg2rad(60)
        r_avg = (self.stator_inner_diameter / 2 + self.stator_outer_diameter / 2
                 - (self.slot_Hs0 + self.slot_Hs1 + self.slot_Hs2)) / 2
        self._get_material(self.material_iron)
        femm.mi_addblocklabel(r_avg * np.cos(ang_avg), r_avg * np.sin(ang_avg))
        femm.mi_selectlabel(r_avg * np.cos(ang_avg), r_avg * np.sin(ang_avg))
        femm.mi_setblockprop(self.material_iron, 0, 1, "", np.rad2deg(ang_avg), 0, 0)
        femm.mi_clearselected()

    # ------------------------------------------------------------------
    # contorno externo + cortes periódicos
    # ------------------------------------------------------------------

    def _set_boundary_sector(self):
        res = 1
        femm.mi_addboundprop("A=0", 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)

        r = self.outer_diameter / 2
        femm.mi_drawarc(r * np.cos(0), r * np.sin(0),
                         r * np.cos(np.deg2rad(120)), r * np.sin(np.deg2rad(120)),
                         120, res)
        femm.mi_selectarcsegment(r * np.cos(np.deg2rad(60)), r * np.sin(np.deg2rad(60)))
        femm.mi_setarcsegmentprop(1, "A=0", 0, 0)
        femm.mi_clearselected()

        r_in = self.outer_diameter / 2 * 0.99
        theta = np.deg2rad(60)
        self._get_material(self.material_gap)
        femm.mi_addblocklabel(r_in * np.cos(theta), r_in * np.sin(theta))
        femm.mi_selectlabel(r_in * np.cos(theta), r_in * np.sin(theta))
        femm.mi_setblockprop(self.material_gap, 0, 1, "", 0, 0, 0)
        femm.mi_clearselected()

        # arco interno A=0, se o domínio for truncado radialmente (subclasse)
        r_bore = self._bore_start_radius()
        if r_bore > 0:
            femm.mi_drawarc(r_bore * np.cos(0), r_bore * np.sin(0),
                             r_bore * np.cos(np.deg2rad(120)), r_bore * np.sin(np.deg2rad(120)),
                             120, res)
            femm.mi_selectarcsegment(r_bore * np.cos(np.deg2rad(60)), r_bore * np.sin(np.deg2rad(60)))
            femm.mi_setarcsegmentprop(1, "A=0", 0, 0)
            femm.mi_clearselected()

    def _draw_radial_cut(self, r1, r2, ang_deg, propname):
        """mi_addsegment só conecta pontos que já existem como nó — se
        nenhuma geometria anterior tocou esse extremo (ex: a origem), a
        chamada falha em silêncio sem criar nada. mi_addnode garante o
        nó antes; se o ponto já existir (extremo de um arco desenhado
        antes), é um no-op seguro (não duplica)."""
        ang = np.deg2rad(ang_deg)
        x1, y1 = r1 * np.cos(ang), r1 * np.sin(ang)
        x2, y2 = r2 * np.cos(ang), r2 * np.sin(ang)
        femm.mi_addnode(x1, y1)
        femm.mi_addnode(x2, y2)
        femm.mi_addsegment(x1, y1, x2, y2)
        self._apply_periodic_existing(r1, r2, ang_deg, propname)

    def _apply_periodic_existing(self, r1, r2, ang_deg, propname):
        ang = np.deg2rad(ang_deg)
        rm = (r1 + r2) / 2
        femm.mi_selectsegment(rm * np.cos(ang), rm * np.sin(ang))
        femm.mi_setsegmentprop(propname, 0, 1, 0, 0)
        femm.mi_clearselected()

    def _draw_periodic_cuts(self):
        """Cada par (theta=0, theta=120) na MESMA faixa radial precisa de um
        nome de propriedade periódica PRÓPRIO — o FEMM só aceita exatamente
        2 segmentos por propriedade periódica (erro 'assigned to more than
        two segments' se reusar o mesmo nome em várias faixas radiais)."""
        r_bs0 = (self.stator_outer_diameter / 2 - self.slot_Hs0) * np.cos(self.slotoppener_ang / 2)
        slot_bottom = r_bs0 - self.slot_Hs1 - self.slot_Hs2

        # (nome, r1, r2, precisa_desenhar_novo_segmento)
        ranges = [
            # furo do eixo até o furo do estator, e daí até o fundo da ranhura —
            # separados pq o segmento único 0->slot_bottom cruzaria o círculo do
            # furo do estator (r=stator_inner_diameter/2) e o FEMM o fragmenta
            # (fragmentos desiguais nos 2 lados -> erro "dissimilar segments")
            ("per_bore_hole", self._bore_start_radius(),              self.stator_inner_diameter / 2, True),
            ("per_core", self.stator_inner_diameter / 2,              slot_bottom,                    True),
            ("per_slot_bottom", slot_bottom,                         r_bs0,                           False),  # reaproveita aresta do lóbulo
            ("per_slot_throat", r_bs0,                                self.stator_outer_diameter / 2,  True),
            ("per_airgap", self.stator_outer_diameter / 2,           self.rotor_inner_diameter / 2,    True),
            ("per_pole_ring", self.rotor_inner_diameter / 2,          self.rotor_inner_diameter / 2 + self.pole_thickness, "poles"),  # reaproveita aresta de polo que cair no corte
            ("per_back_iron", self.rotor_inner_diameter / 2 + self.pole_thickness, self.rotor_outer_diameter / 2, "existing"),  # já desenhado por _create_sec
            ("per_outer_margin", self.rotor_outer_diameter / 2,      self.outer_diameter / 2,          True),
        ]

        for name, r1, r2, mode in ranges:
            femm.mi_addboundprop(name, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0)
            if mode is True:
                # sem geometria prévia em nenhum dos dois ângulos: desenha os 2
                self._draw_radial_cut(r1, r2, 0, name)
                self._draw_radial_cut(r1, r2, 120, name)
            elif mode is False:
                # aresta já existe nos 2 ângulos (lóbulo vp/vn da bobina)
                self._apply_periodic_existing(r1, r2, 0, name)
                self._apply_periodic_existing(r1, r2, 120, name)
            elif mode == "pole0":
                # theta=0: aresta do polo 0 já existe; theta=120: precisa de novo segmento
                self._apply_periodic_existing(r1, r2, 0, name)
                self._draw_radial_cut(r1, r2, 120, name)
            elif mode == "poles":
                # faixa dos ímãs: aresta de polo no corte (registrada em draw_motor,
                # _pole_edges_at_cut) é reaproveitada; senão (vão entre polos) desenha
                for ang in (0.0, self._SECTOR_DEG):
                    if ang in self._pole_edges_at_cut:
                        self._apply_periodic_existing(r1, r2, ang, name)
                    else:
                        self._draw_radial_cut(r1, r2, ang, name)
            elif mode == "existing":
                # back iron: _create_sec já desenhou os 2 segmentos (seg_1/seg_2), só falta a propriedade
                self._apply_periodic_existing(r1, r2, 0, name)
                self._apply_periodic_existing(r1, r2, 120, name)


class BLDC_FEMM_Model_Sym120_Annular(BLDC_FEMM_Model_Sym120):
    """Setor de 120 graus + truncamento radial em [inner_diameter/2,
    outer_diameter/2] — a mesma janela já usada para amostrar os dados hoje
    (grade H×W dos parsers), em vez de ir até o centro (r=0). Só sobrescreve
    _bore_start_radius — o resto da geometria (arco interno A=0, label de
    vácuo na fatia remanescente, corte periódico até esse raio) já é
    parametrizado no pai.

    Promovido de tests/proto_femm_sym120_annular.py (validado 2026-07-22,
    promovido 2026-07-23). Usada como base do método de geração via malha
    real do FEMM (ver src/data_gen/femm_solve.py).
    """

    def _bore_start_radius(self):
        return self.inner_diameter / 2
