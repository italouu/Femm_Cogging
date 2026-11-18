"""
motor_constants.py
-------------------
Constantes de material e geometria do motor BLDC — sem NENHUMA dependência
de projeto (não importa femm/shapely/matplotlib/DatagenConfig/etc.), só
literais Python puros.

Separado de motor_model.py porque aquele importa `femm` (COM, só Windows): os
parsers (src/data_gen/parsers/) e o pós-processamento leem estas constantes e rodam
sem FEMM.

motor_model.py importa daqui (BLDC_FEMM_Model_Sym120.N_POLES_SECTOR, N_SLOTS_SECTOR,
_SECTOR_DEG, WINDING_SECTOR, STACK_LENGTH_MM, BLDC_Process.phase_currents) em vez de literais
duplicados -- fonte única. phase_currents usa numpy (import dentro da função).
"""

# Propriedades de materiais — fonte única.
MAGNETIZATION = {'iron_1008': 0,      'N35p': 1,    'N35n': -1,  'copper': 0,     'vacuum': 0  }
PERMEABILITY  = {'iron_1008': 5000.0, 'N35p': 1.05, 'N35n': 1.05,'copper': 0.999, 'vacuum': 1.0}
# N35p e N35n unificados como ima=2; ferro=0, ar=1, bobina=3
MATERIAL_ID   = {'iron_1008': 0,      'N35p': 2,    'N35n': 2,   'copper': 3,     'vacuum': 1  }

# Definição completa (no FEMM) dos materiais que NÃO existem na biblioteca
# padrão do FEMM 4.2. Valores iguais aos do [BlockProps] do raw.
# Usados por BLDC_FEMM_Model._get_material (motor_model.py) via mi_addmaterial
# + mi_addbhpoint quando o material não está na biblioteca local; se estiver,
# a biblioteca precisa bater com estes valores (senão erro).
# Chaves = campos do matlib.dat/.ans; Sigma em MS/m; bh = (B [T], H [A/m]).
# 'N35' fica de fora: vem da biblioteca padrão (Hard Magnetic Materials).
FEMM_MATERIAL_DEFS = {
    'iron_1008': dict(
        Mu_x=1.0, Mu_y=1.0, H_c=0.0, H_cAngle=0.0, J_re=0.0, J_im=0.0, Sigma=0.0,
        d_lam=0.0, Phi_h=0.0, Phi_hx=0.0, Phi_hy=0.0, LamType=0, LamFill=1.0,
        NStrands=0, WireD=0.0,
        bh=((0.0, 0.0), (0.2402, 159.2), (0.8654, 318.3), (1.1106, 477.5),
            (1.2458, 636.6), (1.331, 795.8), (1.5, 1591.5), (1.6, 3183.1),
            (1.683, 4774.6), (1.741, 6366.2), (1.78, 7957.7), (1.905, 15915.5),
            (2.025, 31831.0), (2.085, 47746.5), (2.13, 63662.0), (2.165, 79577.5),
            (2.28, 159155.0), (2.485, 318310.0), (2.5851, 397887.0)),
    ),
    'vacuum': dict(
        Mu_x=1.0, Mu_y=1.0, H_c=0.0, H_cAngle=0.0, J_re=0.0, J_im=0.0, Sigma=0.0,
        d_lam=0.0, Phi_h=0.0, Phi_hx=0.0, Phi_hy=0.0, LamType=0, LamFill=1.0,
        NStrands=0, WireD=0.0, bh=(),
    ),
    'copper': dict(
        Mu_x=0.999991, Mu_y=0.999991, H_c=0.0, H_cAngle=0.0, J_re=0.0, J_im=0.0, Sigma=0.0,
        d_lam=0.0, Phi_h=0.0, Phi_hx=0.0, Phi_hy=0.0, LamType=0, LamFill=1.0,
        NStrands=0, WireD=0.0, bh=(),
    ),
}

# Comprimento axial do pacote (stack length) [mm] -- FIXO. Fonte única: coluna stack_length do
# valid_designs.csv, profundidade do problema planar (mi_probdef) e L do torque pelo campo.
STACK_LENGTH_MM = 15.0


def stack_length_mm(motor_params: dict) -> float:
    """Stack length [mm] de um design (motor_params {chave: {'unit', 'value'}}); fallback
    STACK_LENGTH_MM se a chave não existir."""
    sl = motor_params.get('stack_length')
    return float(sl['value']) if sl is not None else STACK_LENGTH_MM


# Simetria 120° (42 polos/36 ranhuras) — ver docstring de
# BLDC_FEMM_Model_Sym120 em motor_model.py para a matemática completa.
N_POLES = 42          # polos do rotor (máquina inteira) -> 21 pares
N_SLOTS = 36          # ranhuras do estator (máquina inteira)
SECTOR_DEG = 120.0    # setor periódico modelado
N_POLES_SECTOR = 14   # 120 / (360/42)
N_SLOTS_SECTOR = round(N_SLOTS * SECTOR_DEG / 360.0)   # 12
SLOT_PITCH_DEG = 360.0 / N_SLOTS                       # 10°; ranhura i centrada em i·passo (corte 0° no centro da ranhura 0)


def phase_currents(amplitude, angle_deg):
    """{'A','B','C'} -> corrente [A]: Ia = I·cos θe, Ib = I·cos(θe − 120°),
    Ic = I·cos(θe + 120°) (Ia + Ib + Ic = 0). `+ 0.0` evita −0.0 (com I = 0 o .ans
    sai idêntico ao de um modelo sem circuitos). Fonte única: motor_model
    (BLDC_Process.phase_currents) e o pós-processamento (flux_linkage_direct).
    numpy importado aqui dentro para o módulo continuar sem dependências no import."""
    import numpy as np
    th = np.deg2rad(angle_deg)
    return {ph: float(amplitude * np.cos(th - shift)) + 0.0
            for ph, shift in (('A', 0.0), ('B', 2 * np.pi / 3), ('C', -2 * np.pi / 3))}

# Enrolamento do setor 120°: 1 célula 12 ranhuras / 14 polos, camada dupla
# concentrada. Dente k -> (fase, sinal); a bobina do dente k são os circuitos
# coil_k_0 (ida, +sinal·I_fase, lado de θ menor) e coil_k_1 (volta, −sinal·I_fase).
# Layout clássico do 12/14 com B <-> C trocados, para que ABC com θe crescente gire
# o campo em +θ, mesmo sentido de rotor_phase crescente (θe = +21·θ_rotor).
WINDING_SECTOR = (('A', +1), ('A', -1), ('C', -1), ('C', +1), ('B', +1), ('B', -1),
                  ('A', -1), ('A', +1), ('C', +1), ('C', -1), ('B', -1), ('B', +1))
