"""
motor_constants.py
-------------------
Constantes de material e geometria do motor BLDC — sem NENHUMA dependência
de projeto (não importa femm/shapely/matplotlib/DatagenConfig/etc.), só
literais Python puros.

Extraído de src/data_gen/motor_model.py em 2026-08-13: MATERIAL_ID/
PERMEABILITY/MAGNETIZATION viviam como atributos de classe de BLDC_Process,
e N_POLES_SECTOR de BLDC_FEMM_Model_Sym120 — mas motor_model.py tem
`import femm` na primeira linha (necessário só pras classes que desenham no
FEMM de verdade), então importar essas classes só pra ler 3 dicts/1 inteiro
arrastava femm (+ matplotlib/shapely/pandas) junto. Isso quebrava a promessa
de src/data_gen/parsers/ans_parsing.py e
src/data_gen/parsers/femm_mesh_v2.py de rodar 100% sem FEMM (ver docstring
desses módulos e CLAUDE.md "Raw -- malha real do FEMM v2") -- eles usavam
BLDC_Process/BLDC_FEMM_Model_Sym120_Annular só por essas constantes.

motor_model.py continua sendo a fonte de definição (BLDC_Process.MATERIAL_ID
etc. e BLDC_FEMM_Model_Sym120.N_POLES_SECTOR seguem existindo, com o mesmo
valor, pra não quebrar nada que já lê por ali) -- só importa esses valores
daqui em vez de literais duplicados, pra manter fonte única.
"""

# Propriedades de materiais — fonte única (BLDC_Process reimporta essas 3).
MAGNETIZATION = {'iron_1008': 0,      'N35p': 1,    'N35n': -1,  'copper': 0,     'vacuum': 0  }
PERMEABILITY  = {'iron_1008': 5000.0, 'N35p': 1.05, 'N35n': 1.05,'copper': 0.999, 'vacuum': 1.0}
# N35p e N35n unificados como ima=2; ferro=0, ar=1, bobina=3
MATERIAL_ID   = {'iron_1008': 0,      'N35p': 2,    'N35n': 2,   'copper': 3,     'vacuum': 1  }

# Definição completa (no FEMM) dos materiais que NÃO existem na biblioteca
# padrão do FEMM 4.2 — foram inseridos à mão no matlib.dat da máquina original
# (D:\femm42\bin\matlib.dat). Valores copiados do matlib.dat e conferidos
# idênticos ao [BlockProps] do raw oficial mesh_ans_138x276 (2026-10-07).
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

# Simetria 120° (42 polos/36 ranhuras) — ver docstring de
# BLDC_FEMM_Model_Sym120 em motor_model.py para a matemática completa.
N_POLES_SECTOR = 14   # 120 / (360/42)
