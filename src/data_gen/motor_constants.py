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

# Simetria 120° (42 polos/36 ranhuras) — ver docstring de
# BLDC_FEMM_Model_Sym120 em motor_model.py para a matemática completa.
N_POLES_SECTOR = 14   # 120 / (360/42)
