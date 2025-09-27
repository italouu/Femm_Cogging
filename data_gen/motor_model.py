import femm
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import pandas as pd
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "raw_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

class BLDC_Process:
    def __init__(self, motor_params = {}):
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

        self.material_iron='iron_1008'
        self.material_mag = 'N35'
        self.material_copper= 'copper'
        self.material_gap= 'vacuum'

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

        self.data_path = 'raw_data/'

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
    
    @staticmethod
    def generate_samples(num_samples, seed=42):

        space = { 
            'number_rotor_poles':     {'unit': '',   'x_min': 42,  'x_max': 42},
            'number_stator_slots':    {'unit': '',   'x_min': 36,  'x_max': 36},
            'stator_outer_diameter':  {'unit': 'mm', 'x_min': 0,   'x_max': 0},
            'stator_inner_diameter':  {'unit': 'mm', 'x_min': 55,  'x_max': 55},
            'stack_length':           {'unit': 'mm', 'x_min': 0,   'x_max': 0},
            'slot_Hs0':               {'unit': 'mm', 'x_min': 0,   'x_max': 0},
            'slot_Hs1':               {'unit': 'mm', 'x_min': 0,   'x_max': 0},
            'slot_Hs2':               {'unit': 'mm', 'x_min': 0,   'x_max': 0},
            'slot_Bs0':               {'unit': 'mm', 'x_min': 1,   'x_max': 0},
            'slot_Bs1':               {'unit': 'mm', 'x_min': 0,   'x_max': 0},
            'slot_Bs2':               {'unit': 'mm', 'x_min': 0,   'x_max': 0},
            'slot_Rs':                {'unit': 'mm', 'x_min': 0,   'x_max': 0},
            'rotor_outer_diameter':   {'unit': 'mm', 'x_min': 90,  'x_max': 90},
            'rotor_inner_diameter':   {'unit': 'mm', 'x_min': 0,   'x_max': 0},
            'pole_embrance':          {'unit': '',   'x_min': 0.6, 'x_max': 0.95},
            'pole_thickness':         {'unit': 'mm', 'x_min': 0,   'x_max': 0}
            }
        
        rng = np.random.default_rng(seed)
        samples = {key: {'unit': cfg['unit'], 'value': []} for key, cfg in space.items()}

        for _ in range(num_samples):
            # length
            min = 5
            max = 10
            mode = min + 0.85 * (max - min)/2
            stack_length = rng.triangular(left=min,right=max,mode=mode)

            # stator outer diameter
            min_rotor = 4 # min rotor thickness
            min_gap = 0.5
            max_gap = 2

            max_stator = space['rotor_outer_diameter']['x_min'] - 2*(min_rotor + min_gap)
            min_stator = (space['stator_inner_diameter']['x_min'] + max_stator)/2
            
            min = min_stator
            max = max_stator
            mode = min + 0.8 * (max - min)
            stator_outer_diameter = rng.triangular(left=min,right=max,mode=mode)
            
            # rotor inner diameter
            min = min_gap
            max = max_gap
            mode = min + 0.3 * (max - min)
            gap = rng.triangular(left=min,right=max,mode=mode)
            
            rotor_inner_diameter = stator_outer_diameter + 2*gap
            rotor_outer_diameter = space['rotor_outer_diameter']['x_min']

            stator_inner_diameter = space['stator_inner_diameter']['x_min']

            # poles    
            min_back_iron = 1
            min_pole_thickness = 1.5

            min = min_pole_thickness
            max = rotor_outer_diameter/2 - rotor_inner_diameter/2 - min_back_iron
            mode = min + 0.8 * (max - min)
            pole_thickness = rng.triangular(left=min,right=max,mode=mode)

            min = space['pole_embrance']['x_min']
            max = space['pole_embrance']['x_max']
            mode = min + 0.8 * (max - min)
            pole_embrance = rng.triangular(left=min,right=max,mode=mode)

            # slots (height)
            # stator yoke
            n_slots = space['number_stator_slots']['x_min']
            slot_arc = 2*np.pi * stator_outer_diameter/2 / n_slots

            h = (stator_outer_diameter - stator_inner_diameter)/2

            min = 0.10 * h
            max = 0.30 * h
            mode = min + 0.5 * (max - min)
            stator_yoke = rng.triangular(left=min,right=max,mode=mode)

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
            slot_Bs0 = rng.triangular(left=min,right=max,mode=mode)

            Hs = slot_Hs0 + slot_Hs1
            slot_arc = 2*np.pi * (stator_outer_diameter/2 - Hs) / n_slots      
            min = 0.3 * slot_arc
            max = 0.5 * slot_arc
            mode = min + 0.5 * (max - min)
            slot_Bs1 = rng.triangular(left=min,right=max,mode=mode)

            Hs = slot_Hs0 + slot_Hs1 + slot_Hs2
            slot_arc = 2*np.pi * (stator_outer_diameter/2 - Hs) / n_slots  
            min = 0.40 * slot_arc
            max = 0.65 * slot_arc
            mode = min + 0.5 * (max - min)
            slot_Bs2 = rng.triangular(left=min,right=max,mode=mode)

            number_stator_slots = space['number_stator_slots']['x_min']
            number_rotor_poles = space['number_rotor_poles']['x_min']
            slot_Rs = 0

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

        r = self.rotor_outer_diameter/2 *1.1
        r_in = self.rotor_outer_diameter/2 *1.05
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

        ang = np.deg2rad(ang)
        r_bs0 = (self.stator_outer_diameter/2 - self.slot_Hs0) * np.cos(self.slotoppener_ang/2)

        xp = np.array([r_bs0,
            r_bs0 - self.slot_Hs1,
            r_bs0 - self.slot_Hs1 - self.slot_Hs2,
            r_bs0 - self.slot_Hs1 - self.slot_Hs2,
            r_bs0,
            r_bs0])
        
        xn = np.array([r_bs0,
            r_bs0 - self.slot_Hs1,
            r_bs0 - self.slot_Hs1 - self.slot_Hs2,
            r_bs0 - self.slot_Hs1 - self.slot_Hs2,
            r_bs0,
            r_bs0])
        
        yp = np.array([self.slot_Bs0/2,
            self.slot_Bs1/2,
            self.slot_Bs2/2,
            0,
            0,
            self.slot_Bs0/2])
        
        yn = -np.array([self.slot_Bs0/2,
            self.slot_Bs1/2,
            self.slot_Bs2/2,
            0,
            0,
            self.slot_Bs0/2])

        vp = np.column_stack((xp, yp))
        vn = np.column_stack((xn, yn))

        R = np.array([[np.cos(ang), -np.sin(ang)],
                    [np.sin(ang),  np.cos(ang)]])

        vp = vp @ R.T
        vn = vn @ R.T

        for i in range(len(vp)-1):
            femm.mi_drawline(vp[i, 0], vp[i, 1], vp[i+1, 0], vp[i+1, 1])
            femm.mi_drawline(vn[i, 0], vn[i, 1], vn[i+1, 0], vn[i+1, 1])

        # femm.mi_createradius(vp[1, 0], vp[1, 1],self.slot_Rs)
        # femm.mi_createradius(vn[1, 0], vn[1, 1],self.slot_Rs)

        # femm.mi_createradius(vp[2, 0], vp[2, 1],self.slot_Rs)
        # femm.mi_createradius(vn[2, 0], vn[2, 1],self.slot_Rs)

    def _coil_props(self):
        n_turns = 18

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

        r_in = self.stator_inner_diameter/2
        r_ext = self.rotor_outer_diameter/2
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

        # air gap
        # self.air_gap = self._create_air_gap()
        # self.geometries['vacuum'].append(self.air_gap)
        # self.plot(self.geometries)

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
        # mesmos pontos do original
        ang_rad = np.deg2rad(ang)
        r_bs0 = (self.stator_outer_diameter/2 - self.slot_Hs0) * np.cos(self.slotoppener_ang/2)

        xp = np.array([r_bs0,
            r_bs0 - self.slot_Hs1,
            r_bs0 - self.slot_Hs1 - self.slot_Hs2,
            r_bs0 - self.slot_Hs1 - self.slot_Hs2,
            r_bs0,
            r_bs0])

        xn = np.array([r_bs0,
            r_bs0 - self.slot_Hs1,
            r_bs0 - self.slot_Hs1 - self.slot_Hs2,
            r_bs0 - self.slot_Hs1 - self.slot_Hs2,
            r_bs0,
            r_bs0])

        yp = np.array([self.slot_Bs0/2,
            self.slot_Bs1/2,
            self.slot_Bs2/2,
            0,
            0,
            self.slot_Bs0/2])

        yn = -np.array([self.slot_Bs0/2,
            self.slot_Bs1/2,
            self.slot_Bs2/2,
            0,
            0,
            self.slot_Bs0/2])

        vp = np.column_stack((xp, yp))
        vn = np.column_stack((xn, yn))

        # rotação (mesma matriz do original)
        R = np.array([[np.cos(ang_rad), -np.sin(ang_rad)],
                      [np.sin(ang_rad),  np.cos(ang_rad)]])
        vp = vp @ R.T
        vn = vn @ R.T

        # polígonos (fechados) equivalentes às linhas do FEMM
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

    def save_magnetization(self, ang_1, ang_2, n_r, n_a, code = 0):
        M = {
            'iron_1008': 0,
            'N35p': 1,
            'N35n': -1,
            'copper': 0,
            'vacuum': 0
        }
        
        r_in = self.stator_inner_diameter/2
        r_ext = self.rotor_outer_diameter/2
        ang_1 = np.deg2rad(ang_1)
        ang_2 = np.deg2rad(ang_2)

        ang_step = np.abs(ang_1-ang_2) / (n_a+1)
        
        Field = np.empty((n_r, n_a), dtype=float)
        field_list = []
        
        res = 20
        allowed = {'N35p', 'N35n'}

        for x, y, r, th in self._iter_polar_points(
                r_in=r_in, r_ext=r_ext,
                ang_1=ang_1, ang_2=ang_2,
                n_r=n_r, n_a=n_a):

            radius = r * ang_step/3
            disk = Point(x, y).buffer(radius, resolution=res)
            disk_area = disk.area
            area_dict = {m: 0.0 for m in allowed}

            for material, geoms in self.geometries.items():
                if material not in allowed:
                    continue
                for geom in geoms:
                    inter = disk.intersection(geom)
                    if not inter.is_empty:
                        area_dict[material] += inter.area

            sum_area = sum(area_dict.values())
            area_dict['vacuum'] = max(disk_area - sum_area, 0.0)

            M_avg = sum(M[m] * a for m, a in area_dict.items()) / disk_area
            field_list.append(M_avg)

        Field[:] = np.asarray(field_list, dtype=float).reshape(n_r, n_a)

        Fields = [Field]
        names = [f"Magnetization_{code}.csv"]
        
        self.save_csv(Fields=Fields,names=names)

    def save_material_mu_avg(self, ang_1, ang_2, n_r, n_a, code = 0):
        mu = {
            'iron_1008': 5000.0,
            'N35p': 1.05,
            'N35n': 1.05,
            'copper': 0.999,
            'vacuum': 1.0
        }
        
        r_in = self.stator_inner_diameter/2
        r_ext = self.rotor_outer_diameter/2
        ang_1 = np.deg2rad(ang_1)
        ang_2 = np.deg2rad(ang_2)

        ang_step = np.abs(ang_1-ang_2) / (n_a+1)
        
        Field = np.empty((n_r, n_a), dtype=float)
        field_list = []

        res = 20
        for x, y, r, th in self._iter_polar_points(
                r_in=r_in, r_ext=r_ext,
                ang_1=ang_1, ang_2=ang_2,
                n_r=n_r, n_a=n_a):

            radius = r * ang_step/3
            disk = Point(x, y).buffer(radius, resolution=res)
            disk_area = disk.area
            area_dict = {mat: 0.0 for mat in self.geometries}

            for material, geoms in self.geometries.items():
                for geom in geoms:
                    inter = disk.intersection(geom)
                    if not inter.is_empty:
                        area_dict[material] += inter.area

            sum_area = sum(area_dict.values())
            area_dict['vacuum'] = max(disk_area - sum_area, 0.0)

            mu_avg = sum(mu[m] * a for m, a in area_dict.items()) / disk_area
            field_list.append(mu_avg)

        Field[:] = np.asarray(field_list, dtype=float).reshape(n_r, n_a)

        Fields = [Field]
        names = [f"Mu_avg_{code}.csv"]
        
        self.save_csv(Fields=Fields,names=names)

    def save_material_grid(self, ang_1, ang_2, n_r, n_a, code = 0):
        material_table = {
            'iron_1008': 6,
            'N35p': 1,
            'N35n': 5,
            'copper': 3,
            'vacuum': 0
        }
        
        r_in = self.stator_inner_diameter/2
        r_ext = self.rotor_outer_diameter/2
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

