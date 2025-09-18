from data_gen.motor_model import BLDC_Process, BLDC_FEMM_Model, BLDC_Shapely_Model
import femm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import plot_grid, plot_ang_grid
import time

if __name__ == "__main__":

    ######### params ############

    ang_1 = 0
    ang_2 = 120
    r_in = 55/2
    r_ext = 90/2
    n_r = 80
    n_a = 240
    code = 0
    n_samples = 1

    data_path = 'data_gen/Raw_Data/'

    ######### samples ###########
    motor_params_list = BLDC_Process.generate_samples(num_samples=n_samples)
    BLDC_Process.export_params(params=motor_params_list)

    for code in range(n_samples):

        start = time.time()
        print(f'modelo {code+1}/{n_samples}')

        ######### Model ##########

        motor_params = BLDC_Process.extract_params_at_index(motor_params=motor_params_list,code=code)

        ######### FEMM ###########

        model = BLDC_FEMM_Model(motor_params=motor_params)
        femm.openfemm() # disable gui with 'bHide'
        femm.main_resize(1000, 1000)
        femm.newdocument(0)
        femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, 0, 200)

        model.draw_motor()

        femm.mi_refreshview()
        femm.mi_zoomnatural()
        femm.mi_saveas("cogging.fem")
        femm.mi_createmesh()

        femm.mi_analyze()
        femm.mi_loadsolution()

        model.save_B_grid(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=code)
        femm.closefemm()

        ######## Field Plot ##########

        # Bx = pd.read_csv(data_path+f"Mag_Bx_{code}.csv", header=None).values
        # By = pd.read_csv(data_path+f"Mag_By_{code}.csv", header=None).values

        # Bmag = np.sqrt(Bx**2 + By**2)

        # plot_grid(Field=Bmag)

        # plot_ang_grid(Field=Bmag, r_in=r_in, r_ext=r_ext, ang_1=ang_1, ang_2=ang_2)

        ######### shapely ###########

        model = BLDC_Shapely_Model(motor_params)
        model.draw_motor()
        # fig, ax = model.plot()
        # plt.show(block=False)
        # plt.pause(2)
        # plt.close(fig)

        model.save_material_grid(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=code)

        # Material = pd.read_csv(data_path+f"Material_{code}.csv", header=None).values

        # plot_grid(Field=Material)
        
        # plot_ang_grid(Field=Material, r_in=r_in, r_ext=r_ext, ang_1=ang_1, ang_2=ang_2)


        model.save_material_mu_avg(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=code)

        # Material = pd.read_csv(data_path+f"Mu_avg_{code}.csv", header=None).values

        # plot_grid(Field=Material)
        
        # plot_ang_grid(Field=Material, r_in=r_in, r_ext=r_ext, ang_1=ang_1, ang_2=ang_2)
        
        end = time.time()
        print(f"Iteração {code+1}: {end - start:.4f} s")
