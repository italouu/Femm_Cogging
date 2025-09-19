import os, time, shutil
import femm
from data_gen.motor_model import BLDC_Process, BLDC_FEMM_Model, BLDC_Shapely_Model

def generate_one_case(motor_params_list, code, n_samples, n_r, n_a, ang_1, ang_2, temp_path):

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

    temp = os.path.join(temp_path, f"cogging_{code}")
    os.makedirs(temp, exist_ok=True)
    femm.mi_saveas(os.path.join(temp, f"cogging_{code}.fem"))
    femm.mi_createmesh()

    femm.mi_analyze()
    femm.mi_loadsolution()

    model.save_B_grid(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=code)
    femm.closefemm()
    shutil.rmtree(temp)

    ######### shapely ###########

    model = BLDC_Shapely_Model(motor_params)
    model.draw_motor()

    model.save_material_grid(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=code)

    model.save_material_mu_avg(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=code)
    
    end = time.time()
    print(f"Iteração {code+1}: {end - start:.4f} s")

def generate_one_batch(motor_params_list,
                       temp_path, 
                       code_list, 
                       n_r, 
                       n_a,
                       ang_1 = 0, 
                       ang_2 = 120):
    
    # Importar femm dentro do processo filho
    import femm
    from data_gen.motor_model import BLDC_Process, BLDC_FEMM_Model, BLDC_Shapely_Model

    for code in code_list:
        # Temp exclusivo
        tmp_dir = os.path.abspath(os.path.join(temp_path, f"tmp_{code}"))
        os.makedirs(tmp_dir, exist_ok=True)

        fem_file = os.path.join(tmp_dir, f"model_{code}.fem")

        cwd0 = os.getcwd()

        os.chdir(tmp_dir)

        motor_params = BLDC_Process.extract_params_at_index(motor_params=motor_params_list,code=code)

        ######### FEMM ###########

        model = BLDC_FEMM_Model(motor_params=motor_params)

        femm.openfemm('bHide') # disable gui with 'bHide'
        femm.main_resize(1000, 1000)
        femm.newdocument(0)
        femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, 0, 200)

        model.draw_motor()

        femm.mi_refreshview()
        femm.mi_zoomnatural()
        femm.mi_saveas(fem_file)
        femm.mi_createmesh()

        femm.mi_analyze()
        femm.mi_loadsolution()

        model.save_B_grid(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=code)
        femm.closefemm()

        ######### shapely ###########
        
        model = BLDC_Shapely_Model(motor_params)
        model.draw_motor()

        model.save_material_grid(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=code)
        model.save_material_mu_avg(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=code)

        os.chdir(cwd0)
        shutil.rmtree(tmp_dir, ignore_errors=True)