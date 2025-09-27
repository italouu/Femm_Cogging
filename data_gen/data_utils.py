import os, shutil
import pandas as pd
import torch

def generate_one_batch(motor_params_list,
                       code_list, 
                       n_r, 
                       n_a,
                       ang_1 = 0, 
                       ang_2 = 120,
                       n_phases = 1):
    
    import femm
    from data_gen.motor_model import BLDC_Process, BLDC_FEMM_Model, BLDC_Shapely_Model
    import random

    #### phase ####
    ini = 0
    end = 360 / int(motor_params_list['number_rotor_poles']['value'][0]) * 1

    temp_path = 'data_gen/temp/'

    for code in code_list:
        
        phases = [random.uniform(ini, end) for _ in range(n_phases)]
        
        for i, phase in enumerate(phases):
            tmp_dir = os.path.abspath(os.path.join(temp_path, f"tmp_{code}"))
            os.makedirs(tmp_dir, exist_ok=True)

            fem_file = os.path.join(tmp_dir, f"model_{code}.fem")

            cwd0 = os.getcwd()

            os.chdir(tmp_dir)

            motor_params = BLDC_Process.extract_params_at_index(motor_params=motor_params_list,code=code)
            
            lcl_code = f"{code}_p{i}"

            ######### FEMM ###########
            
            model = BLDC_FEMM_Model(motor_params=motor_params, phase=phase)
            
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

            model.save_B_grid(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=lcl_code)
            
            femm.closefemm()

            ######### shapely ###########
            
            model = BLDC_Shapely_Model(motor_params=motor_params, phase=phase)
            model.draw_motor()

            model.save_material_grid(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=lcl_code)
            model.save_material_mu_avg(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=lcl_code)
            model.save_magnetization(ang_1=ang_1, ang_2=ang_2, n_r=n_r, n_a=n_a, code=lcl_code)

            os.chdir(cwd0)
            shutil.rmtree(tmp_dir, ignore_errors=True)

def check_data(n_phases = 1):
    data_path = 'data_gen/raw_data/'
    models = pd.read_csv(data_path+"valid_designs.csv", header=None).values

    n_models = (len(models)-1)
    n_data = n_models * n_phases
    prefixes = ["Mag_Bx_","Mag_By_","Magnetization_","Material_","Mu_avg_"]
    ext = ".csv"

    missing_models = set()

    for m in range(n_models):
        for p in range(n_phases):
            for pref in prefixes:
                fname = f"{pref}{m}_p{p}{ext}"
                if not os.path.exists(os.path.join(data_path, fname)):
                    missing_models.add(m)
                    break
            if m in missing_models:
                break

    return list(missing_models)

def csv_to_pt(n_phases, x_prefixes=["Mu_avg_","Magnetization_"], y_prefixes=["Mag_Bx_","Mag_By_"]):
    
    dtype=torch.float32

    data_path = 'data_gen/raw_data/'
    save_path = 'data_gen/torch_data'
    out_file  = 'data_test.pt'

    models = pd.read_csv(os.path.join(data_path, "valid_designs.csv"), header=None).values
    n_models = len(models) - 1
    samples_x, samples_y = [], []

    for model_idx in range(n_models):
        for phase in range(n_phases):

            x_chs = []
            for pref in x_prefixes:
                fpath = os.path.join(data_path, f"{pref}{model_idx}_p{phase}.csv")
                arr = pd.read_csv(fpath, header=None).values
                x_chs.append(torch.tensor(arr, dtype=dtype))

            y_chs = []
            for pref in y_prefixes:
                fpath = os.path.join(data_path, f"{pref}{model_idx}_p{phase}.csv")
                arr = pd.read_csv(fpath, header=None).values
                y_chs.append(torch.tensor(arr, dtype=dtype))

            x_tensor = torch.stack(x_chs, dim=0)
            y_tensor = torch.stack(y_chs, dim=0)

            samples_x.append(x_tensor)
            samples_y.append(y_tensor)

    X = torch.stack(samples_x, dim=0)
    Y = torch.stack(samples_y, dim=0)

    out_file = os.path.join(save_path, out_file)
    torch.save({"x": X, "y": Y}, out_file)
    print(f"{out_file} | x: {tuple(X.shape)}, y: {tuple(Y.shape)}")

def load_tensor(file  = 'data_test.pt'):
    data_path = 'data_gen/torch_data'

    fpath = os.path.join(data_path, file)

    data = torch.load(fpath)
    x = data['x']
    y = data['y']

    return x, y

def duplicate_phases(n_phases = 3):
    data_path = 'data_gen/raw_data/'
    models = pd.read_csv(data_path+"valid_designs.csv", header=None).values

    n_models = (len(models)-1)
    prefixes = ["Mag_Bx_","Mag_By_","Magnetization_","Material_","Mu_avg_"]
    ext = ".csv"

    # Para cada modelo
    for model_id in range(n_models):
        # Para cada prefixo (tipo de dado)
        for prefix in prefixes:
            # Ler as fases originais (0 até n_phases-1)
            for p in range(n_phases):
                fname = f"{prefix}{model_id}_p{p}{ext}"
                fpath = os.path.join(data_path, fname)
                if not os.path.exists(fpath):
                    continue  # pula se o arquivo não existir

                # Carrega o CSV
                df = pd.read_csv(fpath, header=None)

                # Inverte as colunas
                df_inv = df.iloc[:, ::-1]

                # Nova fase = p + n_phases
                new_phase = p + n_phases
                new_fname = f"{prefix}{model_id}_p{new_phase}{ext}"
                new_fpath = os.path.join(data_path, new_fname)

                # Salva duplicata
                df_inv.to_csv(new_fpath, index=False, header=False)
                print(f"Duplicado: {fname} -> {new_fname}")