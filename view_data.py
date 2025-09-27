from utils import plot_grid, plot_ang_grid
import pandas as pd
import numpy as np

ang_1 = 0
ang_2 = 120
r_in = 55/2
r_ext = 90/2
n_r = 80
n_a = 240

data_path = 'data_gen/Raw_Data/'

code = "0_p3"

Bx = pd.read_csv(data_path+f"Mag_Bx_{code}.csv", header=None).values

plot_ang_grid(Field=Bx, r_in=r_in, r_ext=r_ext, ang_1=ang_1, ang_2=ang_2)

By = pd.read_csv(data_path+f"Mag_By_{code}.csv", header=None).values

plot_ang_grid(Field=By, r_in=r_in, r_ext=r_ext, ang_1=ang_1, ang_2=ang_2)

Bmag = np.sqrt(Bx**2 + By**2)

plot_ang_grid(Field=Bmag, r_in=r_in, r_ext=r_ext, ang_1=ang_1, ang_2=ang_2)

Material = pd.read_csv(data_path+f"Material_{code}.csv", header=None).values

plot_ang_grid(Field=Material, r_in=r_in, r_ext=r_ext, ang_1=ang_1, ang_2=ang_2)

Mu_avg = pd.read_csv(data_path+f"Mu_avg_{code}.csv", header=None).values

plot_ang_grid(Field=Mu_avg, r_in=r_in, r_ext=r_ext, ang_1=ang_1, ang_2=ang_2)

Magnetization = pd.read_csv(data_path+f"Magnetization_{code}.csv", header=None).values

plot_ang_grid(Field=Magnetization, r_in=r_in, r_ext=r_ext, ang_1=ang_1, ang_2=ang_2)
