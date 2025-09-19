from data_gen.motor_model import BLDC_Process, BLDC_FEMM_Model, BLDC_Shapely_Model
import femm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import plot_grid, plot_ang_grid
import os, time, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_gen.generate_utils import generate_one_batch, generate_one_case
from functools import partial

## single process

def sigle_process():

    for code in range(n_samples):

        generate_one_case(motor_params_list=motor_params_list, 
                    code=code, 
                    n_samples=n_samples, 
                    n_r=n_r, 
                    n_a=n_a,
                    ang_1 = ang_1, 
                    ang_2 = ang_2,
                    temp_path=temp_path)

def mult_process():

    ######### Multi Process ##########

    all_codes = list(range(n_samples))
    num_workers = os.cpu_count()

    chunks = [all_codes[i::num_workers] for i in range(num_workers)]

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = [
            ex.submit(
                generate_one_batch,
                motor_params_list=motor_params_list,
                temp_path=temp_path,
                code_list=chunk,
                n_r=n_r,
                n_a=n_a,
                ang_1=ang_1,
                ang_2=ang_2,
            )
            for chunk in chunks
        ]
        for f in as_completed(futs):
            f.result()


if __name__ == "__main__":
    ######### params ############
    ang_1 = 0
    ang_2 = 120
    r_in = 55/2
    r_ext = 90/2
    n_r = 80
    n_a = 240
    n_samples = 10

    temp_path = 'data_gen/temp/'

    ######### samples ###########
    motor_params_list = BLDC_Process.generate_samples(num_samples=n_samples,seed=12)
    BLDC_Process.export_params(params=motor_params_list)

    mult_process()