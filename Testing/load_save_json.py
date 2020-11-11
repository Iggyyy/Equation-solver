import json
from datetime import date
import numpy as np



def save_to_json(weights_0_1, weights_1_2, kernel_rows, kernel_cols ):
    """
    Save data to json
    """
    weights_0_1 = weights_0_1.tolist()
    weights_1_2 = weights_1_2.tolist()

    data = {}

    data['model_num'] = 'model' + str(date.today()) 
    data['weights_0_1'] = weights_0_1
    data['weights_1_2'] = weights_1_2
    data['kernel_properties'] = {
        'kernel_rows' : kernel_rows,
        'kernel_cols' : kernel_cols
    }

   

    name_str = r'Models\model_' + str(date.today()) + '.txt'

    with open(name_str, 'w') as outfile:
        json.dump(data, outfile)

def get_from_json(model_path):
    """
    Gets model parameters from json. Returns w_01, w_12, k_rows, k_cols, k_num.
    """

    with open(model_path) as json_file:
        data = json.load(json_file)

        print("Json loaded succesfully") if len(data) > 0 else print("Failed to load json")
            

        w_01 = data['weights_0_1']
        w_02 = data['weights_1_2']
        kernel = data['kernel_properties']
        k_rows = kernel['kernel_rows']
        k_cols = kernel['kernel_cols']
      
    
        return w_01, w_02, k_rows, k_cols



