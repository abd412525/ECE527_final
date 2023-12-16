import json
import numpy as np


json_path = "inference/inf_dfg_dsp_pna_layer_5.json"

with open(json_path, 'r') as file:
    data = json.load(file)

test_true = data['test_true']
test_pred = data['test_pred']

mape = np.mean([abs((true - pred) / true) for true, pred in zip(test_true, test_pred) if true != 0]) * 100

print(f'MAPE: {mape}%')