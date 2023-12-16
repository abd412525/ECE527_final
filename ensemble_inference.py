import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
# from gnn import GNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree

from tqdm import tqdm
import argparse
import time
import numpy as np
import json
import operator
from functools import reduce

import ARMA
import film
import gat
import pna
import pan
import sage
import sgn
import unet
import rgcn
import ggnn

### importing evaluator
from dataset_pyg import PygGraphPropPredDataset
from evaluate import Evaluator
from ensemble import MyEnsemble

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def eval(ensembleNet, device, loader, evaluator):
    ensembleNet.eval()

    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            continue
        else:
            with torch.no_grad():
                pred = ensembleNet(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())


    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict), y_true, y_pred





def ensemble_train(ensemble, device, loader, optimizer, task_type):
    ensemble.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            ensemble_pred = ensemble(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(ensemble_pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(ensemble_pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]) 
            loss.backward()
            optimizer.step()

def save_models(models, optimizers, epoch, test_loss, test_curve, args):
    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        if test_loss <= np.min(np.array(test_curve)):
            PATH = f'ensemble_model/{args.dataset}_model_{i}_layer_{args.num_layer}_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss
            }, PATH)


if __name__ == "__main__":
        # Training settings
    parser = argparse.ArgumentParser(description='Ensemble model')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="dfg_dsp",
                        help='dataset name (default: lut)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)
    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = 0)

    deg = torch.zeros(80, dtype=torch.long)
    train_dataset = dataset[split_idx["train"]]
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())


    # Create models and load state_dicts    
    modelA = pna.pnaNet(deg=deg, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    modelB = rgcn.rgcnNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    modelC = sage.Net(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)

    task_type = "regression"
    ensembleNet = MyEnsemble(modelA, modelB).to(device)
    optimizer = optim.Adam(ensembleNet.parameters(), lr=0.001) 
    scheduler= ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, min_lr=0.00001) 


    # PATH='model/'+args.dataset + '_'+ args.gnn+ '_layer_'+ str(args.num_layer)+'_model.pt'
    # PATH='ensemble_model/dfg_cp_pna_rgcn_sage_5_epoch_100_model.pt'
    PATH = 'test_model/dfg_cp_pna_rgcn_5epoch_17+model.pt'
    
    checkpoint = torch.load(PATH)
    ensembleNet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    test_perf, t_true, t_pred = eval(ensembleNet, device, test_loader, evaluator)
    
    test_true_value=reduce(operator.add, t_true.tolist())
    test_pred_value=reduce(operator.add, t_pred.tolist())


    file_path = 'ensemble_inference/inf_' + args.dataset + '_pna_rgcn' + '_layer_' + str(args.num_layer) + '.json'

    
    with open(file_path, 'w') as f:
        result = dict(test_true=test_true_value, test_pred=test_pred_value)
        json.dump(result, f)
