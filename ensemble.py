
import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
# from gnn import GNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
# import matplotlib.pyplot as plt
import numpy as np

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
import torch.nn as nn
import torch.nn.functional as F
### importing evaluator
from dataset_pyg import PygGraphPropPredDataset
from evaluate import Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def eval(ensembleNet, device, loader, evaluator):
    ensembleNet.eval()

    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = ensembleNet(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        # 각 모델별로 실제값과 예측값을 병합
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

####for three models 
# class MyEnsemble(torch.nn.Module):
#     def __init__(self, modelA, modelB, modelC):
#         super(MyEnsemble, self).__init__()
#         self.modelA = modelA
#         self.modelB = modelB
#         self.modelC = modelC
#         # This combiner will combine the two outputs into a single output
#         self.combiner = torch.nn.Linear(3, 1)  # Accepts a concatenated vector from out1 and out2

#     def forward(self, x):
#         out1 = self.modelA(x)  # torch.Size([batch_size, 1])
#         out2 = self.modelB(x)  # torch.Size([batch_size, 1])
#         out3 = self.modelC(x)
#         # Concatenating along the last dimension which is the feature dimension in this case
#         combined_out = torch.cat((out1, out2,out3), dim=1)  # Now has size [batch_size, 2]

#         # Pass the combined output through the linear combiner to get the final output
#         out = self.combiner(combined_out)  # Now has size [batch_size, 1]

#         return out



class MyEnsemble(torch.nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # This combiner will combine the two outputs into a single output
        self.combiner = torch.nn.Linear(2, 1)  # Accepts a concatenated vector from out1 and out2

    def forward(self, x):
        out1 = self.modelA(x)  # torch.Size([batch_size, 1])
        out2 = self.modelB(x)  # torch.Size([batch_size, 1])

        # Concatenating along the last dimension which is the feature dimension in this case
        combined_out = torch.cat((out1, out2), dim=1)  # Now has size [batch_size, 2]
        
        # Pass the combined output through the linear combiner to get the final output
        out = self.combiner(combined_out)  # Now has size [batch_size, 1]

        return out



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
    parser.add_argument('--epochs', type=int, default=20,
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

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_dataset = dataset[split_idx["train"]]
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    # Compute in-degree histogram over training data.
    deg = torch.zeros(80, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    # Create models and load state_dicts    
    modelA = pna.Net(deg=deg, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    modelB = sage.Net(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    modelC= rgcn.Net(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    models = [modelA, modelB, modelC]
    PATH_pna = 'model/dfg_cp_pna_layer_5_model_epoch100.pt'
    PATH_sage = 'model/dfg_cp_sage_layer_5_model_epoch100.pt'
    PATH_rgcn = 'model/dfg_cp_rgcn_layer_5_model_epoch100.pt'
    PATHS = [PATH_pna,PATH_sage, PATH_rgcn]
    

    for i in range(len(models)):
        checkpoint = torch.load(PATHS[i])
        models[i].load_state_dict(checkpoint['model_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
    for model in models:
      for param in model.parameters():
          param.requires_grad = False


    task_type = "regression"
    ensembleNet = MyEnsemble(modelA, modelC).to(device)
    optimizer = optim.Adam(ensembleNet.parameters(), lr=0.001) 
    scheduler= ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, min_lr=0.00001)

    valid_curve = []
    test_curve = []
    train_curve = []

    test_predict_value= []
    test_true_value= []
    valid_predict_value= []
    valid_true_value= []

    # Add early stopping
    early_stopping_patience = 5  # max epochs to wait when no improvement 
    early_stopping_counter = 0  # current counter 
    best_val_loss = float('inf')  
    best_epoch = 0  

    train_rmse_scores = []
    valid_rmse_scores = []
    test_rmse_scores = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        ensemble_train(ensembleNet, device, train_loader, optimizer, task_type)

        print('Evaluating...')
        train_perf, _, _ = eval(ensembleNet, device, train_loader, evaluator)
        valid_perf, v_true,  v_pred= eval(ensembleNet, device, valid_loader, evaluator)
        test_perf, t_true, t_pred = eval(ensembleNet, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        scheduler.step(valid_perf[dataset.eval_metric])
        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        test_predict_value.append(reduce(operator.add, t_pred.tolist()))
        valid_predict_value.append(reduce(operator.add, v_pred.tolist()))

        test_loss=test_perf[dataset.eval_metric]

        if test_loss<=np.min(np.array(test_curve)):
            PATH='ensemble_dsp/'+args.dataset + '_sage_rgcn_'+ str(args.num_layer)+'epoch_'+str(epoch)+'+model.pt'
            torch.save({'epoch': epoch,
                        'model_state_dict': ensembleNet.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': test_loss
                        }, PATH)

        # for early stopping 
        current_train_loss = train_perf[dataset.eval_metric]
        current_val_loss = valid_perf[dataset.eval_metric]
        current_test_loss = test_perf[dataset.eval_metric]

        # Store scores
        train_rmse_scores.append(current_train_loss)
        valid_rmse_scores.append(current_val_loss)
        test_rmse_scores.append(current_test_loss)
        # if current_val_loss < best_val_loss:
        #     best_val_loss = current_val_loss
        #     best_epoch = epoch
        #     early_stopping_counter = 0

        #     best_model_path = 'ensemble_model/' + args.dataset + '_pna_rgcn_sage_' + str(args.num_layer) + 'epoch_' + str(epoch) + '+best_model.pt'
        #     torch.save({'epoch': epoch, 'model_state_dict': ensembleNet.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': best_val_loss}, best_model_path)
        # else:
        #     early_stopping_counter += 1
        #     if early_stopping_counter >= early_stopping_patience:
        #         print(f'Early stopping triggered at epoch {epoch}')
        #         break

        elapsed_time = time.time() - start_time
        print('Time elapsed for epoch {}: {:.2f} seconds'.format(epoch, elapsed_time))
    


    test_true_value=reduce(operator.add, t_true.tolist())
    valid_true_value=reduce(operator.add, v_true.tolist())

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    f = open('ensemble_result/'+args.dataset + '_rgcn_sage_'+str(args.num_layer)+'.json', 'w')
    result=dict(val=valid_curve[best_val_epoch], \
        test=test_curve[best_val_epoch],train=train_curve[best_val_epoch], \
        test_pred=test_predict_value, value_pred=valid_predict_value, 
        test_true=test_true_value, valid_true=valid_true_value,\
        train_curve=train_curve, test_curve=test_curve, valid_curve=valid_curve)
    json.dump(result, f)
    f.close()

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)