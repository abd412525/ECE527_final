import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from ARMA import Net as armaNet
from film import Net as filmNet
from gat import Net as gatNet
from ggnn import Net as ggnnNet
from gnn import GNN as gnnNet
from pan import Net as panNet
from pna import Net as pnaNet
from sage import Net as sageNet
from sgn import Net as sgnNet
from unet import Net as unetNet
from rgcn import Net as rgcnNet
from torch_geometric.utils import degree
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric.transforms as T

from tqdm import tqdm
import argparse
import time
import numpy as np
import json
import operator
from functools import reduce

from dataset_pyg import PygGraphPropPredDataset
from evaluate import Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict), y_true, y_pred

def main():
    parser = argparse.ArgumentParser(description='Ensemble methods')

    parser.add_argument('--gnn1', type=str, default="rgcn",
                        help="first GNN to ensemble")
    parser.add_argument('--gnn2', type=str, default="pna",
                        help="second GNN to ensemble")
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
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="dfg_dsp_binary",
                        help='dataset name')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

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

    gnn1_model = None
    gnn1_optimizer = None
    gnn1_scheduler = None
    gnn2_model = None
    gnn2_optimizer = None
    gnn2_scheduler = None

    if args.gnn1 == 'arma':
        gnn1_model = armaNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn1_optimizer = optim.Adam(gnn1_model.parameters(), lr=0.0005)
        gnn1_scheduler = ReduceLROnPlateau(gnn1_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn1 == 'film':
        gnn1_model = filmNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn1_optimizer = optim.Adam(gnn1_model.parameters(), lr=0.0005)
        gnn1_scheduler = ReduceLROnPlateau(gnn1_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn1 == 'gat':
        head=8
        gnn1_model = gatNet(heads=head, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn1_optimizer = optim.Adam(gnn1_model.parameters(), lr=0.0005)
        gnn1_scheduler = ReduceLROnPlateau(gnn1_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn1 == 'ggnn':
        gnn1_model = ggnnNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn1_optimizer = optim.Adam(gnn1_model.parameters(), lr=0.0005)
        gnn1_scheduler = ReduceLROnPlateau(gnn1_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn1 == 'gin':
        gnn1_model = gnnNet(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        gnn1_optimizer = optim.Adam(gnn1_model.parameters(), lr=0.001)
        gnn1_scheduler = ReduceLROnPlateau(gnn1_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn1 == 'gin-virtual':
        gnn1_model = gnnNet(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
        gnn1_optimizer = optim.Adam(gnn1_model.parameters(), lr=0.001)
        gnn1_scheduler = ReduceLROnPlateau(gnn1_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn1 == 'gcn':
        gnn1_model = gnnNet(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        gnn1_optimizer = optim.Adam(gnn1_model.parameters(), lr=0.001)
        gnn1_scheduler = ReduceLROnPlateau(gnn1_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn1 == 'gcn-virtual':
        gnn1_model = gnnNet(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
        gnn1_optimizer = optim.Adam(gnn1_model.parameters(), lr=0.001)
        gnn1_scheduler = ReduceLROnPlateau(gnn1_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn1 == 'pan':
        gnn1_model = panNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn1_optimizer = optim.Adam(gnn1_model.parameters(), lr=0.0005)
        gnn1_scheduler = ReduceLROnPlateau(gnn1_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn1 == 'pna':
        gnn1_model = pnaNet(deg=deg, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn1_optimizer = optim.Adam(gnn1_model.parameters(), lr=0.001)
        gnn1_scheduler = ReduceLROnPlateau(gnn1_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn1 == 'rgcn':
        gnn1_model = rgcnNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn1_optimizer = optim.Adam(gnn1_model.parameters(), lr=0.001, weight_decay=1e-5)
        gnn1_scheduler = ReduceLROnPlateau(gnn1_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn1 == 'sage':
        gnn1_model = sageNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn1_optimizer = optim.Adam(gnn1_model.parameters(), lr=0.0005)
        gnn1_scheduler = ReduceLROnPlateau(gnn1_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn1 == 'sgn':
        gnn1_model = sgnNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn1_optimizer = optim.Adam(gnn1_model.parameters(), lr=0.0005)
        gnn1_scheduler = ReduceLROnPlateau(gnn1_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn1 == 'unet':
        gnn1_model = unetNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn1_optimizer = optim.Adam(gnn1_model.parameters(), lr=0.0005)
        gnn1_scheduler = ReduceLROnPlateau(gnn1_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)

    if args.gnn2 == 'arma':
        gnn2_model = armaNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn2_optimizer = optim.Adam(gnn2_model.parameters(), lr=0.0005)
        gnn2_scheduler = ReduceLROnPlateau(gnn2_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn2 == 'film':
        gnn2_model = filmNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn2_optimizer = optim.Adam(gnn2_model.parameters(), lr=0.0005)
        gnn2_scheduler = ReduceLROnPlateau(gnn2_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn2 == 'gat':
        head=8
        gnn2_model = gatNet(heads=head, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn2_optimizer = optim.Adam(gnn2_model.parameters(), lr=0.0005)
        gnn2_scheduler = ReduceLROnPlateau(gnn2_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn2 == 'ggnn':
        gnn2_model = ggnnNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn2_optimizer = optim.Adam(gnn2_model.parameters(), lr=0.0005)
        gnn2_scheduler = ReduceLROnPlateau(gnn2_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn2 == 'gin':
        gnn2_model = gnnNet(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        gnn2_optimizer = optim.Adam(gnn2_model.parameters(), lr=0.001)
        gnn2_scheduler = ReduceLROnPlateau(gnn2_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn2 == 'gin-virtual':
        gnn2_model = gnnNet(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
        gnn2_optimizer = optim.Adam(gnn2_model.parameters(), lr=0.001)
        gnn2_scheduler = ReduceLROnPlateau(gnn2_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn2 == 'gcn':
        gnn2_model = gnnNet(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        gnn2_optimizer = optim.Adam(gnn2_model.parameters(), lr=0.001)
        gnn2_scheduler = ReduceLROnPlateau(gnn2_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn2 == 'gcn-virtual':
        gnn2_model = gnnNet(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
        gnn2_optimizer = optim.Adam(gnn2_model.parameters(), lr=0.001)
        gnn2_scheduler = ReduceLROnPlateau(gnn2_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn2 == 'pan':
        gnn2_model = panNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn2_optimizer = optim.Adam(gnn2_model.parameters(), lr=0.0005)
        gnn2_scheduler = ReduceLROnPlateau(gnn2_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn2 == 'pna':
        gnn2_model = pnaNet(deg=deg, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn2_optimizer = optim.Adam(gnn2_model.parameters(), lr=0.001)
        gnn2_scheduler = ReduceLROnPlateau(gnn2_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn2 == 'rgcn':
        gnn2_model = rgcnNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn2_optimizer = optim.Adam(gnn2_model.parameters(), lr=0.001, weight_decay=1e-5)
        gnn2_scheduler = ReduceLROnPlateau(gnn2_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn2 == 'sage':
        gnn2_model = sageNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn2_optimizer = optim.Adam(gnn2_model.parameters(), lr=0.0005)
        gnn2_scheduler = ReduceLROnPlateau(gnn2_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn2 == 'sgn':
        gnn2_model = sgnNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn2_optimizer = optim.Adam(gnn2_model.parameters(), lr=0.0005)
        gnn2_scheduler = ReduceLROnPlateau(gnn2_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)
    elif args.gnn2 == 'unet':
        gnn2_model = unetNet(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        gnn2_optimizer = optim.Adam(gnn2_model.parameters(), lr=0.0005)
        gnn2_scheduler = ReduceLROnPlateau(gnn2_optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)

    gnn1_valid_curve = []
    gnn1_test_curve = []
    gnn1_train_curve = []

    gnn1_test_predict_value= []
    gnn1_test_true_value= []
    gnn1_valid_predict_value= []
    gnn1_valid_true_value= []

    gnn2_valid_curve = []
    gnn2_test_curve = []
    gnn2_train_curve = []

    gnn2_test_predict_value= []
    gnn2_test_true_value= []
    gnn2_valid_predict_value= []
    gnn2_valid_true_value= []

    valid_curve = []
    test_curve = []
    train_curve = []
    test_predict_value= []
    test_true_value= []
    valid_predict_value= []
    valid_true_value= []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training ' + args.gnn1 + '...')
        train(gnn1_model, device, train_loader, gnn1_optimizer, dataset.task_type)
        print('Training ' + args.gnn2 + '...')
        train(gnn2_model, device, train_loader, gnn2_optimizer, dataset.task_type)

        print('Evaluating ' + args.gnn1 + '...')
        gnn1_train_perf, gnn1_tr_true, gnn1_tr_pred = eval(gnn1_model, device, train_loader, evaluator)
        gnn1_valid_perf, gnn1_v_true,  gnn1_v_pred= eval(gnn1_model, device, valid_loader, evaluator)
        gnn1_test_perf, gnn1_t_true, gnn1_t_pred = eval(gnn1_model, device, test_loader, evaluator)

        print({'Train': gnn1_train_perf, 'Validation': gnn1_valid_perf, 'Test': gnn1_test_perf})

        print('Evaluating ' + args.gnn2 + '...')
        gnn2_train_perf, gnn2_tr_true, gnn2_tr_pred = eval(gnn2_model, device, train_loader, evaluator)
        gnn2_valid_perf, gnn2_v_true,  gnn2_v_pred= eval(gnn2_model, device, valid_loader, evaluator)
        gnn2_test_perf, gnn2_t_true, gnn2_t_pred = eval(gnn2_model, device, test_loader, evaluator)

        print({'Train': gnn2_train_perf, 'Validation': gnn2_valid_perf, 'Test': gnn2_test_perf})

        tr_true = np.zeros((len(gnn2_tr_true), 1))

        for i in range(len(gnn2_tr_true)):
            tr_true[i][0] = (gnn1_tr_true[i][0] + gnn2_tr_true[i][0]) / 2
        
        v_true = np.zeros((len(gnn2_v_true), 1))

        for i in range(len(gnn2_v_true)):
            v_true[i][0] = (gnn1_v_true[i][0] + gnn2_v_true[i][0]) / 2

        t_true = np.zeros((len(gnn2_t_true), 1))

        for i in range(len(gnn2_t_true)):
            t_true[i][0] = (gnn1_t_true[i][0] + gnn2_t_true[i][0]) / 2

        tr_pred = np.zeros((len(gnn2_tr_pred), 1))

        for i in range(len(gnn2_tr_pred)):
            tr_pred[i][0] = (gnn1_tr_pred[i][0] + gnn2_tr_pred[i][0]) / 2
        
        v_pred = np.zeros((len(gnn2_v_pred), 1))

        for i in range(len(gnn2_v_pred)):
            v_pred[i][0] = (gnn1_v_pred[i][0] + gnn2_v_pred[i][0]) / 2

        t_pred = np.zeros((len(gnn2_t_pred), 1))

        for i in range(len(gnn2_t_pred)):
            t_pred[i][0] = (gnn1_t_pred[i][0] + gnn2_t_pred[i][0]) / 2

        train_dict = {"y_true": tr_true, "y_pred": tr_pred}
        valid_dict = {"y_true": gnn1_v_true, "y_pred": v_pred}
        test_dict = {"y_true": gnn1_t_true, "y_pred": t_pred}
        train_perf = evaluator.eval(train_dict)
        valid_perf = evaluator.eval(valid_dict)
        test_perf = evaluator.eval(test_dict)
        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        gnn1_train_curve.append(gnn1_train_perf[dataset.eval_metric])
        gnn1_valid_curve.append(gnn1_valid_perf[dataset.eval_metric])
        gnn1_test_curve.append(gnn1_test_perf[dataset.eval_metric])
    
        gnn1_test_predict_value.append(reduce(operator.add, gnn1_t_pred.tolist()))
        gnn1_valid_predict_value.append(reduce(operator.add, gnn1_v_pred.tolist()))

        gnn2_train_curve.append(gnn2_train_perf[dataset.eval_metric])
        gnn2_valid_curve.append(gnn2_valid_perf[dataset.eval_metric])
        gnn2_test_curve.append(gnn2_test_perf[dataset.eval_metric])

        gnn2_test_predict_value.append(reduce(operator.add, gnn2_t_pred.tolist()))
        gnn2_valid_predict_value.append(reduce(operator.add, gnn2_v_pred.tolist()))

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        test_predict_value.append(reduce(operator.add, t_pred.tolist()))
        valid_predict_value.append(reduce(operator.add, v_pred.tolist()))

        gnn1_test_loss=gnn1_test_perf[dataset.eval_metric]
        if gnn1_test_loss<=np.min(np.array(gnn1_test_curve)):
            PATH='model/'+args.dataset + '_' + args.gnn1 + '_layer_'+ str(args.num_layer)+'_model.pt'
            torch.save({'epoch': epoch,
                        'model_state_dict': gnn1_model.state_dict(),
                        'optimizer_state_dict': gnn1_optimizer.state_dict(),
                        'loss': gnn1_test_loss
                        }, PATH)
        
        gnn2_test_loss=gnn2_test_perf[dataset.eval_metric]
        if gnn2_test_loss<=np.min(np.array(gnn2_test_curve)):
            PATH='model/'+args.dataset + '_' + args.gnn2 + '_layer_'+ str(args.num_layer)+'_model.pt'
            torch.save({'epoch': epoch,
                        'model_state_dict': gnn2_model.state_dict(),
                        'optimizer_state_dict': gnn2_optimizer.state_dict(),
                        'loss': gnn2_test_loss
                        }, PATH)
        
        test_loss=test_perf[dataset.eval_metric]
        if test_loss<=np.min(np.array(test_curve)):
            PATH='model/'+args.dataset + '_' + args.gnn1 + '_' + args.gnn2 + '_layer_avg_'+ str(args.num_layer)+'_model.pt'
            torch.save({'epoch': epoch,
                        'gnn1_model_state_dict': gnn1_model.state_dict(),
                        'gnn2_model_state_dict': gnn2_model.state_dict(),
                        'gnn1_optimizer_state_dict': gnn1_optimizer.state_dict(),
                        'gnn2_optimizer_state_dict': gnn2_optimizer.state_dict(),
                        'loss': test_loss
                        }, PATH)
    
    gnn1_test_true_value=reduce(operator.add, gnn1_t_true.tolist())
    gnn1_valid_true_value=reduce(operator.add, gnn1_v_true.tolist())

    gnn2_test_true_value=reduce(operator.add, gnn2_t_true.tolist())
    gnn2_valid_true_value=reduce(operator.add, gnn2_v_true.tolist())

    test_true_value=reduce(operator.add, t_true.tolist())
    valid_true_value=reduce(operator.add, v_true.tolist())

    if 'classification' in dataset.task_type:
        gnn1_best_val_epoch = np.argmax(np.array(gnn1_valid_curve))
        gnn1_best_train = max(gnn1_valid_curve)

        gnn2_best_val_epoch = np.argmax(np.array(gnn2_valid_curve))
        gnn2_best_train = max(gnn2_train_curve)

        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        gnn1_best_val_epoch = np.argmin(np.array(gnn1_valid_curve))
        gnn1_best_train = min(gnn1_valid_curve)

        gnn2_best_val_epoch = np.argmin(np.array(gnn2_valid_curve))
        gnn2_best_train = min(gnn2_train_curve)

        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print(args.gnn1 + ' Best validation score: {}'.format(gnn1_valid_curve[gnn1_best_val_epoch]))
    print(args.gnn1 + ' Test score: {}'.format(gnn1_test_curve[gnn1_best_val_epoch]))

    print('Finished training!')
    print(args.gnn2 + ' validation score: {}'.format(gnn2_valid_curve[gnn2_best_val_epoch]))
    print(args.gnn2 + ' score: {}'.format(gnn2_test_curve[gnn2_best_val_epoch]))

    print('Finished training!')
    print('AVG Ensemble Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('AVG Ensemble Test score: {}'.format(test_curve[best_val_epoch]))

    f = open('result/'+args.dataset + '_' + args.gnn1 + '_layer_'+str(args.num_layer)+'.json', 'w')
    result=dict(val=gnn1_valid_curve[gnn1_best_val_epoch], \
        test=gnn1_test_curve[gnn1_best_val_epoch],train=gnn1_train_curve[gnn1_best_val_epoch], \
        test_pred=gnn1_test_predict_value, value_pred=gnn1_valid_predict_value, 
        test_true=gnn1_test_true_value, valid_true=gnn1_valid_true_value,\
        train_curve=gnn1_train_curve, test_curve=gnn1_test_curve, valid_curve=gnn1_valid_curve)
    json.dump(result, f)
    f.close()

    f = open('result/'+args.dataset + '_' + args.gnn2 + '_layer_'+str(args.num_layer)+'.json', 'w')
    result=dict(val=gnn2_valid_curve[gnn2_best_val_epoch], \
        test=gnn2_test_curve[gnn2_best_val_epoch],train=gnn2_train_curve[gnn2_best_val_epoch], \
        test_pred=gnn2_test_predict_value, value_pred=gnn2_valid_predict_value, 
        test_true=gnn2_test_true_value, valid_true=gnn2_valid_true_value,\
        train_curve=gnn2_train_curve, test_curve=gnn2_test_curve, valid_curve=gnn2_valid_curve)
    json.dump(result, f)
    f.close()

    f = open('result/'+args.dataset + '_' + args.gnn1 + '_' + args.gnn2 + '_layer_avg_'+str(args.num_layer)+'.json', 'w')
    result=dict(val=valid_curve[best_val_epoch], \
        test=test_curve[best_val_epoch],train=train_curve[best_val_epoch], \
        test_pred=test_predict_value, value_pred=valid_predict_value, 
        test_true=test_true_value, valid_true=valid_true_value,\
        train_curve=train_curve, test_curve=test_curve, valid_curve=valid_curve)
    json.dump(result, f)
    f.close()

    if not args.filename == '':
        torch.save({'Val': gnn1_valid_curve[gnn1_best_val_epoch], 'Test': gnn1_test_curve[gnn1_best_val_epoch], 'Train': gnn1_train_curve[gnn1_best_val_epoch], 'BestTrain': gnn1_best_train}, args.filename)

    if not args.filename == '':
        torch.save({'Val': gnn2_valid_curve[gnn2_best_val_epoch], 'Test': gnn2_test_curve[gnn2_best_val_epoch], 'Train': gnn2_train_curve[gnn2_best_val_epoch], 'BestTrain': gnn2_best_train}, args.filename)

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)

if __name__ == "__main__":
    main()