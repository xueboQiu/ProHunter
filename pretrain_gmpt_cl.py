import gc
import os
import time
import util
import psutil
from numpy.random import default_rng
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from splitters import random_split
from util import GraphAug
from dataloader import DataLoaderContrastive
from graph_matching import AdaGMNConv

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z_i.device)

        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
class GMPT_CL(nn.Module):

    def __init__(self, edge_dim, args, gnn):
        super(GMPT_CL, self).__init__()
        self.gnn = gnn
        self.cgmn = AdaGMNConv(edge_dim, args.emb_dim, mode=args.mode)

        self.mode = args.mode
        self.temperature = args.temperature
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([2*(args.batch_size-1)/1]))
        # self.criterion = InfoNCELoss(temperature=0.5)
        labels = torch.cat([torch.arange(args.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        self.mask = mask.to(args.device)
        self.labels = labels.to(args.device)
        self.zeros = torch.zeros(1, dtype=torch.long).to(args.device)
        self.ablation = args.ablation_test

    def from_pretrained(self, model_file, dataset, input_epoch):
        model_file = os.path.join(model_file, dataset) + "/"
        print(f'loading pre-trained model from {model_file}, input_epoch = {input_epoch}.', flush=True)
        self.gnn.load_state_dict(torch.load(model_file + f'{input_epoch}.pth', map_location=lambda storage, loc: storage))
        self.cgmn.load_state_dict(torch.load(model_file + f'{input_epoch}.cgmn.pth', map_location=lambda storage, loc: storage))

    def forward_cl(self, gid, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)  # (node_num, emb_dim)
        similarity_matrix = self.cgmn(gid, x, edge_index, edge_attr, batch)
        # remove gid idx
        similarity_matrix = torch.cat([similarity_matrix[:gid], similarity_matrix[gid+1:]], dim=0)

        positives = similarity_matrix[self.labels[gid].bool()]
        negatives = similarity_matrix[~self.labels[gid].bool()]
        logits = torch.cat([positives, negatives])

        logits /= self.temperature  # logits[0] is positive.
        # Create correct labels, positive sample is 1, negative sample is 0
        labels = torch.zeros_like(logits)
        labels[0] = 1  # Assume the first logit is a positive sample

        loss = self.criterion(logits.unsqueeze(0), labels.unsqueeze(0))
        return loss

    def forward_cl_2(self, gid, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)  # (node_num, emb_dim)
        # Use global mean pooling to obtain graph-level representation
        out = F.normalize(global_mean_pool(x, batch), dim=-1)  # (batch_size, emb_dim)
        # Get the graph embedding vector corresponding to gid
        gid_embedding = out[gid]  # (emb_dim)
        # Calculate the similarity between gid_embedding and other graph embedding vectors
        similarity_matrix = torch.matmul(out, gid_embedding)  # (batch_size)
        similarity_matrix = similarity_matrix[~self.mask[gid]]

        positives = similarity_matrix[self.labels[gid].bool()]
        negatives = similarity_matrix[~self.labels[gid].bool()]
        logits = torch.cat([positives, negatives])

        logits /= self.temperature  # logits[0] is positive.
        # Create correct labels, positive sample is 1, negative sample is 0
        labels = torch.zeros_like(logits)
        labels[0] = 1  # Assume the first logit is a positive sample

        loss = self.criterion(logits.unsqueeze(0), labels.unsqueeze(0))
        return loss

    def test_compare(self, gid, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)  # (node_num, emb_dim)
        if self.ablation:
            # Use global mean pooling to obtain graph-level representation
            out = F.normalize(global_mean_pool(x, batch), dim=-1)  # (batch_size, emb_dim)
            # Get the graph embedding vector corresponding to gid
            gid_embedding = out[gid]  # (emb_dim)
            # Calculate the similarity between gid_embedding and other graph embedding vectors
            similarity_matrix = torch.matmul(out, gid_embedding)  # (batch_size)
        else:
            similarity_matrix = self.cgmn(gid, x, edge_index, edge_attr, batch)
        return similarity_matrix[:gid]

    def compare(self, gid, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)  # (node_num, emb_dim)

        if self.ablation:
            # Use global mean pooling to obtain graph-level representation
            out = F.normalize(global_mean_pool(x, batch), dim=-1)  # (batch_size, emb_dim)
            # Get the graph embedding vector corresponding to gid
            gid_embedding = out[gid]  # (emb_dim)
            # Calculate the similarity between gid_embedding and other graph embedding vectors
            similarity_matrix = torch.matmul(out, gid_embedding)  # (batch_size)
        else:
            similarity_matrix = self.cgmn(gid, x, edge_index, edge_attr, batch)

        y_true = [0] * len(similarity_matrix)
        y_true[gid] = y_true[gid+int(len(y_true)/2)] = 1  # Mark positive samples as 1

        if gid == 0:
            y_true = y_true[1:]
            similarity_matrix = similarity_matrix[1:]
        else:
            y_true = y_true[:gid] + y_true[gid+1:]
            similarity_matrix = torch.concat((similarity_matrix[:gid], similarity_matrix[gid+1:]), dim=-1)
        return y_true, similarity_matrix

def train(args, model, loader, dataset, optimizer, device):

    model.train()

    loss_values = []
    num_graph = args.batch_size * 2
    # Before the training loop
    rng = default_rng()
    all_g_ids = rng.permutation(num_graph)
    slices = [all_g_ids[i:i + args.sample_num] for i in range(0, len(all_g_ids), args.sample_num)]

    num_batches = 0  # Record the number of batches
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        batch = batch.to(device)
        g_ids = slices[step % len(slices)]

        total_loss = 0
        for g_i in g_ids:
            # if is ablation test
            if args.ablation_test:
                loss = model.forward_cl_2(g_i, batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                loss = model.forward_cl(g_i, batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            total_loss += loss
            loss_values.append(loss.detach())
        total_loss = total_loss / args.sample_num
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        num_batches += 1  # Increase batch count

    train_loss_accum = sum(loss.item() for loss in loss_values) / len(loss_values)
    return train_loss_accum/(step+1)
def eval_test(args, model, device, test_dataset):

    model.eval()
    tp = fp = tn = fn = 0
    fn_less_thre = 0

    query_samples, sg_samples = test_dataset
    batch_size = len(query_samples) + 1

    neg_scores = []
    pos_scores = []

    y_pred = []
    y_label = []

    for i, sg_tuple in enumerate(tqdm(sg_samples)):
        sg, label = sg_tuple
        # Use DataLoader to batch the graph data
        from torch_geometric.data import DataLoader
        loader = DataLoader(query_samples+[sg], batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                similarities = model.test_compare(len(query_samples), batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                max_idx = torch.argmax(similarities).item()
                predicted_score = similarities[max_idx].item()
                true_score = similarities[label].item()
                y_pred.append(predicted_score)
                y_label.append(1 if label != -1 else 0)

                # Set base threshold to determine if the graph is similar
                if label == -1:
                    neg_scores.append(predicted_score)
                    if predicted_score >= args.thre:
                        fp += 1
                    else:
                        tn += 1
                else:
                    pos_scores.append(predicted_score)
                    if predicted_score < args.thre:
                        fn += 1
                        fn_less_thre += 1
                    else:
                        tp += 1

    # Calculate AUC
    auc_roc = roc_auc_score(y_label, y_pred)
    # Calculate Precision-Recall AUC
    precision, recall, thresholds = precision_recall_curve(y_label, y_pred)
    pr_auc = auc(recall, precision)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp/(fp+tn)
    acc = (tp+tn)/(tp + fp + tn + fn)

    print(f"recall:{recall:.2f}({tp}/{tp + fn}), fnlt:{fn_less_thre} ,fpr:{fpr:.2f}({fp}/{fp + tn}), acc:{acc:.2f}({tp + tn}/{tp + fp + tn + fn}), auc:{auc_roc:.2f},"
          f"pr_auc:{pr_auc:.2f}")
    # Output rounded to two decimal places
    neg_avg_score = sum(neg_scores)/len(neg_scores) if len(neg_scores) > 0 else 0
    neg_standard_deviation = np.std(neg_scores) if len(neg_scores) > 0 else 0
    neg_min_score = min(neg_scores) if len(neg_scores) > 0 else 0
    neg_max_score = max(neg_scores) if len(neg_scores) > 0 else 0
    print(f"avg neg score:{neg_avg_score:.2f}, std:{neg_standard_deviation:.2f}, min:{neg_min_score:.2f}, max:{neg_max_score:.2f}")

    pos_avg_score = sum(pos_scores)/len(pos_scores) if len(pos_scores) > 0 else 0
    pos_standard_deviation = np.std(pos_scores) if len(pos_scores) > 0 else 0
    pos_min_score = min(pos_scores) if len(pos_scores) > 0 else 0
    pos_max_score = max(pos_scores) if len(pos_scores) > 0 else 0
    print(f"avg pos score:{pos_avg_score:.2f}, std:{pos_standard_deviation:.2f}, min:{pos_min_score:.2f}, max:{pos_max_score:.2f}")

    return recall,fpr
def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_pred = []
    y_score = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            for gid in range(args.eval_sample_num):
                batch_y_true, batch_y_score = model.compare(gid, batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                y_true.extend(batch_y_true)
                y_score.extend(batch_y_score)
                y_pred.extend(batch_y_score)
                # y_pred.extend(batch_y_score>args.thre)

    def eval_metrics(y_true, y_pred, y_score):
        tp = sum(1 for i in range(len(y_true)) if y_pred[i] == 1 and y_true[i] == 1)
        tn = sum(1 for i in range(len(y_true)) if y_pred[i] == 0 and y_true[i] == 0)
        fp = sum(1 for i in range(len(y_true)) if y_pred[i] == 1 and y_true[i] == 0)
        fn = sum(1 for i in range(len(y_true)) if y_pred[i] == 0 and y_true[i] == 1)

        acc = (tp + tn) / len(y_true)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # Calculate AUC
        y_score = [u.cpu() for u in y_score]
        auc_score = roc_auc_score(y_true, y_score)

        return acc, recall, fpr, precision, f1, auc_score

    y_pred = np.array([u.cpu() for u in y_pred], dtype=float)
    acc, recall, fpr, precision, f1, auc_score = eval_metrics(y_true, y_pred > args.thre, y_score)
    print("validation result: thre:{:.1f}, acc:{:.4f}, recall:{:.4f}, fpr:{:.4f}, precision:{:.4f}, f1:{:.4f}, auc:{:.4f}".format(args.thre, acc, recall, fpr, precision, f1, auc_score))
    return acc

def main():
    args = util.parse_args()
    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print("device:",device)
    args.device = device

    # set up dataset
    from darpa_loader import DarpaDataset
    from darpa_model import GNN

    root = os.path.join(args.dataset_path, f'dataset/darpa_{args.mode}/train')
    # dataset = DarpaDataset(root, data_type='unsupervised', pre_transform=GraphAug(aug_ratio=args.aug_ratio))
    dataset = DarpaDataset(root, data_type='unsupervised', transform=GraphAug(aug_ratio=args.aug_ratio))
    test_dataset = torch.load( os.path.join(args.dataset_path, f'dataset/darpa_{args.mode}/test/processed/test_prohunter.pt'))

    train_dataset, valid_dataset, _ = random_split(dataset, seed=args.seed)

    # set up model
    node_dim = max(dataset.data.x).to(torch.long) + 1
    edge_dim = dataset.data.edge_attr.shape[1]
    gnn = GNN(node_dim, edge_dim, args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
              gnn_type=args.gnn_type)

    loader = DataLoaderContrastive(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoaderContrastive(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, drop_last=True)
    # test_loader = DataLoaderContrastive(test_dataset, batch_size=2, shuffle=False,num_workers=args.num_workers, drop_last=True)
    model = GMPT_CL(edge_dim,args, gnn)
    if args.eval:
        model.from_pretrained(args.model_file,args.mode, "best")
        model = model.to(device)
        eval_test(args, model, device, test_dataset)
        return

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    ### for random splitting
    best_val_acc = 0
    for epoch in range(args.input_epochs + 1, args.input_epochs + args.epochs + 1):
        print("====epoch " + str(epoch), flush=True)
        train_loss = train(args, model, loader, dataset, optimizer, device)
        print("train_loss:",train_loss)
        # Validation set evaluation
        acc = eval(args, model, device, val_loader)
        if best_val_acc <= acc:
            best_val_acc = acc
            torch.save(model.gnn.state_dict(), f'models/{args.mode}/{epoch}.pth')
            torch.save(model.cgmn.state_dict(), f'models/{args.mode}/{epoch}.cgmn.pth')

if __name__ == "__main__":
    main()