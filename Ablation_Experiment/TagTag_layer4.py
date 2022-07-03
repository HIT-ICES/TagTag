import pickle
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from Data.InputData import test_dataWord2Vec128, val_dataWord2Vec128, train_dataWord2Vec128
from torchmetrics import F1, Precision, Recall
from torch_geometric.nn import GCNConv, global_mean_pool
from pytorch_lightning import seed_everything


def select_negative_samples(label, negative_sample_ratio: int = 4):
    num_candidate = label.size(0)
    positive_idx = label.nonzero(as_tuple=True)[0]
    positive_idx = positive_idx.cpu().numpy()
    size = negative_sample_ratio * len(positive_idx)
    if size>len(label)-len(positive_idx):
        size = len(label)-len(positive_idx)
    negative_idx = np.random.choice(np.delete(np.arange(num_candidate), positive_idx),
                                    size=size, replace=False)
    sample_idx = np.concatenate((positive_idx, negative_idx), axis=None)
    label_new = torch.tensor([1] * len(positive_idx) + [0] * len(negative_idx), dtype=torch.float32)
    return positive_idx, negative_idx, sample_idx, label_new.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

class FC(pl.LightningModule):

    def __init__(self, graph):
        super(FC, self).__init__()
        self.conv1 = GCNConv(128, 108)
        self.layer_norm1 = nn.LayerNorm(108)
        self.conv2 = GCNConv(108, 88)
        self.layer_norm2 = nn.LayerNorm(88)
        self.conv3 = GCNConv(88, 68)
        self.layer_norm3 = nn.LayerNorm(68)
        self.conv4 = GCNConv(68, 50)
        self.batch_norm = nn.BatchNorm1d(50)
        self.graph = graph.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.register_buffer('labels', torch.zeros(50))
        self.register_buffer('global_mean_pool_batch', torch.tensor([0]))
        self.relu = nn.ReLU()
        self.linear_new_label = nn.Sequential(
            nn.Linear(228, 114),
            nn.ReLU(),
            nn.Linear(114, 1),
            nn.Sigmoid(),
        )
        self.linear_label = nn.Sequential(
            nn.Linear(178, 89),
            nn.ReLU(),
            nn.Linear(89, 1),
            nn.Sigmoid(),
        )
        params = torch.full([2], 0.5, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.f1 = F1(threshold=0.5)
        self.pre = Precision(average='micro')
        self.recall = Recall(average='micro')


    def forward(self, x, stage, negative_sample):
        data = self.graph
        nodes = self.conv1(data.x, data.edge_index, data.edge_attr)
        nodes = self.layer_norm1(nodes.to(torch.float32))
        nodes = self.relu(nodes.to(torch.float))
        nodes = self.conv2(nodes, data.edge_index, data.edge_attr)
        nodes = self.layer_norm2(nodes.to(torch.float32))
        nodes = self.relu(nodes.to(torch.float))
        nodes = self.conv3(nodes, data.edge_index, data.edge_attr)
        nodes = self.layer_norm3(nodes.to(torch.float32))
        nodes = self.relu(nodes.to(torch.float))
        nodes = self.conv4(nodes, data.edge_index, data.edge_attr)
        nodes = self.batch_norm(nodes.to(torch.float32))

        y1 = torch.zeros(10)
        n = np.arange(len(nodes))
        if stage == "train":
            n = negative_sample
            x = x.repeat(1, 1)

        for i in n:
            node = nodes[i].repeat(len(x), 1)
            text_node = torch.cat([x, node], dim=1).to(torch.float)
            answer = self.linear_label(text_node)
            if i == n[0]:
                y1 = answer
            else:
                y1 = torch.cat((y1, answer), dim=1)

        graph = global_mean_pool(nodes, batch=self.global_mean_pool_batch)
        graph = graph.repeat(len(x), 1)
        labels = self.labels
        labels = labels.repeat(len(x), 1)
        if stage == "train":
            y1 = y1.squeeze(0)
            for i in range(y1.size(0)):
                if y1[i] > 0.5:
                    labels[0][n[i]] = 1
        else:
            for i in range(len(y1)):
                for j in range(len(y1[0])):
                    if y1[i][j] > 0.5:
                        labels[i][j] = 1

        text_graph = torch.cat([x, graph, labels], dim=1).to(torch.float)
        y2 = self.linear_new_label(text_graph)

        return y1, y2

    def training_step(self, train_batch, batch_idx):
        sequences, label, newLabel, _ = train_batch

        loss1, loss2 = 0, 0
        num_sample = 0

        positive_idx_new_label, negative_idx_new_label, sample_idx_new_label, label_new_label = select_negative_samples(newLabel, 1)
        sequences_sample = sequences[sample_idx_new_label, :]
        new_label_num = 0

        for api, api_labels, api_new_label in zip(sequences, label, newLabel):
            positive_idx, negative_idx, sample_idx, label_new = select_negative_samples(api_labels)
            output1, output2 = self(api, "train", sample_idx)
            num_sample += len(sample_idx)
            if len(label_new) == 0:
                loss1 += 0
            else:
                loss1 += F.binary_cross_entropy(output1.squeeze(0), label_new.to(torch.float32), size_average=False, reduction='sum')
            if api in sequences_sample:
                loss2 += F.binary_cross_entropy(output2.squeeze(0), api_new_label, size_average=False, reduction='sum')
                new_label_num += 1
        if new_label_num == 0:
            loss2 = 0
            new_label_num = 1
        loss = 0.5/(self.params[0] ** 2)*(loss1/num_sample) + torch.log(self.params[0]) + 0.5/(self.params[1]**2)*(loss2/new_label_num) + torch.log(self.params[1])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        sequences, label, newLabel, _ = val_batch
        output1, output2 = self(sequences, "val", 0)

        label = label.to(torch.float32)
        loss1 = F.binary_cross_entropy(output1, label).mean()
        loss2 = F.binary_cross_entropy(output2, newLabel).mean()
        loss = 0.5/(self.params[0] ** 2)*loss1 + torch.log(self.params[0]) + 0.5/(self.params[1]**2)*loss2 + torch.log(self.params[1])
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx):
        sequences, labels, newLabel, _ = test_batch
        output1, output2 = self(sequences, "test", 0)

        target1 = labels.to(torch.int)
        target2 = newLabel.to(torch.int)

        f1_R = self.f1(output1, target1)
        p_R = self.pre(output1, target1)
        r_R = self.recall(output1, target1)

        f1_P = self.f1(output2, target2)
        p_P = self.pre(output2, target2)
        r_P = self.recall(output2, target2)

        return {"F1_R": f1_R, "Precisio_R": p_R, "Recall_R": r_R,
                "F1_P": f1_P, "Precision_P": p_P, "Recall_P": r_P}

    def test_epoch_end(self, outputs):
        i, F1_R, Precision_R, Recall_R = 0, 0, 0, 0
        F1_P, Precision_P, Recall_P = 0, 0, 0
        for out in outputs:
            i += 1
            F1_R += out["F1_R"]
            Precision_R += out["Precision_R"]
            Recall_R += out["Recall_R"]
            F1_P += out["F1_P"]
            Precision_P += out["Precision_P"]
            Recall_P += out["Recall_P"]
        print()
        print({"F1_R": F1_R / i, "Precision_R": Precision_R / i, "Recall_R": Recall_R / i})
        print({"F1_P": F1_P / i, "Precision_P": Precision_P / i, "Recall_P": Recall_P / i})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

seed_everything(5)

train_dataset = train_dataWord2Vec128()
val_dataset = val_dataWord2Vec128()
test_dataset = test_dataWord2Vec128()

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=8)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=4)

fn = '..//myData//final_data//graph.pkl'
with open(fn, 'rb+') as f:
    graph = pickle.load(f)
model = FC(graph)

trainer = pl.Trainer(max_epochs=100, gpus='1')
trainer.fit(model, train_dl, val_dl)
trainer.test(model, test_dl)
