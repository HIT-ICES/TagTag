import pickle
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from Data.InputData import test_dataWord2Vec128, val_dataWord2Vec128, train_dataWord2Vec128
from torchmetrics import F1, Precision, Recall
from torch_geometric.nn import GCNConv

def select_negative_samples(label, negative_sample_ratio: int = 3):
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
        self.conv1 = GCNConv(128, 100)
        self.layer_norm1 = nn.LayerNorm(100)
        self.conv2 = GCNConv(100, 75)
        self.layer_norm2 = nn.LayerNorm(75)
        self.conv3 = GCNConv(75, 50)
        self.batch_norm = nn.BatchNorm1d(50)
        self.graph = graph.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.register_buffer('labels', torch.zeros(50))
        self.relu = nn.ReLU()
        self.linear_label = nn.Sequential(
            nn.Linear(178, 89),
            nn.ReLU(),
            nn.Linear(89, 1),
            nn.Sigmoid(),
        )
        self.f1 = F1(threshold=0.5)
        self.pre = Precision(average='micro')
        self.recall = Recall(average='micro')

    def forward(self, x, stage, negative_sample):
        data = self.graph  # 获取图数据，共50个节点，节点的特征维度是128维
        nodes = self.conv1(data.x, data.edge_index, data.edge_attr)  # 图卷积进行图编码
        nodes = self.layer_norm1(nodes.to(torch.float32))
        nodes = self.relu(nodes.to(torch.float))
        nodes = self.conv2(nodes, data.edge_index, data.edge_attr)  # 图卷积进行图编码
        nodes = self.layer_norm2(nodes.to(torch.float32))
        nodes = self.relu(nodes.to(torch.float))
        nodes = self.conv3(nodes, data.edge_index, data.edge_attr)  # 图卷积进行图编码
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

        return y1

    def training_step(self, train_batch, batch_idx):
        sequences, label, _, _ = train_batch

        loss = 0
        num_sample = 0

        for api, api_labels in zip(sequences, label):
            positive_idx, negative_idx, sample_idx, label_new = select_negative_samples(api_labels)
            output1 = self(api, "train", sample_idx)
            num_sample += len(sample_idx)
            if len(label_new) != 0:
                loss += F.binary_cross_entropy(output1.squeeze(0), label_new.to(torch.float32), size_average=False, reduction='sum')

        loss = loss/num_sample
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        sequences, label, _, _ = val_batch
        output1 = self(sequences, "val", 0)

        label = label.to(torch.float32)
        loss1 = F.binary_cross_entropy(output1, label).mean()
        self.log('val_loss', loss1)

    def test_step(self, test_batch, batch_idx):
        sequences, labels, _, _ = test_batch
        output1 = self(sequences, "test", 0)

        target1 = labels.to(torch.int)

        f1 = self.f1(output1, target1)
        p = self.pre(output1, target1)
        r = self.recall(output1, target1)

        return {"F1": f1, "Precision": p, "Recall": r}

    def test_epoch_end(self, outputs):
        i, F1,  Precision, Recall= 0, 0, 0, 0
        for out in outputs:
            i += 1
            F1 += out["F1"]
            Precision += out["Precision"]
            Recall += out["Recall"]
        print()
        print({"F1": F1/i, "Precision": Precision/i, "Recall": Recall/i})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

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
