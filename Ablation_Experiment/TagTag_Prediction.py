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

def select_negative_samples(label, negative_sample_ratio: int = 3):
    num_candidate = label.size(0)
    positive_idx = label.nonzero(as_tuple=True)[0]
    positive_idx = positive_idx.cpu().numpy()
    size = negative_sample_ratio * len(positive_idx)
    if size == 0:
        size = 1
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
        self.register_buffer('global_mean_pool_batch', torch.tensor([0]))
        self.relu = nn.ReLU()
        self.linear_new_label = nn.Sequential(
            nn.Linear(178, 89),
            nn.ReLU(),
            nn.Linear(89, 1),
            nn.Sigmoid(),
        )
        self.f1 = F1(threshold=0.5)
        self.pre = Precision(average='micro')
        self.recall = Recall(average='micro')

    def forward(self, x, stage):
        data = self.graph  # 获取图数据，共50个节点，节点的特征维度是128维
        nodes = self.conv1(data.x, data.edge_index, data.edge_attr)  # 图卷积进行图编码
        nodes = self.layer_norm1(nodes.to(torch.float32))
        nodes = self.relu(nodes.to(torch.float))
        nodes = self.conv2(nodes, data.edge_index, data.edge_attr)  # 图卷积进行图编码
        nodes = self.layer_norm2(nodes.to(torch.float32))
        nodes = self.relu(nodes.to(torch.float))
        nodes = self.conv3(nodes, data.edge_index, data.edge_attr)  # 图卷积进行图编码
        nodes = self.batch_norm(nodes.to(torch.float32))

        n = np.arange(len(nodes))
        if stage == "train":
            x = x.repeat(1, 1)

        graph = global_mean_pool(nodes, batch=self.global_mean_pool_batch)
        graph = graph.repeat(len(x), 1)

        text_graph = torch.cat([x, graph], dim=1).to(torch.float)  # 将图和文本编码拼接\
        y2 = self.linear_new_label(text_graph)

        return y2

    def training_step(self, train_batch, batch_idx):
        sequences, _, newLabel, _ = train_batch
        loss = 0
        positive_idx_new_label, negative_idx_new_label, sample_idx_new_label, label_new_label = select_negative_samples(newLabel, 1)
        sequences_sample = sequences[sample_idx_new_label, :]
        new_label_num = 0

        for api, api_new_label in zip(sequences, newLabel):
            if api in sequences_sample:
                output2 = self(api, "train")
                loss += F.binary_cross_entropy(output2.squeeze(0), api_new_label, size_average=False, reduction='sum')
                new_label_num += 1
        if new_label_num == 0:
            new_label_num = 1
        loss = loss/new_label_num
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        sequences, _, newLabel, _ = val_batch
        output2 = self(sequences, "val")

        loss = F.binary_cross_entropy(output2, newLabel).mean()
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx):
        sequences, _, newLabel, _ = test_batch
        output2 = self(sequences, "test")

        target2 = newLabel.to(torch.int)

        f1 = self.f1(output2, target2)
        p = self.pre(output2, target2)
        r = self.recall(output2, target2)

        return {"F1": f1, "Precision_New": p, "Recall": r}

    def test_epoch_end(self, outputs):
        i= 0
        F1, Precision, Recall, = 0, 0, 0
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
