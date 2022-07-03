import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from Data.InputData import train_dataWord2Vec, val_dataWord2Vec, test_dataWord2Vec
from torchmetrics import F1, Precision, Recall
from torch.nn.utils.rnn import pack_sequence

def packSequence(x):
    tensorList = list()
    zero = torch.zeros(128)
    zero = zero.cuda()
    for tensor in x:
        for i in range(len(tensor)):
            if zero.equal(tensor[i]):
                break
        tensorList.append(tensor[:i, :])
    return pack_sequence(tensorList, enforce_sorted=False)

class LSTM(pl.LightningModule):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, bidirectional=False, batch_first=True)
        self.decoder = nn.Linear(in_features=64, out_features=1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = packSequence(x)
        lstm_output, (hn, cn) = self.lstm(x)
        hn = hn.squeeze()
        y = self.decoder(hn)
        y = self.sigmoid(y)
        return y

    def training_step(self, train_batch, batch_idx):
        sequences, _, label, _ = train_batch
        output = self(sequences)
        loss = F.binary_cross_entropy(output, label)
        loss = loss.mean()
        self.log('train loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        sequences, _, label, _ = val_batch
        output = self(sequences)
        if output is None:
            print('output none,', batch_idx)
        if label is None:
            print('label none,', batch_idx)
        loss = F.binary_cross_entropy(output, label)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        sequences, _, labels, _ = test_batch
        output = self(sequences)
        output = output.to('cpu').numpy()
        target = labels.to('cpu').numpy()
        f1 = F1(threshold=0.5)
        precision = Precision(average='micro')
        recall = Recall(average='micro')
        output = torch.from_numpy(output)
        target = torch.from_numpy(target).type(torch.int)
        f1One = f1(output, target)
        p = precision(output, target)
        r = recall(output, target)
        return {"F1": f1One, "Precision": p, "Recall": r}

    def test_epoch_end(self, outputs):
        i, F1, Precision, Recall = 0, 0, 0, 0
        for out in outputs:
            i += 1
            F1 += out["F1"]
            Precision += out["Precision"]
            Recall += out["Recall"]
        print()
        print({"F1": F1 / i, "Precision": Precision / i, "Recall": Recall / i})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

NUM_CLASS = 50

train_dataset = train_dataWord2Vec()
val_dataset = val_dataWord2Vec()
test_dataset = test_dataWord2Vec()

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=8)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=4)

model = LSTM()
trainer = pl.Trainer(max_epochs=100, gpus='1', callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(model, train_dl, val_dl)
trainer.test(model, test_dl)
