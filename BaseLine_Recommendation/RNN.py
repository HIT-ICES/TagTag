import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from Data.InputData import train_dataWord2Vec, val_dataWord2Vec, test_dataWord2Vec
from torchmetrics import F1, Precision, Recall
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

def packSequence(x):
    tensorList = list()
    length = list()
    zero = torch.zeros(128)
    zero = zero.cuda()
    for tensor in x:
        for i in range(len(tensor)):
            if zero.equal(tensor[i]):
                break
        tensorList.append(tensor[:i, :])
        length.append(i)
    return pack_sequence(tensorList, enforce_sorted=False), length

class RNN(pl.LightningModule):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(128, 90, 2, batch_first=True, nonlinearity='relu', bidirectional=False)
        self.fc = nn.Linear(90, 50)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x, length = packSequence(x)
        out, h_n = self.rnn(x)
        encoder_outputs, _ = pad_packed_sequence(out, batch_first=True)
        out = list()
        for i in range(len(length)):
            o = encoder_outputs[i][:length[i]]
            o = o[-1]
            out.append(o)
        out = torch.stack(out)
        out = self.fc(out)
        output = self.sigmoid(out)
        return output

    def training_step(self, train_batch, batch_idx):
        sequences, label, _, _ = train_batch
        output = self(sequences)
        label = torch.tensor(label, dtype=torch.float32)
        loss = F.binary_cross_entropy(output, label)
        self.log('train loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        sequences, label, _, _ = val_batch
        # print(sequences)
        output = self(sequences)
        if output is None:
            print('output none,', batch_idx)
        if label is None:
            print('label none,', batch_idx)
        label = label.to(torch.float32)
        loss = F.binary_cross_entropy(output, label)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        sequences, labels, _, _ = test_batch
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
        print(
            {"F1": F1 / i, "Precision": Precision / i, "Recall": Recall / i})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

NUM_CLASS = 50

train_dataset = train_dataWord2Vec()
val_dataset = val_dataWord2Vec()
test_dataset = test_dataWord2Vec()

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=8)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=4)

model = RNN()
trainer = pl.Trainer(max_epochs=100, gpus='1', callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(model, train_dl, val_dl)
trainer.test(model, test_dl)
