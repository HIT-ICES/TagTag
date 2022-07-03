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
        self.fc = nn.Linear(90, 51)
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
        sequences, _, _, label = train_batch
        output = self(sequences)
        label = torch.tensor(label, dtype=torch.float32)
        loss = F.binary_cross_entropy(output, label)
        self.log('train loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        sequences, _, _, label = val_batch
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
        sequences, target1, target2, _ = test_batch
        output = self(sequences)
        output = output.to('cpu').numpy()
        output1 = torch.from_numpy(output[:, 0:50])
        output2 = torch.from_numpy(output[:, -1:])
        f1 = F1(threshold=0.5).to('cpu')
        precision = Precision(average='micro')
        recall = Recall(average='micro')
        target1 = torch.tensor(target1, dtype=torch.int).to('cpu')
        target2 = torch.tensor(target2, dtype=torch.int).to('cpu')
        f1_R = f1(output1, target1)
        p_R = precision(output1, target1)
        r_R = recall(output1, target1)

        f1_P = f1(output2, target2)
        p_P = precision(output2, target2)
        r_P = recall(output2, target2)

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
