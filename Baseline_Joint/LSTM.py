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
    def __init__(self, wv_dim, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=wv_dim, hidden_size=hidden_size, bidirectional=False, batch_first=True)
        self.decoder = nn.Linear(in_features=hidden_size, out_features=51)
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
        sequences, _, _, label = train_batch
        output = self(sequences)
        loss = F.binary_cross_entropy(output, label)
        loss = loss.mean()
        self.log('train loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        sequences, _, _, label = val_batch
        output = self(sequences)
        if output is None:
            print('output none,', batch_idx)
        if label is None:
            print('label none,', batch_idx)
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

model = LSTM(128, 90, NUM_CLASS)
trainer = pl.Trainer(max_epochs=100, gpus='1', callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(model, train_dl, val_dl)
trainer.test(model, test_dl)
