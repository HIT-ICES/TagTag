import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Data.InputData import train_dataWord2Vec, test_dataWord2Vec, val_dataWord2Vec
from torchmetrics import F1, Precision, Recall

class ServeNet(pl.LightningModule):
    def __init__(self, classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.desc_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
        )
        self.desc_cnn2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

        self.desc_bilstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True,
                                   bidirectional=True)
        self.final = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, desc_output): # [batch_size, 150, 128]
        desc_output = desc_output.unsqueeze(1) # [batch_size, 1, 150, 128]
        desc_output = F.pad(desc_output, (1, 1, 1, 1)) # [batch_size, 1, 152, 130]
        desc_output = torch.tensor(desc_output, dtype=torch.float32)
        desc_output = self.desc_cnn(desc_output) # [batch_size, 32, 150, 128]
        desc_output = F.pad(desc_output, (0, 0, 0, 0))
        desc_output = self.desc_cnn2(desc_output) # [batch_size, 1, 150, 128]
        desc_output = torch.reshape(desc_output, (-1, 150, 128)) # [batch_size, 150, 128]
        desc_output, _ = self.desc_bilstm(desc_output) # [batch_size, 150, 1024]
        desc_output1 = desc_output[:, 0, :] # [batch_size, 1024]
        desc_output2 = desc_output[:, -1, :]
        desc_output = torch.cat((desc_output1, desc_output2), dim=-1)
        output = self.final(desc_output) # [batch_size, 50]
        output = self.sigmoid(output)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, eps=1e-8, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        input, _, target, _ = train_batch
        output = self.forward(input)
        loss = F.binary_cross_entropy(output, target, reduction='none')
        loss = loss.mean()
        self.log('train loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        input, _, target, _ = val_batch
        output = self.forward(input)
        loss = F.binary_cross_entropy(output, target)
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx):
        input, _, target, _ = test_batch
        output = self(input)
        output = output.to('cpu').numpy()
        target = target.to('cpu').numpy()
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

CLASSNUM = 50

train_dataset = train_dataWord2Vec()
val_dataset = val_dataWord2Vec()
test_dataset = test_dataWord2Vec()

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=8)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=4)

serv_model = ServeNet(CLASSNUM)

trainer = pl.Trainer(max_epochs=100, gpus='1', callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(serv_model, train_dl, val_dl)
trainer.test(serv_model, test_dl)
