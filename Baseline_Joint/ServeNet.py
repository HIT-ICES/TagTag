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
            nn.Linear(in_features=1024, out_features=51)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, desc_output):
        desc_output = desc_output.unsqueeze(1)
        desc_output = F.pad(desc_output, (1, 1, 1, 1))
        desc_output = torch.tensor(desc_output, dtype=torch.float32)
        desc_output = self.desc_cnn(desc_output)
        desc_output = F.pad(desc_output, (0, 0, 0, 0))
        desc_output = self.desc_cnn2(desc_output)
        desc_output = torch.reshape(desc_output, (-1, 150, 128))
        desc_output, _ = self.desc_bilstm(desc_output)
        desc_output1 = desc_output[:, 0, :]
        desc_output2 = desc_output[:, -1, :]
        desc_output = torch.cat((desc_output1, desc_output2), dim=-1)
        output = self.final(desc_output)
        output = self.sigmoid(output)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, eps=1e-8, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        input, _, _, target = train_batch
        output = self.forward(input)
        loss = F.binary_cross_entropy(output, target, reduction='none')
        loss = loss.mean()
        self.log('train loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        input, _, _, target = val_batch
        output = self.forward(input)
        loss = F.binary_cross_entropy(output, target)
        self.log('val_loss', loss)

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
