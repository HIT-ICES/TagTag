import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from Data.InputData import train_dataWord2Vec, val_dataWord2Vec, test_dataWord2Vec
from torchmetrics import F1, Precision, Recall
from torch.utils.data import DataLoader

class TextCNN(pl.LightningModule):
    def __init__(self, embed_size, feature_size, window_size, max_seq_len):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=embed_size,
                                    out_channels=feature_size,
                                    kernel_size=kernel),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=max_seq_len - kernel + 1))
            for kernel in window_size
        ])
        self.fc = nn.Linear(in_features=feature_size * len(window_size),
                            out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.tensor(x, dtype=torch.float32)
        outputs = [conv(x) for conv in self.convs]
        output = torch.cat(outputs, dim=1)

        output = output.squeeze(-1)

        output = F.dropout(input=output, p=0.5)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output

    def training_step(self, train_batch, batch_idx):
        sequences, _, label, _ = train_batch
        output = self(sequences)
        loss = F.binary_cross_entropy(output, label, reduction='none')
        loss = loss.mean()
        self.log('train loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        sequences, _, label, _ = val_batch
        output = self(sequences)
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
SEQ_MAX_LEN = 150

train_dataset = train_dataWord2Vec()
val_dataset = val_dataWord2Vec()
test_dataset = test_dataWord2Vec()

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=8)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=4)

model = TextCNN(128, 90, [5, 6, 7, 8], SEQ_MAX_LEN)
trainer = pl.Trainer(max_epochs=100, gpus=1, callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(model, train_dl, val_dl)
trainer.test(model, test_dl)
