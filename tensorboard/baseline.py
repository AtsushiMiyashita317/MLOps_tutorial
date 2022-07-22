#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
B4輪講最終課題 パターン認識に挑戦してみよう
ベースラインスクリプト(Pytorch Lightning版)
特徴量；MFCCの平均（0次項含まず）
識別器；MLP
"""

"""
pytorch-lightning 
    Docs: https://pytorch-lightning.readthedocs.io/
LightningModule
    Docs: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    API Refference: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html
Trainer
    Docs: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    API Refference: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html
"""


import argparse
import os

import hydra
from matplotlib import pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import seaborn as sns
import torch
import torchaudio
import torchmetrics
from torch.utils.data import Dataset, random_split

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class my_MLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim, model, loss_fn, optimizer):
        super().__init__()
        self.model = self.create_model(input_dim, output_dim, model)
        self.loss_fn = loss_fn
        self.optimizer = getattr(torch.optim, optimizer.optimizer)(self.model.parameters(), lr=optimizer.lr)
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.confm = torchmetrics.ConfusionMatrix(10, normalize='true')
        self.save_hyperparameters({**model, **optimizer}, logger=False)
      
    def create_model(self, input_dim, output_dim, model):
        """
        MLPモデルの構築
        Args:
            input_dim: 入力の形
            output_dim: 出力次元
        Returns:
            model: 定義済みモデル
        """
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, model.dim1),
            torch.nn.ReLU(),
            torch.nn.Dropout(model.dropout),
            torch.nn.Linear(model.dim1, model.dim2),
            torch.nn.ReLU(),
            torch.nn.Dropout(model.dropout),
            torch.nn.Linear(model.dim2, output_dim),
            torch.nn.Softmax(dim=-1)
        )
        # モデル構成の表示
        print(model)
        return model
      
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log('train/acc', self.train_acc(pred,y), on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        acc = self.val_acc(pred,y)
        self.log('val/acc', acc, prog_bar=True, logger=True)
        return acc
    
    def validation_epoch_end(self, outputs) -> None:
        avg_acc = torch.stack(outputs).mean()
        self.logger.log_hyperparams(self.hparams, metrics={'val/acc':avg_acc})
    
    def test_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        acc = self.test_acc(pred, y)
        self.log('test/acc', acc, prog_bar=True, logger=True)
        return {'pred':torch.argmax(pred, dim=-1), 'target':y}
    
    def test_epoch_end(self, outputs) -> None:
        # 混同行列を tensorboard に出力
        preds = torch.cat([tmp['pred'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        confusion_matrix = self.confm(preds, targets)
        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(10), columns=range(10))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='gray_r').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def configure_optimizers(self):
        return self.optimizer

class FSDD(Dataset):
    def __init__(self, path_list, label) -> None:
        super().__init__()
        self.features = self.feature_extraction(path_list)
        self.label = label

    def feature_extraction(self, path_list):
        """
        wavファイルのリストから特徴抽出を行いリストで返す
        扱う特徴量はMFCC13次元の平均（0次は含めない）
        Args:
            root: dataset が存在するディレクトリ
            path_list: 特徴抽出するファイルのパスリスト
        Returns:
            features: 特徴量
        """
        n_mfcc = 13
        datasize = len(path_list)
        features = torch.zeros(datasize, n_mfcc)
        transform = torchaudio.transforms.MFCC(n_mfcc=13, melkwargs={'n_mels':64, 'n_fft':512})
        for i, path in enumerate(path_list):
            # data.shape==(channel,time)
            data, _ = torchaudio.load(os.path.join(root, path))
            features[i] = torch.mean(transform(data[0]), axis=1)
        return features
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, index):
        return self.features[index], self.label[index]

@hydra.main(config_path='conf',config_name='config.yaml')
def main(cfg):
    # データの読み込み
    training = pd.read_csv(os.path.join(root, "training.csv"))
    
    # Dataset の作成
    train_dataset = FSDD(training["path"].values, training['label'].values)
    
    # Train/Validation 分割
    val_size = int(len(train_dataset)*0.2)
    train_size = len(train_dataset)-val_size
    train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            torch.Generator().manual_seed(20200616))

    # Test Dataset の作成
    test = pd.read_csv(os.path.join(root, "test.csv"))
    test_dataset = FSDD(test["path"].values, test['label'].values)
    
    # DataModule の作成
    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=32,
        num_workers=4)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    # モデルの構築
    model = my_MLP(input_dim=train_dataset[0][0].shape[0],
                   output_dim=10,
                   model=cfg.model,
                   loss_fn=loss_fn,
                   optimizer=cfg.optim)
    
    # 学習の設定
    logger = TensorBoardLogger(
        name=cfg.experiment_name, 
        save_dir=os.path.join(
            hydra.utils.get_original_cwd(),
            'lightning_logs'),
        default_hp_metric=False)
    trainer = pl.Trainer(max_epochs=100, gpus=1, logger=logger)
        
    # モデルの学習
    trainer.fit(model=model, datamodule=datamodule)
    
    # バリデーション
    trainer.validate(model=model, datamodule=datamodule)
    
    # テスト
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
