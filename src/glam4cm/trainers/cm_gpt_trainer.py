import os
import time
from typing import Union
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn as nn

from glam4cm.models.cmgpt import CMGPT, CMGPTClassifier
import torch
from glam4cm.settings import device


from glam4cm.trainers.metrics import compute_classification_metrics


class CMGPTTrainer:
    def __init__(
        self,
        model: Union[CMGPT, CMGPTClassifier],
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int = 32,

        lr: float = 1e-5,
        num_epochs: int = 10,
        log_dir: str = 'logs',
        results_dir: str = 'results/cmgpt',
        compute_metrics: callable = None
    ):
        self.model = model
        self.model.to(device)

        # self.model = torch.compile(self.model)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        }

        self.num_epochs = num_epochs
        if not compute_metrics and isinstance(self.model, CMGPTClassifier):
            self.compute_metrics = compute_classification_metrics
        else:
            self.compute_metrics = compute_metrics
        
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)/ 1000000:.3f}M")


    def step(self, batch, idx=None):
        # B, T = batch['input_ids'].shape
        # t0 = time.time()
        self.optimizer.zero_grad()
        logits, loss = self.model(
            batch['input_ids'].to(device), 
            batch['attention_mask'].to(device),
            batch['labels'].to(device)
        )
            
        loss.backward()
        self.optimizer.step()
        # torch.cuda.synchronize()
        # t1 = time.time()
        # dt = (t1 - t0)*1000
        # tokens_per_sec = B*T/(t1-t0)
        # if idx is not None:
        #     print(f"Batch: {idx}, Loss: {loss.item()}, Time: {dt} ms, Tokens/s: {tokens_per_sec}")
        # else:
        #     print(f"Loss: {loss.item()}, Time: {dt} ms, Tokens/s: {tokens_per_sec}")
        # if idx > 100:
        #     print("Breaking")
        #     exit()
        return logits, loss


    def train(self):
        for epoch in tqdm(range(self.num_epochs), desc='Training Epoch'):
            self.model.train()
            train_loss = 0
            all_preds, all_labels = list(), list()
            for i, batch in tqdm(enumerate(self.dataloaders['train']), desc='Training Batches', total=len(self.dataloaders['train'])):
                logits, loss = self.step(batch, i)
                train_loss += loss.item()

                self.writer.add_scalar('loss/train', loss.item(), epoch * len(self.dataloaders['train']) + i)

                if self.compute_metrics is not None:
                    all_preds.append(logits.detach().cpu())
                    all_labels.append(batch['labels'].cpu())
                # break
            print("Train loss: ", train_loss / len(self.dataloaders['train']))
            
            # if self.compute_metrics is not None:
            #     all_preds = torch.cat(all_preds, dim=0)
            #     all_labels = torch.cat(all_labels, dim=0)
            #     metrics = self.compute_metrics(all_preds, all_labels)
            #     for key, value in metrics.items():
            #         self.writer.add_scalar(key, value, epoch)

            #     # print("Train Metrics: ", metrics) 
            
            
            
            self.test(epoch)
            self.scheduler.step()


    def test(self, epoch=None):
        self.model.eval()
        test_loss = 0
        all_preds, all_labels = list(), list()
        for i, batch in tqdm(enumerate(self.dataloaders['test']), desc='Testing Batches', total=len(self.dataloaders['test'])):
            logits, loss = self.model(
                batch['input_ids'].to(device), 
                batch['attention_mask'].to(device),
                batch['labels'].to(device)
            )
            test_loss += loss.item()

            if self.compute_metrics is not None:
                all_preds.append(logits.detach().cpu())
                all_labels.append(batch['labels'].cpu())
            
        if epoch is not None:
            self.writer.add_scalar('loss/test', test_loss / len(self.dataloaders['test']), epoch)
            
            # break

        print("Test loss: ", test_loss / len(self.dataloaders['test']))

        if self.compute_metrics is not None:
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            metrics = self.compute_metrics(all_preds, all_labels)
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)

            print("Test Metrics: ", metrics)
    

    def save_model(self):
        if isinstance(self.model, CMGPT):
            path = f'{self.results_dir}/cmgpt.pth'
        elif isinstance(self.model, CMGPTClassifier):
            path = f'{self.results_dir}/cmgpt-classifier.pth'
        torch.save(self.model.state_dict(), path)