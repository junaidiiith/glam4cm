import os
import time
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torch.nn as nn

from models.cmgpt import CMGPT
import torch
from settings import device

class CMGPTTrainer:
    def __init__(
        self,
        model: CMGPT,
        dataset: Dataset,
        test_ratio: float = 0.05,
        batch_size: int = 32,

        lr: float = 1e-5,
        num_epochs: int = 10,
        log_dir: str = 'logs',
        results_dir: str = 'results/cmgpt',
    ):
        self.model = model
        self.model.to(device)

        # self.model = torch.compile(self.model)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        train_dataset, test_dataset = train_test_split(dataset, test_size=test_ratio)
        self.dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        }

        self.num_epochs = num_epochs


    def step(self, batch, idx=None):
        # B, T = batch['input_ids'].shape
        # t0 = time.time()
        self.optimizer.zero_grad()
        logits = self.model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        loss = self.model.get_loss(logits, batch['input_ids'].to(device))
        
            
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
        return loss


    def train(self):
        for epoch in tqdm(range(self.num_epochs), desc='Training Epoch'):
            self.model.train()
            train_loss = 0
            for i, batch in tqdm(enumerate(self.dataloaders['train']), desc='Training Batches', total=len(self.dataloaders['train'])):
                loss = self.step(batch, i)
                train_loss += loss.item()

                self.writer.add_scalar('loss/train', loss.item(), epoch * len(self.dataloaders['train']) + i)
                # break
            
            
            print("Train loss: ", train_loss / len(self.dataloaders['train']))
            
            self.test(epoch)
            self.scheduler.step()


    def test(self, epoch=None):
        self.model.eval()
        test_loss = 0
        for i, batch in tqdm(enumerate(self.dataloaders['test']), desc='Testing Batches', total=len(self.dataloaders['test'])):
            logits = self.model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            loss = self.model.get_loss(logits, batch['input_ids'].to(device))
            if isinstance(self.model, nn.DataParallel):
                loss = self.model.module.get_loss(logits, batch['input_ids'].to(device))
            else:
                loss = self.model.get_loss(logits, batch['input_ids'].to(device))

            test_loss += loss.item()

            if epoch is not None:
                self.writer.add_scalar('loss/test', loss.item(), epoch * len(self.dataloaders['test']) + i)
            
            # break

        print("Test loss: ", test_loss / len(self.dataloaders['test']))
    

    def save_model(self):
        path = f'{self.results_dir}/cmgpt.pth'
        torch.save(self.model.state_dict(), path)