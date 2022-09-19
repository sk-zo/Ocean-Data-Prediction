import torch
import torch.nn as nn
from torch.nn import MSELoss
from tqdm import tqdm
from transformers import AdamW

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

class Trainer:
    def __init__(self, args, model, dataset):
        self.args = args
        self.model = model
        self.dataset = dataset

    def fit(self, eval=False):
        device = self.args.device
        loss_fct = MSELoss()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        
        max_epoch = self.args.max_epoch
        # max_grad_norm = self.args.max_grad_norm

        for epoch in range(max_epoch):
            self.model.train()
            loss_log = 0
            for (inputs, labels) in tqdm(self.dataset, total=len(self.dataset)):
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)

                out = self.model(inputs)
                loss = loss_fct(out, labels)
                loss.backward()
                loss_log += loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

            print(f"epoch: {epoch+1}, loss: {loss_log.item()/len(self.dataset)}")
        with open(self.args.model_save_path + "_loss.txt", 'w') as f:
            f.write(str(loss_log.item()/len(self.dataset)))

     
    def save_model(self):
        model_save_path = self.args.model_save_path
        torch.save(self.model.state_dict(), model_save_path)
        print(f"model_save_path: {model_save_path}")
