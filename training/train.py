import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler

from models.reward_model import RewardModel
from utils.data_utils import load_data
from datasets import load_metric

class RLHFTrainer:
    def __init__(self, model_name='gpt2', device='cuda', lr=5e-5, batch_size=4, epochs=3):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = RewardModel(model_name=model_name, device=device)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_dataset = None
        self.val_dataset = None
        self.optimizer = None
        self.scheduler = None
        self.metric = load_metric('accuracy')

    def load_datasets(self, train_path, val_path):
        self.train_dataset = load_data(train_path)
        self.val_dataset = load_data(val_path)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def train_epoch(self, loader):
        self.model.train()
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(batch['input_ids'], labels=batch['labels'])
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

    def eval_epoch(self, loader):
        self.model.eval()
        total_acc = 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(batch['input_ids'], labels=batch['labels'])
            predictions = outputs.logits.argmax(-1)
            acc = self.metric.compute(predictions=predictions, references=batch['labels'])
            total_acc += acc['accuracy']
        return total_acc / len(loader)

    def train(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        num_training_steps = self.epochs * len(self.train_loader)
        self.scheduler = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        for epoch in range(self.epochs):
            self.train_epoch(self.train_loader)
            val_accuracy = self.eval_epoch(self.val_loader)
            print(f'Epoch {epoch+1}: Validation Accuracy: {val_accuracy:.4f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')
