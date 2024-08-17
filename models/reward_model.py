import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

class RewardModel:
    def __init__(self, model_name='gpt2', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.optimizer = None

    def train(self, train_dataset, val_dataset=None, epochs=3, batch_size=4, learning_rate=5e-5):
        self.model.train()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        total_steps = len(train_loader) * epochs
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(epochs):
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for batch in progress_bar:
                inputs, labels = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
                self.model.zero_grad()
                outputs = self.model(input_ids=inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                progress_bar.set_postfix(loss=loss.item())

        if val_dataset:
            self.evaluate(val_dataset)

    def evaluate(self, val_dataset, batch_size=4):
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
                outputs = self.model(input_ids=inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        print(f'Validation Loss: {avg_loss}')
        return avg_loss

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        print(f'Model loaded from {model_path}')

