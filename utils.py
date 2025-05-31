import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import functional as F
import torch.nn as nn
import math
from dataprocessing import TextDataset


def temperature_sampling(history, logits, temperature, top_k_num, repetition_penalty):
    """
    Perform temperature sampling on logits

    Args:
        history: output history
        repetition_penalty: the coefficient of reducing the probability of selecting a repetitive token
        top_k_num: only tokens with the top_k_num-th biggest prob will be selected
        logits (torch.Tensor): Logits tensor from the model
        temperature (float): Temperature parameter

    Returns:
        torch.Tensor: Sampled output tensor
    """
    # Scale the logits by the temperature
    scaled_logits = logits / temperature
    # Apply repetition penalty and top_k
    top_k_logits, top_k_tokens = torch.topk(scaled_logits, top_k_num)
    for token_idx in range(top_k_num):
        if top_k_tokens[token_idx] in history:
            top_k_logits[token_idx] /= repetition_penalty
    # Apply softmax to get a probability distribution
    probs = F.softmax(top_k_logits, dim=-1)

    # Sample from the probability distribution
    sample = top_k_tokens[torch.multinomial(probs, num_samples=1)[0]]

    return sample


class CosineDecayWithWarmupLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Follows the GPT-3 paper"""

    def __init__(self, optimizer, min_lr, max_lr, warmup_steps, max_decay_steps, last_epoch=-1, verbose=False) -> None:
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.max_decay_steps = max_decay_steps

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warm-up phase
            return [self.min_lr + (self.max_lr - self.min_lr) * self.last_epoch / self.warmup_steps] * len(
                self.optimizer.param_groups
            )
        elif self.warmup_steps <= self.last_epoch < self.max_decay_steps:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / (self.max_decay_steps - self.warmup_steps)
            return [self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))] * len(
                self.optimizer.param_groups
            )
        else:
            return [self.min_lr] * len(self.optimizer.param_groups)


def obtain_grad_norm(model: nn.Module):
    # measures a model's gradient
    total_norm = 0
    for parms in model.parameters():
        if parms.requires_grad and parms.grad is not None:
            gradient_norm = parms.grad.norm()
            total_norm += gradient_norm ** 2
    total_norm = total_norm.sqrt()
    return total_norm.item()


def get_eval_loss(gpt_model, eval_dataloader, vocab_size):
    with torch.no_grad():
        eval_src, eval_tgt = next(iter(eval_dataloader))
        eval_output = gpt_model.forward_no_inference(eval_src)
        eval_loss = F.cross_entropy(eval_output.view(-1, vocab_size), eval_tgt.view(-1))
        return eval_loss.item()


def eval_simplegpt_model(model: nn.Module, tokenizer, initial_text, seq_len=1024, temperature=0.5, top_k_num=10, repetition_penalty=1.4):
    model.eval()
    device = next(model.parameters()).device
    print(initial_text+"<bos>", end="")
    text_len = 800
    token_ids = tokenizer.encode(1024*chr(0))+tokenizer.encode(initial_text)
    for i in range(text_len):
        token_ids.append(temperature_sampling(token_ids[-200:],
                                              model(torch.tensor(token_ids)[-seq_len:].to(device)),
                                              temperature, top_k_num, repetition_penalty).item())
        print(tokenizer.decode(token_ids[-1]), end='')


def train_simplegpt_model(model, device,  train_dataset_path, eval_dataset_path, save_path,
                          num_epochs, training_ratio=1, lr=1e-4, batch_size=2,
                          gradient_accumulation_steps=8, seq_len=1024):
    # 创建数据集
    train_dataset = TextDataset(json_path=train_dataset_path, seq_len=seq_len, device=device)
    eval_dataset = TextDataset(json_path=eval_dataset_path, seq_len=seq_len, device=device)

    # 创建数据加载器
    train_data_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    eval_data_loader = DataLoader(eval_dataset, sampler=RandomSampler(eval_dataset), batch_size=4)
    train_data_length = len(train_data_loader)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # use weight decay to reduce overfit
    # note: 1e-4 may be a better learning rate
    # scheduler = CosineDecayWithWarmupLRScheduler(optimizer, 0.5e-4, 4e-4, 2000, 6000)

    # 训练模型
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for steps, batch in enumerate(train_data_loader):
            src, tgt = batch
            output = model.forward_no_inference(src)
            # You should use no inference mode here, the outcome will be much better
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            total_loss += loss.item()
            # Note: this is quite weird. Only you add up the losses of the sequence you will find the loss is very low
            # Update: Over_fitted? Yes, it has been proven.
            # Only one epoch on the training set can make the model memorize everything
            loss = loss/gradient_accumulation_steps
            loss.backward()
            # scheduler.step()
            # logging
            if (steps+1) % 256 == 0:
                eval_loss = get_eval_loss(gpt_model=model, eval_dataloader=eval_data_loader,
                                          vocab_size=output.size(-1))
                print(f"Steps:{steps}, Loss: {total_loss / 256}, "
                      f"Current gradient norm:{obtain_grad_norm(model)}, eval_loss:{eval_loss}")
                total_loss = 0
            # accumulate gradient. update model only when accumulated enough steps
            if (steps+1) % gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()

            if (steps+1) % 2048 == 0:
                # Save models in every 1000 steps
                torch.save(model, save_path)
                print("model saved")
                # Early stop
                if steps > (train_data_length/seq_len*1.5*training_ratio):
                    print("training finished, early stopping...")
                    break


def training_simplegpt_with_scheduler(model, device,  training_schedule, eval_dataset_path, save_path,
                                      num_epochs, lr=1e-4, batch_size=2,
                                      gradient_accumulation_steps=8, seq_len=1024):
    for training_parts in training_schedule:
        train_simplegpt_model(model, device,  training_parts["file_path"], eval_dataset_path, save_path,
                              num_epochs, training_ratio=training_parts["ratio"], lr=lr, batch_size=batch_size,
                              gradient_accumulation_steps=gradient_accumulation_steps, seq_len=seq_len)
