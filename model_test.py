from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import torch.nn as nn
from torch.cuda import empty_cache
from transformers import get_cosine_schedule_with_warmup, Adafactor


empty_cache()

from test_data import Dataset, custom_collate


def validation_step(model, validation_loader, save_dir):
    print('start validation')
    validation_loss = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            data = {
                'input_ids': data['input_ids'].to(device),
                'labels': data['labels'].to(device),
                'attention_mask': data['attention_mask'].to(device)
            }
            output = model(input_ids=data['input_ids'], labels=data['labels'], attention_mask=data['attention_mask'], return_dict=True)
            loss = output.loss
            validation_loss.append(loss.mean().item())
    print('validation loss:', round(sum(validation_loss) / len(validation_loss), 4))

    model.save_pretrained(save_dir)
    model.train()

def finetune(bug_train_file, bug_val_file, fix_train_file, fix_val_file, epochs=5, batch_size=32, save_dir='models/codegen-350M-finetune'):
    tokenizer = AutoTokenizer.from_pretrained(vocabulary_file)
    model = AutoModelForCausalLM.from_pretrained(pretrained_file)
    model.to(device)
    training_dataset = Dataset(bug_train_file, fix_train_file, tokenizer)
    validation_dataset = Dataset(bug_val_file, fix_val_file, tokenizer)
    training_sampler = torch.utils.data.SequentialSampler(training_dataset)
    validation_sampler = torch.utils.data.SequentialSampler(validation_dataset)
    training_loader = torch.utils.data.DataLoader(
                                    dataset=training_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True, sampler=training_sampler, collate_fn=custom_collate)
    validation_loader = torch.utils.data.DataLoader(
                                    dataset=validation_dataset, batch_size=3*batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True, sampler=validation_sampler, collate_fn=custom_collate)
    
    optimizer = Adafactor(model.parameters(), lr=1e-5, scale_parameter=False, relative_step=False)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=0, num_training_steps=int(epochs * len(training_loader))
    ) # Modifies learning rate during training

    for epoch in range(epochs):
        model.train()
        training_loss = []
        start_time = time.time()
        oom = 0
        for i, data in enumerate(training_loader):
            data = {
                'input_ids': data['input_ids'].to(device),
                'labels': data['labels'].to(device),
                'attention_mask': data['attention_mask'].to(device)
            }
            try:
                optimizer.zero_grad()
                output = model(input_ids=data['input_ids'], labels=data['labels'], attention_mask=data['attention_mask'], return_dict=True)
                loss = output.loss
                loss.mean().backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.3)
                optimizer.step()
                training_loss.append(loss.mean().item())
            except Exception as e:
                print(str(e))
                if 'out of memory' in str(e):
                    oom += 1
                model.zero_grad()
                optimizer.zero_grad()
                del data

                empty_cache()

            if i % 1000 == 0:
                print('epoch: {}, step: {}/{}, loss: {}, lr: {}, oom: {}, time: {}s'.format(
                    epoch + 1, i, len(training_loader),
                    round(sum(training_loss) / len(training_loss), 4),
                    round(scheduler.get_last_lr()[0], 7), oom,
                    int(time.time() - start_time)
                ))
                start_time = time.time()
                oom = 0
            if i % 10000 == 0 and i > 0:
                validation_step(model, validation_loader, save_dir)
        validation_step(model, validation_loader, save_dir)
 

if __name__ == '__main__':
    vocabulary_file = 'Salesforce/codegen-350M-multi'
    pretrained_file = 'Salesforce/codegen-350M-multi'
    batch_size = 1
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bug_train_file = "BFP_datasets/datasets/50/train/buggy.txt"
    bug_val_file = "BFP_datasets/datasets/50/eval/buggy.txt"
    fix_train_file = "BFP_datasets/datasets/50/train/fixed.txt"
    fix_val_file = "BFP_datasets/datasets/50/eval/fixed.txt"

    finetune(bug_train_file, bug_val_file, fix_train_file, fix_val_file, epochs, batch_size)