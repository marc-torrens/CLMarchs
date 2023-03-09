import torch
import random
import numpy as np
from transformers import AutoTokenizer

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path_bug, file_path_fixed, tokenizer, shuffle=False, seed = 123, load_range=None):
        self.data = []
        f_bug = open(file_path_bug, "r")
        f_fix = open(file_path_fixed, "r")

        for (l_bug, l_fix) in zip(f_bug.readlines(), f_fix.readlines()):
            inputs = tokenizer.encode(l_bug, return_tensors='pt')

            outputs = tokenizer.encode(l_fix, return_tensors='pt')

            self.data.append({
                'input_ids': torch.cat([torch.zeros(1, max(0, outputs.size(1) - inputs.size(1))).fill_(220).long(), inputs], dim=1),
                'labels': torch.cat([torch.zeros(1, max(0, inputs.size(1) - outputs.size(1))).fill_(200).long(), outputs], dim=1),
                'attention_mask': torch.ones(max(outputs.size(), inputs.size())).long()
            })

        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)
        
        if load_range is not None:
            self.data = self.data[load_range[0]: ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]

    def divide_data(self, parts):
        return np.array_split(self.data, parts)

    def combine(self, dataset2, shuffle):
        data = self.data + dataset2  
        if shuffle:
            random.shuffle(data)
        return data
    
def create_datasets(bug_train_file, bug_val_file, fix_train_file, fix_val_file, vocabulary_file):
    tokenizer = AutoTokenizer.from_pretrained(vocabulary_file)
    training_dataset = Dataset(bug_train_file, fix_train_file, tokenizer)
    validation_dataset = Dataset(bug_val_file, fix_val_file, tokenizer)
    return training_dataset, validation_dataset

def custom_collate(batch):
    batch_data = {'input_ids': [], 'labels': [], 'attention_mask': []}
    max_len = max([b['input_ids'].size(1) for b in batch])
    eos_id = 50517
    for b in batch:
        batch_data['input_ids'].append(torch.cat([b['input_ids'], torch.zeros(1, max_len - b['input_ids'].size(1)).fill_(eos_id).long()], dim=1))
        batch_data['labels'].append(torch.cat([b['labels'], torch.zeros(1, max_len - b['labels'].size(1)).fill_(220).long()], dim=1))
        batch_data['attention_mask'].append(torch.cat([b['attention_mask'], torch.zeros(1, max_len - b['attention_mask'].size(1))], dim=1))
    batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
    batch_data['labels'] = torch.cat(batch_data['labels'], dim=0)
    batch_data['attention_mask'] = torch.cat(batch_data['attention_mask'], dim=0)
    return batch_data