from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from torch.cuda import empty_cache

empty_cache()

from test_data import Dataset, custom_collate

vocabulary_file = 'facebook/incoder-1B'
pretrained_file = 'facebook/incoder-1B'
batch_size = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(vocabulary_file)
model = AutoModelForCausalLM.from_pretrained(pretrained_file)

print(model.device)

print('model parameters:', sum(param.numel() for param in model.parameters()))

training_dataset = Dataset("BFP_datasets/datasets/50/train/buggy.txt", "BFP_datasets/datasets/50/train/fixed.txt", tokenizer)
validation_dataset = Dataset("BFP_datasets/datasets/50/train/buggy.txt", "BFP_datasets/datasets/50/eval/fixed.txt", tokenizer)
training_sampler = torch.utils.data.SequentialSampler(training_dataset)
validation_sampler = torch.utils.data.SequentialSampler(validation_dataset)
training_loader = torch.utils.data.DataLoader(
                                dataset=training_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True, sampler=training_sampler, collate_fn=custom_collate)
validation_loader = torch.utils.data.DataLoader(
                                dataset=validation_dataset, batch_size=3*batch_size, shuffle=False,
                                num_workers=0, pin_memory=True, sampler=validation_sampler, collate_fn=custom_collate
)