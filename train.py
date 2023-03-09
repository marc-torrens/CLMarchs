from data import Dataset, custom_collate, create_datasets
from finetuning import finetune
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.cuda import empty_cache

if __name__ == '__main__':
    vocabulary_file = 'Salesforce/codegen-350M-multi'
    pretrained_file = 'Salesforce/codegen-350M-multi'
    bug_train_file = "BFP_datasets/datasets/50/train/buggy.txt"
    bug_val_file = "BFP_datasets/datasets/50/eval/buggy.txt"
    fix_train_file = "BFP_datasets/datasets/50/train/fixed.txt"
    fix_val_file = "BFP_datasets/datasets/50/eval/fixed.txt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, val_data = create_datasets(bug_train_file, bug_val_file, fix_train_file, fix_val_file, vocabulary_file)
    train = train_data.divide_data(10)
    val = val_data.divide_data(10)
    for i, (train_dataset, val_dataset) in enumerate(zip(train, val)):
        empty_cache()
        finetune(train_dataset, val_dataset, pretrained_file, device, 1, 1, 'models/codegen_finetune-' + str(i))