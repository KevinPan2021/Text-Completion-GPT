import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm


from model_GPT2 import GPT2
from training import model_training

# supports MacOS mps and CUDA
def GPU_Device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'


def batch_tokenize_lines(input_path, batch_size, max_lines):
    with open(input_path, 'r', encoding='utf-8') as text_file:
        i = 0
        lines = []
        for line in text_file:
            lines.append(line)
            if len(lines) >= batch_size:
                yield lines
                lines = []
            if max_lines!=None and i > max_lines:
                break
            i += 1
        if lines:
            yield lines


# convert from text file to binary file
def convert_to_tokenized_binary(input_path, output_path, tokenizer, max_lines=3e5, batch_size=1000):
    with open(output_path, 'ab') as file:
        for batch in tqdm(batch_tokenize_lines(input_path, batch_size, max_lines)):
            token_values_batch = []
            for line in batch:
                token_values = tokenizer.encode_ordinary(line)
                token_values_batch.extend(token_values)
            token_values_batch = np.array(token_values_batch).astype(np.ushort)
            token_values_batch.tofile(file)
            
            

class LargeTextDataset(Dataset):
    def __init__(self, file_path, seq_length):
        self.file_path = file_path
        self.seq_length = seq_length
        self.num_tokens = None
        
        # Open the file and get the number of tokens
        with open(file_path, 'rb') as file:
            file.seek(0, 2)
            self.num_tokens = np.int64(file.tell()) // 2  # Each token is stored as ushort
        
    
    def get_num_tokens(self):
        return self.num_tokens
    
    
    def __len__(self):
        return self.num_tokens // (self.seq_length)
    
    
    def __getitem__(self, idx):
        # Calculate the offset in the file for the given index
        offset = idx * (self.seq_length) * 2  # 2 bytes per token
        
        # Read the data from the file
        with open(self.file_path, 'rb') as file:
            file.seek(offset)
            data = np.fromfile(file, dtype=np.ushort, count=self.seq_length)
        
        # Convert to PyTorch tensor
        data = data.astype(np.int32)
        data_tensor = torch.tensor(data, dtype=torch.long)
        
        x = data_tensor[:-1]
        y = data_tensor[1:]
        return x, y

        
            
            
def main():
    # file path
    file_path = '../Datasets/OpenWebText/'
    
    # tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # building the vocab
    vocab_size = tokenizer.max_token_value + 1
    
    # convert from raw text file to tokenized binary file
    train_txt_file = file_path + 'train_split.txt'
    train_bin_file = 'train.bin'
    valid_txt_file = file_path + 'val_split.txt'
    valid_bin_file = 'valid.bin'
    if train_bin_file not in os.listdir():
        # load the first 300,000 lines
        convert_to_tokenized_binary(train_txt_file, train_bin_file, tokenizer, max_lines=3e5)
    if valid_bin_file not in os.listdir():
        # load the first 30,000 lines
        convert_to_tokenized_binary(valid_txt_file, valid_bin_file, tokenizer, max_lines=3e4)
    
    
    # Define train and valid data loader
    batch_size = 64
    block_size = 64 # max sequence length
    train_dataset = LargeTextDataset(train_bin_file, block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = LargeTextDataset(valid_bin_file, block_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    

    # load model
    num_embed = 384
    num_heads = 6
    num_layers = 6
    dropout = 0.2
    model = GPT2(vocab_size, block_size, num_embed, num_heads, num_layers, dropout)
    model = model.to(GPU_Device()) # move the model to device
    
    # model training
    model_training(model, train_loader, valid_loader, GPU_Device())
    
    # loading the best model
    model.load_state_dict(torch.load(f'{type(model).__name__}.pth'))
    
    # generate
    context = torch.zeros((1,1), dtype=torch.long).to(GPU_Device())
    generated = model.generate(context, max_new_tokens=500)[0].tolist()
    generated = tokenizer.decode(generated)
    print(generated)
    
    
if __name__ == '__main__':
    main()