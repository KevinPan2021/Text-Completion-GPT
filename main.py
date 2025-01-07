import torch
from torch.nn import functional as F
import tiktoken

from model_converter import load_from_standard_weights
from gpt import GPT2


# CUDA or CPU
def compute_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


@torch.no_grad()
def inference(model, tokenizer, input_sentence, max_new_tokens=512, 
              temperature=1.0, top_k=None):
    model = model.eval()
    
    # encode string to list
    input_tok = tokenizer.encode(input_sentence)
    
    # convert to tensor
    input_tok = torch.tensor(input_tok, dtype=torch.long)
    
    # unsqueeze the batch dimension
    input_tok = input_tok.unsqueeze(0)
    
    # move to compute device
    idx = input_tok.to(compute_device())
    
    # generate
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -1024:]
       # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        
    generated = idx[0].tolist()
    
    # decode
    generated = tokenizer.decode(generated)[len(input_sentence):]
    
    return generated


            
def main():
    # create GPT model
    # load model
    num_embed = 768
    num_heads = 12
    num_layers = 12
    model = GPT2(num_embed, num_heads, num_layers)
    model = model.to(compute_device())
    
    # load the pretained weights
    pretrained_path = '../pretrained_models/GPT/GPT2.bin'
    model.load_state_dict(load_from_standard_weights(pretrained_path))

    # tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # inference
    input_sentence = 'Once upon a time, '
    generated = inference(model, tokenizer, input_sentence)
    print(input_sentence + generated)
    
    
if __name__ == '__main__':
    main()