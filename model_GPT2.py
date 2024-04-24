import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
import tiktoken
import math


# use sin and cosine position encoding
class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, num_embed):
        super().__init__()
        
        self.num_embed = num_embed
        
        # Make initial positional encoding matrix with 0
        pe_matrix= torch.zeros(seq_len, num_embed) # (L, num_embed)

        # Calculate positional encoding values
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_embed, 2) * (-math.log(10000.0) / num_embed))
        pe_matrix[:, 0::2] = torch.sin(position * div_term)
        pe_matrix[:, 1::2] = torch.cos(position * div_term)
        
        pe_matrix = pe_matrix.unsqueeze(0) # (1, L, d_model)
        self.positional_encoding = pe_matrix.requires_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(self.num_embed) # (B, L, d_model)
        pos_enc = self.positional_encoding[:,:x.size(1),:] # trim off the excess
        x = x + pos_enc.to(x.device) # (B, L, d_model)

        return x
    
    

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, block_size, num_embed, dropout):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(num_embed, 3 * num_embed, bias=False)
        # output projection
        self.c_proj = nn.Linear(num_embed, num_embed, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.num_embed = num_embed
        self.dropout = dropout
        
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.num_embed, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, 
            dropout_p=self.dropout if self.training else 0, 
            is_causal=True)
        
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


    
    
# linear layer followed by a non-linearity
class FeedForward(nn.Module):
    def __init__(self, num_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embed, 4*num_embed),
            nn.ReLU(), 
            nn.Linear(4*num_embed, num_embed), # residual
            nn.Dropout(dropout),
        )
        
        
    def forward(self, x):
        return self.net(x)
    
    

# transformer decoder block: communication followed by computation
class DecoderBlock(nn.Module):
    def __init__(self, num_embed, block_size, num_heads, dropout):
        super().__init__()
        head_size = num_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, block_size, num_embed, dropout)
        self.ffwd = FeedForward(num_embed, dropout)
        self.layerNorm1 = nn.LayerNorm(num_embed)
        self.layerNorm2 = nn.LayerNorm(num_embed)
        
    def forward(self, x):
        x = x + self.sa(self.layerNorm1(x)) # residual connections
        x = x + self.ffwd(self.layerNorm2(x)) # residual connections
        return x
        
        
        
        
# main GPT2 class, decoder only Transformer
class GPT2(nn.Module):
    def __init__(self, vocab_size, block_size, num_embed, num_heads, num_layers, dropout):
        super().__init__()
        # parameters
        self.block_size = block_size
        
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, num_embed)
        #self.position_embedding_table = nn.Embedding(block_size, num_embed)
        self.position_embedding_table = PositionalEncoder(block_size, num_embed)
        self.decoder_blocks = nn.Sequential(*
            [DecoderBlock(num_embed, block_size, num_heads, dropout) for _ in range(num_layers)]   
        )
        self.layerNorm = nn.LayerNorm(num_embed)
        self.lm_head = nn.Linear(num_embed, vocab_size)
    
    
    def forward(self, idx, targets=None):
        # for torchsummary
        idx = idx.to(dtype=torch.long)
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        #pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        #x = tok_emb + pos_emb # (B, T, C)
        x = self.position_embedding_table(tok_emb) # (B, T, C)
        x = self.decoder_blocks(x) # (B, T, C)
        x = self.layerNorm(x) # (B, T, C)
        logits_3d = self.lm_head(x) # (B, T, vocab_size)
        
        # reshaping 3d tensor
        B, T, C = logits_3d.shape
        logits = logits_3d.view(B*T, C)
        
        if targets is None:
            loss = None
        else:
            targets = targets.view(-1)
            
            # define a loss
            loss = F.cross_entropy(logits, targets)
        
        return logits_3d, logits, loss
    
    
    # generate new tokens
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits3d, _, loss = self(idx_cond)
            # focus only on the last time step
            logits3d = logits3d[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits3d, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    


def main():
    encoder = tiktoken.get_encoding("gpt2")
    vocab_size = encoder.max_token_value

    block_size = 128 # max sequence length
    num_embed = 384
    num_heads = 6
    num_layers = 6
    dropout = 0.2

    # Creating model and testing output shapes 
    model = GPT2(vocab_size, block_size, num_embed, num_heads, num_layers, dropout)
    summary(model, (block_size,))
    
    

if __name__ == "__main__": 
    main()
    
    
    