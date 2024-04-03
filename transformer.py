import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    
    def __init__(self, d_model, n_heads, device):
        
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        
        assert d_model % n_heads == 0, "n_heads must divide d_model"
        
        # 4x comes from original paper
        self.ffn = nn.Sequential(
            nn.Linear(d_model,  4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        # just matrix multiplications
        self.key_net = nn.Sequential(
            nn.Linear(d_model, d_model)
        )
        
        self.query_net = nn.Sequential(
            nn.Linear(d_model, d_model)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(d_model, d_model)
        )
        
        self.after_attention_ln = nn.LayerNorm(d_model)
        self.after_ffn_ln = nn.LayerNorm(d_model)
        
        self.dropout_after_attention = nn.Dropout(p = 0.1)
        self.dropout_after_fnn = nn.Dropout(p = 0.1)
    
    def forward(self, x):
        
        B, T, _ = x.shape
        
        # != self.key_net(x).view((B, self.n_heads, T, -1))
        # with the above, future leaks into past
        keys = self.key_net(x).view((B, T, self.n_heads, -1)).transpose(1, 2)
        queries = self.query_net(x).view((B, T, self.n_heads, -1)).transpose(1, 2)
        values = self.value_net(x).view((B, T, self.n_heads, -1)).transpose(1, 2)
        
        scaling_factor = 1.0 / math.sqrt(self.d_model / self.n_heads)
        attention_matrices = scaling_factor * torch.matmul(queries, keys.transpose(2, 3))
        
        neg_inf = -1e10
        
        # mask the future (upper triangle)
        mask = torch.tril(torch.ones(T, T)).to(self.device)
        mask = mask.masked_fill(mask == 0, -float("inf"))
                        
        # softmax per row
        activated_attention_matrices = F.softmax(attention_matrices + mask, dim=-1)
                
        # (B, head, T, dim_per_head)
        # d_model = head * dim_per_head
        att_output = torch.matmul(activated_attention_matrices, values)
        
        att_output = torch.transpose(att_output, 1, 2)
        
        after_attention_dropout = self.dropout_after_attention(att_output.reshape((B, T, -1)))
        ffn_input = self.after_attention_ln(after_attention_dropout + x)
        
        ffn_output = self.ffn(ffn_input)
        
        return self.after_ffn_ln(self.dropout_after_fnn(ffn_output) + ffn_input)
    
class Transformer(nn.Module):
    
    def __init__(self, n_symbols, context_length, d_model, n_heads, n_layers, device):
        
        super().__init__()
        
        self.n_symbols = n_symbols
        self.d_model = d_model
        self.context_length = context_length
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.device=device
        
        self.token_embedding = nn.Embedding(num_embeddings=n_symbols, embedding_dim=d_model)
        self.pos_embedding = nn.Embedding(num_embeddings=context_length, embedding_dim=d_model)
        
        # TODO: multiple transformer blocks
        tbs = [TransformerBlock(d_model = d_model, n_heads = n_heads, device=device) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*tbs)
        
        self.to_logits = nn.Sequential(
            nn.Linear(d_model, n_symbols, device=device)
        )
        
        self.embedding_dropout = nn.Dropout(p = 0.1)
        
    def forward(self, x):
        
        # batch, time
        B, T = x.shape
        
        embedded = self.token_embedding(x)
        #print(f"{embedded.shape}")
        
        positions = torch.arange(T).to(self.device)
        #print(f"{positions.shape}")

        embedded = self.embedding_dropout(embedded + self.pos_embedding(positions))
        
        after_transformer_layers = self.transformer_blocks(embedded)
                
        return self.to_logits(after_transformer_layers)
    
    def sample(self, prompt, n_tokens, n_samples, c_to_i, beta = 1.0):
        self.eval()
        self.to(self.device)

        # Process the prompt to fit within the context length
        prompt = prompt[-self.context_length:]
        print(f"Prompt: {prompt}")
        
        prompt_tokens = [c_to_i[c] for c in prompt]
        print(f"Prompt tokens: {prompt_tokens}")

        context = torch.tensor(prompt_tokens, dtype=torch.long).repeat(n_samples, 1).to(self.device)
        
        history = torch.zeros_like(context)
        
        for _ in range(n_tokens):
            with torch.no_grad():
                logits = self(context)[:, -1, :] / beta  # Get logits for the last token position only
                probs = F.softmax(logits, dim=-1)
                last_sampled_token = torch.multinomial(probs, num_samples=1)
                
                history = torch.cat((history, last_sampled_token), dim=1)
                context = torch.cat((context, last_sampled_token), dim=1)[:, -self.context_length:]  # Update context
                
                
        response = history[:, -n_tokens:]
        return response
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f'Model saved to {path}')
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')