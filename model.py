import numpy as np

#! hyperparameters
class Config:
    vocab_size = 32768
    ctx_length = 512
    n_layers = 4
    d_model = 256
    n_heads = 8

#! GPT classes
class Linear:
    def __init__(self, features_in: int, features_out: int, bias: bool=True):
        k = 1 / np.sqrt(features_in)

        self.w = np.random.uniform(low=-k, high=k, size=(features_in, features_out))
        self.b = np.random.uniform(low=-k, high=k, size=(features_out))

        self.bias = bias

    def __call__(self, x):
        return x @ self.w + self.b if self.bias else x @ self.w

class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.w = np.random.normal(loc=0, scale=1, size=(num_embeddings, embedding_dim))

        self.embedding_dim = embedding_dim

    def __call__(self, x):
        B, T = x.shape

        new_x = np.zeros((B, T, self.embedding_dim))

        for i in range(B):
            for j in range(T):
                new_x[i, j] = self.w[x[i, j]]

        return np.array(new_x)

class LayerNorm:
    def __init__(self, *normalized_shape: list, eps: int=1e-5, bias: bool=True):
        self.w = np.ones((normalized_shape))
        self.b = np.zeros((normalized_shape))

        self.eps = eps
        self.bias = bias

    def __call__(self, x):
        _, T, _ = x.shape

        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        norm = (x - mean) / np.sqrt(var + self.eps) * self.w + self.b if self.bias else \
               (x - mean) / np.sqrt(var + self.eps) * self.w

        return norm
    
#! GPT functions
def softmax(input, dim):
    shifted = input - np.max(input, axis=dim, keepdims=True)

    exp = np.exp(shifted)

    sum_exp = np.sum(exp, axis=dim, keepdims=True)

    return exp / sum_exp

def gelu(x):
    return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

#! model
class PreProcessing(Config):
    def __init__(self):
        self.embedding = Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = Embedding(self.ctx_length, self.d_model)

        self.n_params = (self.vocab_size * self.d_model) + (self.ctx_length * self.d_model)

    def __call__(self, tokens):
        emb = self.embedding(tokens) # (B, T, C)
        pe = self.pos_encoding(np.arange(0, T).reshape(1, -1)) # (1, T, C)

        x = emb + pe # (B, T, C)

        return x

class Block(Config):
    def __init__(self):
        self.causal_sa = CausalSelfAttention()
        self.ln_1 = LayerNorm(self.d_model)

        self.fnn = FNN()
        self.ln_2 = LayerNorm(self.d_model)

        self.n_params = self.causal_sa.n_params + (self.d_model * 2) + self.fnn.n_params + (self.d_model * 2)

    def __call__(self, x):
        x = x + self.causal_sa(self.ln_1(x))
        x = x + self.fnn(self.ln_1(x))

        return x

class CausalSelfAttention(Config):
    def __init__(self):
        self.q = Linear(self.d_model, self.d_model)
        self.k = Linear(self.d_model, self.d_model)
        self.v = Linear(self.d_model, self.d_model)

        self.ll = Linear(self.d_model, self.d_model)

        self.n_params = (self.d_model * self.d_model) * 4

    def __call__(self, x):
        B, T, C = x.shape

        assert self.d_model % self.n_heads == 0, 'd_model has to be divisible by head_size'

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
   
        q = q.reshape(B, T, self.n_heads, -1).transpose(0, 2, 1, 3) # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.reshape(B, T, self.n_heads, -1).transpose(0, 2, 1, 3) # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        v = v.reshape(B, T, self.n_heads, -1).transpose(0, 2, 1, 3) # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)

        attn = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_model) # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        mask = np.tril(np.ones((T, T), dtype=bool))  # (T, T)
        attn = np.where(mask, attn, -1e10) # (B, nh, T, T)
        attn = softmax(attn, dim=2)

        y = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C) # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) -> (B, T, C)

        y = self.ll(y) # (B, T, C)

        return y

class FNN(Config):
    def __init__(self):
        self.ll_1 = Linear(self.d_model, 4 * self.d_model)
        self.ll_2 = Linear(4 * self.d_model, self.d_model)

        self.n_params = (self.d_model * 4*self.d_model) + (self.d_model * 4*self.d_model)

    def __call__(self, x):
        x = self.ll_1(x)
        x = gelu(x)
        x = self.ll_2(x)

        return x

class PostProcessing(Config):
    def __init__(self):
        self.ll = Linear(self.d_model, self.vocab_size)
        self.ln = LayerNorm(self.d_model)

        self.n_params = (self.d_model * self.vocab_size) + (self.d_model * self.d_model)

    def __call__(self, x):
        x = self.ln(x) # (B, T, C)
        x = self.ll(x) # (B, T, V)
        logits = softmax(x, dim=-1)

        return logits

class GPT(Config):
    def __init__(self):
        super().__init__()

        self.pre_processing = PreProcessing()                  
        self.blocks = [Block() for _ in range(self.n_layers)]
        self.post_processing = PostProcessing()
        
        self.n_params = self.pre_processing.n_params + Block().n_params + self.post_processing.n_params

    def __call__(self, tokens):
        x = self.pre_processing(tokens) # (B, T) -> (B, T, C)

        for block in self.blocks:
            x = block(x) # (B, T, C)

        logits = self.post_processing(x) # (B, T, V)
       
        return logits

model = GPT()

n_params = model.n_params

print(f'Number of parameters: {n_params / 1e6:.2f}M')

B, T = 4, 32
tokens = np.random.randint(low=0, high=2 ** 15 - 1, size=(B, T), dtype=np.int16)
model(tokens)
