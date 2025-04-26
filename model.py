import numpy as np

#! hyperparameters
class Config:
    vocab_size = 32768
    ctx_length = 512
    n_layers = 4
    d_model = 128

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
def softmax(x):
    B, T, C = x.shape

    new_x = np.zeros_like(x)

    for i in range(B):
        for j in range(T):
            x_max = np.max(x[i, j])
        
            exp_x = np.zeros(C)

            for k in range(C):
                exp_x[k] = np.exp(x[i, j, k] - x_max)
       
            exp_sum = np.sum(exp_x)

            for k in range(C):
                new_x[i, j, k] = exp_x[k] / exp_sum
    
    return new_x

def gelu(x):
    return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

#! model
class PreProcessing(Config):
    def __init__(self):
        self.embedding = Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = Embedding(self.ctx_length, self.d_model)

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

    def __call__(self, x):
        x = x + self.causal_sa(self.ln_1(x))
        x = x + self.fnn(self.ln_1(x))

        return x

class CausalSelfAttention:
    def __init__(self):
        pass

    def __call__(self, x):
        return x

class FNN(Config):
    def __init__(self):
        self.ll_1 = Linear(self.d_model, 4 * self.d_model)
        self.ll_2 = Linear(4 * self.d_model, self.d_model)

    def __call__(self, x):
        x = self.ll_1(x)
        x = gelu(x)
        x = self.ll_2(x)

        return x

class PostProcessing(Config):
    def __init__(self):
        self.ll = Linear(self.d_model, self.vocab_size)
        self.ln = LayerNorm(self.d_model)

    def __call__(self, x):
        x = self.ln(x) # (B, T, C)
        x = self.ll(x) # (B, T, V)
        logits = softmax(x)

        return logits

class GPT(Config):
    def __init__(self):
        super().__init__()

        self.pre_processing = PreProcessing()                  
        self.blocks = [Block() for _ in range(self.n_layers)]
        self.post_processing = PostProcessing()

    def __call__(self, tokens):
        x = self.pre_processing(tokens) # (B, T) -> (B, T, C)

        for block in self.blocks:
            x = block(x) # (B, T, C)

        logits = self.post_processing(x) # (B, T, V)
       
        return logits

model = GPT()

B, T = 4, 32
tokens = np.ones((B, T), dtype=np.int16)
print(model(tokens))
