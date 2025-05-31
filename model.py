# Models defined here

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# Note: every component that is tagged as essential is very important for obtaining an available model
# Reimplementation of Multi-head self attention
class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, drop_out):
        super(CausalSelfAttention, self).__init__()
        assert embedding_size % num_heads == 0
        head_size = embedding_size // num_heads
        self.embedding_size = embedding_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.drop_out = drop_out
        # define model parameters
        self.q_proj = nn.Linear(embedding_size, embedding_size)
        self.k_proj = nn.Linear(embedding_size, embedding_size)
        self.v_proj = nn.Linear(embedding_size, embedding_size)
        self.out_proj = nn.Linear(embedding_size, embedding_size)
        self.attn_dropout = nn.Dropout(drop_out)
        self.resid_dropout = nn.Dropout(drop_out)
        # initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x, attn_mask):
        if len(x.size()) == 3:
            # x dim:(batch_size, seq_len, embedding_size)
            batch_size, seq_len, embedding_size = x.size()
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
            # q, k, v dim: (batch_size, num_heads, seq_len, head_size)

            attn_weights = torch.matmul(q, k.transpose(-1, -2))/math.sqrt(self.head_size)
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            # attn_weights dim: (batch_size, num_heads, seq_len, seq_len)
            attn_outputs = torch.matmul(attn_weights, v)
            # attn_outputs dim: (batch_size, num_heads, seq_len, head_size)
            attn_outputs = attn_outputs.transpose(1, 2).\
                contiguous().view(batch_size, seq_len, embedding_size)
            # attn_outputs dim: (batch_size, seq_len, embedding_size)
            attn_outputs = self.resid_dropout(self.out_proj(attn_outputs))
            return attn_outputs
        if len(x.size()) == 2:
            seq_len, embedding_size = x.size()
            q = self.q_proj(x).view(seq_len, self.num_heads, self.head_size).transpose(0, 1)
            k = self.k_proj(x).view(seq_len, self.num_heads, self.head_size).transpose(0, 1)
            v = self.v_proj(x).view(seq_len, self.num_heads, self.head_size).transpose(0, 1)
            # q, k, v dim: (num_heads, seq_len, head_size)
            # The dimension of qkv is lowered to reduce computation cost and capture local feature
            attn_weights = torch.matmul(q, k.transpose(-1, -2))/math.sqrt(self.head_size)
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            # attn_weights dim: (num_heads, seq_len, seq_len)
            attn_outputs = torch.matmul(attn_weights, v)
            # attn_outputs dim: (num_heads, seq_len, head_size)
            attn_outputs = attn_outputs.transpose(0, 1).\
                contiguous().view(seq_len, embedding_size)
            # attn_outputs dim: (seq_len, embedding_size)
            attn_outputs = self.resid_dropout(self.out_proj(attn_outputs))
            return attn_outputs


class DecoderLayer(nn.Module):
    # The GPT DecoderLayer, which has no encoder input
    def __init__(self, embed_size, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.MultiheadAttention = CausalSelfAttention(embed_size, num_heads, dropout)
        self.dropout = dropout
        self.FeedForwardNet = nn.Sequential(
            nn.Linear(embed_size, 4*embed_size),
            nn.GELU(),
            nn.Linear(4*embed_size, embed_size)
        )
        self.FFN_Dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, dec_inputs, attn_mask=None):
        # dec_inputs dim: (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
        norm_inputs = self.norm1(dec_inputs)
        attn_outputs = self.MultiheadAttention(norm_inputs, attn_mask=attn_mask)
        dec_inputs = attn_outputs + dec_inputs  # note: do not use += here. It's an inplace operation
        ff_outputs = self.FFN_Dropout(self.FeedForwardNet(self.norm2(dec_inputs)))
        return dec_inputs + ff_outputs
        # You should correctly use residual connection and Layernorm. When passing by a module
        # you should use the normalized input as the module's input and add its output to your original input


class SimpleGPT(nn.Module):
    # A simple GPT-like model
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, dropout,
                 device, max_seqlen=512):
        super(SimpleGPT, self).__init__()
        self.embed_word = nn.Embedding(vocab_size, embed_size).to(device)  # token embedding
        self.embed_pos = nn.Embedding(max_seqlen, embed_size).to(device)   # position embedding
        self.DecoderLayers = nn.ModuleList(
            [DecoderLayer(embed_size=embed_size, num_heads=num_heads, dropout=dropout) for N in range(num_layers)]
        ).to(device)
        # self.decoder_norm2 = nn.LayerNorm(embed_size).to(device)
        # self.out1 = nn.Linear(embed_size, embed_size).to(device)
        # ).to(device)
        self.device = device
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)

    @staticmethod
    def get_attn_mask(length):
        return nn.Transformer.generate_square_subsequent_mask(length)

    def forward(self, src):
        seq_len = src.size(-1)
        # input dim: (batch_size, seq_len)
        mem_embed1 = self.embed_word(src)
        mem_embed2 = self.embed_pos(torch.flip(torch.arange(seq_len, device=self.device), dims=[-1]))
        # flip the position sequence. make the last token the first.
        x = mem_embed1+mem_embed2
        attn_mask = self.get_attn_mask(seq_len).to(self.device)  # attention mask is essential, as causality requires
        # mask = self.get_attn_mask(src.size()[1])
        for mod in self.DecoderLayers:
            x = mod(x, attn_mask)
        # self.decoder memory input and output dimension: (batch_size, seq_len, embed_size) or (seq_len, embed_size)
        return x


class SimpleGPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, dropout,
                 device, max_seqlen=1024):
        super(SimpleGPTLanguageModel, self).__init__()
        self.subSimpleGPT = SimpleGPT(vocab_size, embed_size, num_layers,
                                      num_heads, dropout, device, max_seqlen)
        self.LM_head = nn.Linear(embed_size, vocab_size).to(device)
        self.LM_head_norm = nn.LayerNorm(embed_size).to(device)
        self.vocab_size = vocab_size
        self.device = device

    def forward(self, src):
        # default inference mode
        assert (len(src.size()) == 1 or len(src.size()) == 2)
        x = self.subSimpleGPT(src)
        x = self.LM_head_norm(x)
        if len(x.size()) == 3:
            return self.LM_head(x)[:, -1, :]
        elif len(x.size()) == 2:
            return self.LM_head(x)[-1, :]

    def forward_no_inference(self, src):
        # This function gives a prediction sequence. Should be used in training
        assert (len(src.size()) == 1 or len(src.size()) == 2)
        x = self.subSimpleGPT(src)
        x = self.LM_head_norm(x)
        return self.LM_head(x)


class SimpleGPTLanguageModelForPFinetune(nn.Module):
    # This model is used for finetune. adds a few new layers and frozen the parameters of pretrained models
    def __init__(self, pretrained_path, num_of_sft_layers, vocab_size, embed_size, num_of_heads, dropout, device):
        super(SimpleGPTLanguageModelForPFinetune, self).__init__()
        self.pretrained_model = torch.load(pretrained_path).to(device)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.lm_norm = nn.LayerNorm(embed_size).to(device)
        self.sft_layers = nn.ModuleList([DecoderLayer(
            embed_size=embed_size, num_heads=num_of_heads, dropout=dropout)
            for _ in range(num_of_sft_layers)]).to(device)
        self.lm_out = nn.Linear(embed_size, vocab_size).to(device)
        self.device = device
        self.vocab_size = vocab_size

    def forward_no_inference(self, src):
        seq_len = src.size(-1)
        # This function gives a prediction sequence. Should be used in training
        assert (len(src.size()) == 1 or len(src.size()) == 2)
        x = self.pretrained_model.subSimpleGPT(src)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(self.device)
        for sft_layer in self.sft_layers:
            x = sft_layer(x, attn_mask)
        x = self.lm_norm(x)
        return self.lm_out(x)

    def forward(self, src):
        assert (len(src.size()) == 1 or len(src.size()) == 2)
        if len(src.size()) == 2:
            return self.forward_no_inference(src)[:, -1, :]
        elif len(src.size()) == 1:
            return self.forward_no_inference(src)[-1, :]


# The following code redefines classes with RoPe embedding
# llama's RoPe code
class RotaryPositionEmbedding:
    def __init__(self, device):
        self.device = device

    def compute_frequency(self, embed_dim: int, seq_len: int, theta=10000.0):
        # freq = 10000^(-2i/d), d the embed_dim, i the dimensional index
        freqs = 1.0/(theta**(torch.arange(0, embed_dim, 2).to(self.device)[: embed_dim//2]/embed_dim))
        pos = torch.arange(seq_len).to(self.device)
        # torch.out: x = torch.tensor([i1, i2, i3]), torch.out(x, x) -> torch.tensor([[i1*i1, i1*i2, i1*i3],
        # [i2*i1, i2*i2, i2*i3] ...])
        freqs = torch.outer(pos, freqs).float()
        # torch.polar: torch.polar(a, theta) -> a*exp(j*theta)
        freqs_exp = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_exp

    def __call__(self, q: torch.Tensor, k: torch.Tensor):
        embed_size = q.size(-1)
        seq_len = q.size(-2)

        freqs_exp = self.compute_frequency(embed_size, seq_len).to(q.device)
        q_ = q.float().reshape(*q.shape[:-1], -1, 2)
        k_ = k.float().reshape(*k.shape[:-1], -1, 2)
        # (x, y) -> (x + iy)
        q_ = torch.view_as_complex(q_)
        k_ = torch.view_as_complex(k_)
        # torch.flatten: flatten the dimensions after the start_dim
        q_out = torch.view_as_real(q_*freqs_exp).flatten(-2)
        k_out = torch.view_as_real(k_*freqs_exp).flatten(-2)

        return q_out, k_out


class RotaryCausalSelfAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, drop_out):
        super(RotaryCausalSelfAttention, self).__init__()
        assert embedding_size % num_heads == 0
        head_size = embedding_size // num_heads
        self.embedding_size = embedding_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.drop_out = drop_out
        # define model parameters
        self.q_proj = nn.Linear(embedding_size, embedding_size)
        self.k_proj = nn.Linear(embedding_size, embedding_size)
        self.v_proj = nn.Linear(embedding_size, embedding_size)
        self.out_proj = nn.Linear(embedding_size, embedding_size)
        self.attn_dropout = nn.Dropout(drop_out)
        self.resid_dropout = nn.Dropout(drop_out)
        self.RoPe_func = RotaryPositionEmbedding(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x, attn_mask):
        if len(x.size()) == 3:
            # x dim:(batch_size, seq_len, embedding_size)
            batch_size, seq_len, embedding_size = x.size()
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
            # q, k, v dim: (batch_size, num_heads, seq_len, head_size)

            # apply rotary position embedding
            q, k = self.RoPe_func(q, k)

            attn_weights = torch.matmul(q, k.transpose(-1, -2))/math.sqrt(self.head_size)
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            # attn_weights dim: (batch_size, num_heads, seq_len, seq_len)
            attn_outputs = torch.matmul(attn_weights, v)
            # attn_outputs dim: (batch_size, num_heads, seq_len, head_size)
            attn_outputs = attn_outputs.transpose(1, 2).\
                contiguous().view(batch_size, seq_len, embedding_size)
            # attn_outputs dim: (batch_size, seq_len, embedding_size)
            attn_outputs = self.resid_dropout(self.out_proj(attn_outputs))
            return attn_outputs
        if len(x.size()) == 2:
            seq_len, embedding_size = x.size()
            q = self.q_proj(x).view(seq_len, self.num_heads, self.head_size).transpose(0, 1)
            k = self.k_proj(x).view(seq_len, self.num_heads, self.head_size).transpose(0, 1)
            v = self.v_proj(x).view(seq_len, self.num_heads, self.head_size).transpose(0, 1)
            # q, k, v dim: (num_heads, seq_len, head_size)
            # The dimension of qkv is lowered to reduce computation cost and capture local feature

            # apply rotary position embedding
            q, k = self.RoPe_func(q, k)

            attn_weights = torch.matmul(q, k.transpose(-1, -2))/math.sqrt(self.head_size)
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            # attn_weights dim: (num_heads, seq_len, seq_len)
            attn_outputs = torch.matmul(attn_weights, v)
            # attn_outputs dim: (num_heads, seq_len, head_size)
            attn_outputs = attn_outputs.transpose(0, 1).\
                contiguous().view(seq_len, embedding_size)
            # attn_outputs dim: (seq_len, embedding_size)
            attn_outputs = self.resid_dropout(self.out_proj(attn_outputs))
            return attn_outputs


# SwiGlu activation, It seems this activation has a very good effect, the losses have reduced a lot
class SwiGluFeedForward(nn.Module):
    def __init__(self, embed_size):
        super(SwiGluFeedForward, self).__init__()
        self.embed_size = embed_size
        self.w1 = nn.Linear(embed_size, 4*embed_size)
        self.w2 = nn.Linear(embed_size, 4*embed_size)
        self.out_proj = nn.Linear(4*embed_size, embed_size, bias=False)

    def forward(self, x):
        return self.out_proj(F.silu(self.w1(x))*self.w2(x))


class DecoderLayerWithRotaryEmbedding(nn.Module):
    # The GPT DecoderLayer, which has no encoder input
    def __init__(self, embed_size, num_heads, dropout):
        super(DecoderLayerWithRotaryEmbedding, self).__init__()
        self.MultiheadAttention = RotaryCausalSelfAttention(embed_size, num_heads, dropout)
        self.dropout = dropout
        self.FeedForwardNet = SwiGluFeedForward(embed_size)
        self.FFN_Dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, dec_inputs, attn_mask=None):
        # dec_inputs dim: (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
        norm_inputs = self.norm1(dec_inputs)
        attn_outputs = self.MultiheadAttention(norm_inputs, attn_mask=attn_mask)
        dec_inputs = attn_outputs + dec_inputs  # note: do not use += here. It's an inplace operation
        ff_outputs = self.FFN_Dropout(self.FeedForwardNet(self.norm2(dec_inputs)))
        return dec_inputs + ff_outputs


class RotaryGPT(nn.Module):
    # A simple GPT-like model with RoPe embedding
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, dropout,
                 device):
        super(RotaryGPT, self).__init__()
        self.embed_word = nn.Embedding(vocab_size, embed_size).to(device)  # token embedding
        self.DecoderLayers = nn.ModuleList(
            [DecoderLayerWithRotaryEmbedding(embed_size=embed_size, num_heads=num_heads, dropout=dropout) for N in range(num_layers)]
        ).to(device)
        # self.decoder_norm2 = nn.LayerNorm(embed_size).to(device)
        # self.out1 = nn.Linear(embed_size, embed_size).to(device)
        # ).to(device)
        self.device = device
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)

    @staticmethod
    def get_attn_mask(length):
        return nn.Transformer.generate_square_subsequent_mask(length)

    def forward(self, src):
        seq_len = src.size(-1)
        # input dim: (batch_size, seq_len)
        mem_embed1 = self.embed_word(src)
        x = mem_embed1
        attn_mask = self.get_attn_mask(seq_len).to(self.device)  # attention mask is essential, as causality requires
        # mask = self.get_attn_mask(src.size()[1])
        for mod in self.DecoderLayers:
            x = mod(x, attn_mask)
        # self.decoder memory input and output dimension: (batch_size, seq_len, embed_size) or (seq_len, embed_size)
        return x


class RotaryGPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, dropout,
                 device):
        super(RotaryGPTLanguageModel, self).__init__()
        self.subSimpleGPT = RotaryGPT(vocab_size, embed_size, num_layers,
                                      num_heads, dropout, device)
        self.LM_head = nn.Linear(embed_size, vocab_size).to(device)
        self.LM_head_norm = nn.LayerNorm(embed_size).to(device)
        self.vocab_size = vocab_size
        self.device = device

    def forward(self, src):
        # default inference mode
        assert (len(src.size()) == 1 or len(src.size()) == 2)
        x = self.subSimpleGPT(src)
        x = self.LM_head_norm(x)
        if len(x.size()) == 3:
            return self.LM_head(x)[:, -1, :]
        elif len(x.size()) == 2:
            return self.LM_head(x)[-1, :]

    def forward_no_inference(self, src):
        # This function gives a prediction sequence. Should be used in training
        assert (len(src.size()) == 1 or len(src.size()) == 2)
        x = self.subSimpleGPT(src)
        x = self.LM_head_norm(x)
        return self.LM_head(x)


class RotaryGPTLanguageModelForPFinetune(nn.Module):
    # This model is used for finetune. adds a few new layers and frozen the parameters of pretrained models
    def __init__(self, pretrained_path, num_of_sft_layers, vocab_size, embed_size, num_of_heads, dropout, device):
        super(RotaryGPTLanguageModelForPFinetune, self).__init__()
        self.pretrained_model = torch.load(pretrained_path).to(device)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.sft_layers = nn.ModuleList([DecoderLayer(
            embed_size=embed_size, num_heads=num_of_heads, dropout=dropout)
            for _ in range(num_of_sft_layers)]).to(device)
        self.device = device
        self.vocab_size = vocab_size
        self.lm_norm = nn.LayerNorm(embed_size).to(device)

    def forward_no_inference(self, src, in_inference=False):
        seq_len = src.size(-1)
        # This function gives a prediction sequence. Should be used in training
        assert (len(src.size()) == 1 or len(src.size()) == 2)
        x = self.pretrained_model.subSimpleGPT(src)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(self.device)
        for sft_layer in self.sft_layers:
            x = sft_layer(x, attn_mask)
        x = self.lm_norm(x)
        return self.pretrained_model.LM_head(x)

    def forward(self, src):
        assert (len(src.size()) == 1 or len(src.size()) == 2)
        if len(src.size()) == 2:
            return self.forward_no_inference(src)[:, -1, :]
        elif len(src.size()) == 1:
            return self.forward_no_inference(src)[-1, :]


# LoRA Linear, implements LoRA architecture for finetune
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, hidden_dim=8):
        super(LoRALinear, self).__init__()
        self.device = next(linear.parameters()).device
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.A = nn.Linear(self.in_features, hidden_dim).to(self.device)
        self.B = nn.Linear(hidden_dim, self.out_features, bias=False).to(self.device)
        # make B=0 at initialization, BAx=0
        nn.init.zeros_(self.B.weight)
        self.original_linear = linear
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.original_linear(x)+self.B(self.A(x))


# you can't directly set an attr of an object's child, use the following func to replace modules
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def get_lora_model(model: nn.Module, rank=8):
    for param in model.parameters():
        param.requires_grad = False
    # replace the linear layers
    for name, module in model.named_modules():
        # only q, v in attention and w1 in FFN will be modified
        if isinstance(module, nn.Linear) and ("q_proj" in name or "v_proj" in name or "w1" in name):
            lora_linear = LoRALinear(module, hidden_dim=rank)
            _set_module(model, name, lora_linear)
    return model


if __name__ == "__main__":
    pass
