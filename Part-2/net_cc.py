import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sentence_transformers import SentenceTransformer
from htm_pytorch import HTMBlock
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, ViTModel
import ast

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
class CFG:
    data_path = "/Users/shiva/IBW/data/tasks/R2R/data/R2R_train.json"

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = F.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)

class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )
class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )

class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))

class attention_Layer(nn.Module):
    def __init__(
        self,
        dim_model: int = 768,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, q: Tensor,k:Tensor,v:Tensor) -> Tensor:
        src = self.attention(q, k, v)
        return src

class language_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer("johngiorgi/declutr-small")
    
    def forward(self, texts):
        with torch.no_grad():
            embeddings = self.model.encode(texts)
        return embeddings

class image_encoder(nn.Module):
    def __init__(self):
        super().__init__()
       # self.image_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = self.model.eval()
    
    def forward(self, image) -> Tensor:
        with torch.no_grad():
            outputs = self.model(image)
        return outputs

class SHVNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_encoder = language_encoder()
        self.image_encoder = image_encoder()
        self.cross_attention_1 = attention_Layer()
        self.HCAM_1 = HTMBlock(
                    dim = 768,
                    heads = 4, 
                    topk_mems = 4,
                    mem_chunk_size = 32
                    )

        self.cross_attention_2 = attention_Layer()
        self.HCAM_2 =  HTMBlock(
                    dim = 768,
                    heads = 4, 
                    topk_mems = 4,
                    mem_chunk_size = 32
                    )
        #self.MLP = nn.Linear(768, 256)
    
    def forward(self, image, text, memories) -> Tensor:
        image_features = self.image_encoder(image)
        image_features = image_features.last_hidden_state[:, 0, :]
        image_features = image_features.unsqueeze(1)
        image_features = image_features.to(device)
        text_features = self.language_encoder(text)
        text_features = torch.from_numpy(text_features).unsqueeze(1)
        text_features = text_features.to(device)
        x = self.cross_attention_1(text_features, image_features, image_features)
        x = self.HCAM_1(x, memories)
        x = self.cross_attention_2(x, text_features, text_features)
        x = self.HCAM_2(x, memoriese)
        #x = self.MLP(x)
        return x