import pandas as pd
import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
from torch import nn, Tensor
class image_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = self.model
        self.model = self.model.eval()
    
    def forward(self, image) -> Tensor:
        with torch.no_grad():
            inputs = self.image_processor(image, return_tensors="pt")
            print(type(inputs))
            print(inputs["pixel_values"].shape)
            outputs = self.model(**inputs)
        return outputs

image1 = Image.open("/users/shiva/downloads/3.png")
image2 = Image.open("/users/shiva/downloads/b8cTxDM8gDG_2.png")

model = image_encoder()
out1 = model(image1)
out2 = model(image2)
out1 = out1.last_hidden_state[0, 0, :]
out2 = out2.last_hidden_state[0, 0, :]
print(out1.shape)
# cos = nn.CosineSimilarity(dim=0, eps=1e-6)
# output = cos(out1, out2)
# print(output)