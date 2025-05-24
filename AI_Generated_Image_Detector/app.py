import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import PIL.Image
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from preprocess import azi_diff

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, rate=0.2):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(rate)
        self.layer_norm1 = nn.LayerNorm(input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(rate),
            nn.Linear(ff_dim, input_dim),
            nn.Dropout(rate)
        )
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(attn_output + x)

        ffn_output = self.ffn(out1)
        out2 = self.layer_norm2(ffn_output + out1)
        return out2

class TextureContrastClassifier(nn.Module):
    def __init__(self, input_shape, num_heads=4, key_dim=64, ff_dim=256, rate=0.1):
        super(TextureContrastClassifier, self).__init__()
        input_dim = input_shape[1]
        self.rich_attention_block = AttentionBlock(input_dim, num_heads, ff_dim, rate)
        self.rich_dense = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.poor_attention_block = AttentionBlock(input_dim, num_heads, ff_dim, rate)
        self.poor_dense = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * input_shape[0], 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, rich_texture, poor_texture):
        rich_texture = rich_texture.permute(1, 0, 2)
        poor_texture = poor_texture.permute(1, 0, 2)
        rich_attention = self.rich_attention_block(rich_texture)
        rich_attention = rich_attention.permute(1, 0, 2)
        rich_features = self.rich_dense(rich_attention)
        poor_attention = self.poor_attention_block(poor_texture)
        poor_attention = poor_attention.permute(1, 0, 2)
        poor_features = self.poor_dense(poor_attention)
        difference = rich_features - poor_features
        difference = difference.view(difference.size(0), -1)
        output = self.fc(difference)
        return output

input_shape = (128, 256)
model = TextureContrastClassifier(input_shape)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

def inference(image, model):
    predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tmp = azi_diff(image, patch_num=128, N=256)
    rich = tmp["total_emb"][0]
    poor = tmp["total_emb"][1]
    rich_texture_tensor = torch.tensor(rich, dtype=torch.float32).unsqueeze(0).to(device)
    poor_texture_tensor = torch.tensor(poor, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(rich_texture_tensor, poor_texture_tensor)
    prediction = output.cpu().numpy().flatten()[0]
    return prediction

# Gradio Interface
def predict(image):
    prediction = inference(image, model)
    if prediction < 0.5:
      return False
    else:
      return True

# gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text").launch()
