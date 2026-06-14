import torch
import torch.nn as nn


class selfTransformer(nn.Module):
    def __init__(self, d_model=128*3):
        super().__init__()
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output


if __name__ == "__main__":
    input_tensor = torch.randn(2, 128 * 3)
    attention = selfTransformer()
    output = attention(input_tensor.unsqueeze(1))
    print(output.squeeze(1).shape)