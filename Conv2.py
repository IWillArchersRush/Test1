import torch
import torch.nn as nn

input = torch.randn(1, 1, 8, 8)


conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)

output = conv(input)

print("Input shape :", input.shape)   # Kết quả [1, 1, 8, 8]
print("Output shape:", output.shape)  # Kết quả [1, 1, 4, 4]
