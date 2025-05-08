import torch
import torch.nn as nn

# Tạo tensor đầu vào 4x4
input_tensor = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0],
                               [9.0,10.0,11.0,12.0],
                               [13.0,14.0,15.0,16.0]]]])

print("Input tensor:")
print(input_tensor)

# Tạo lớp MaxPool2d với kernel_size=2, stride=2
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)


output_tensor = max_pool(input_tensor)

print("\nOutput tensor after MaxPool2d:")
print(output_tensor)
