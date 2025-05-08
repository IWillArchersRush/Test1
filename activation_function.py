import torch

# Tạo dữ liệu đầu vào cho hàm sigmoid
x = torch.linspace(-10, 10, 100)

# Áp dụng hàm sigmoid
sigmoid_x = torch.sigmoid(x)

# In ra một vài giá trị để kiểm tra
for i in range(0, len(x), 10):  # In mỗi 10 giá trị
    print(f"Input: {x[i]:.2f}, Sigmoid Output: {sigmoid_x[i]:.4f}")
