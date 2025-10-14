import numpy as np
import struct

def load_mnist_images(filename):
    with open(filename, 'rb') as f:   #ảnh MNIST là ảnh BInary nên cần read binary, ảnh là ma trận (28x28) với giá trị bit từ 0 đến 255 
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16)) #Đọc 16 bit đầu tiên là header từ đấy biết được magic là loại file, num là số lg ảnh, rows, col là số dòng vs cột
        data = np.frombuffer(f.read(), dtype=np.uint8)  #Đọc toàn bộ dữ liệu còn lại sau khi bỏ qua header rồi biến thảnh mảng 1 chiều
        data = data.reshape(num, rows, cols) #Biến vecto 1D thành 3D
        return data.astype(np.float32) / 255.0  # chuẩn hóa về [0,1] phục vụ quá trình train



def load_mnist_labels(filename, one_hot=True):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)

        if one_hot:
            num_classes = 10
            # chuyển nhãn (0–9) thành vector one-hot
            data = np.eye(num_classes)[data]

        return data

X_train = load_mnist_images("MNIST/raw/train-images-idx3-ubyte")
y_train = load_mnist_labels("MNIST/raw/train-labels-idx1-ubyte", one_hot=True)
X_test  = load_mnist_images("MNIST/raw/t10k-images-idx3-ubyte")
y_test  = load_mnist_labels("MNIST/raw/t10k-labels-idx1-ubyte", one_hot=True)
print("Training set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

X_train = X_train[:, None, :, :]   # (N, 1, 28, 28)
X_test  = X_test[:, None, :, :]

# ================== LAYERS ==================
class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    def backward(self, dout):
        return dout * self.mask

class Flatten:
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)
    def backward(self, dout):
        return dout.reshape(self.x_shape)

class Dense:
    def __init__(self, in_dim, out_dim):
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros((1, out_dim))
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, dout, lr=0.01):
        dW = self.x.T @ dout
        db = np.sum(dout, axis=0, keepdims=True)
        dx = dout @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx

class SoftmaxCrossEntropy:
    def forward(self, logits, y_true):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        self.probs = probs
        self.y_true = y_true
        loss = -np.sum(y_true * np.log(probs + 1e-12)) / logits.shape[0]
        return loss
    def backward(self):
        return (self.probs - self.y_true) / self.y_true.shape[0]

def accuracy(logits, y_true):
    preds = np.argmax(logits, axis=1)
    true = np.argmax(y_true, axis=1)
    return np.mean(preds == true)




def im2col(x, KH, KW, stride=1, padding=0):
    N, C, H, W = x.shape
    out_h = (H + 2*padding - KH) // stride + 1
    out_w = (W + 2*padding - KW) // stride + 1

    x_padded = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    cols = np.zeros((N, C, KH, KW, out_h, out_w))

    for y in range(KH):
        y_max = y + stride*out_h
        for x_ in range(KW):
            x_max = x_ + stride*out_w
            cols[:, :, y, x_, :, :] = x_padded[:, :, y:y_max:stride, x_:x_max:stride]

    cols = cols.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)
    return cols



class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.b = np.zeros((out_channels, 1))

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        F, _, KH, KW = self.W.shape
        self.out_h = (H + 2*self.padding - KH)//self.stride + 1
        self.out_w = (W + 2*self.padding - KW)//self.stride + 1

        self.cols = im2col(x, KH, KW, self.stride, self.padding)   # (N*out_h*out_w, C*KH*KW)
        W_col = self.W.reshape(F, -1).T                           # (C*KH*KW, F)

        out = self.cols @ W_col + self.b.ravel()                  # (N*out_h*out_w, F)
        out = out.reshape(N, self.out_h, self.out_w, F).transpose(0,3,1,2)
        return out

    def backward(self, dout, lr=0.01):
        N, C, H, W = self.x.shape
        F, _, KH, KW = self.W.shape

        #dout: gradient của loss theo output (N, F, out_h, out_w)

        dout_reshaped = dout.transpose(0,2,3,1).reshape(-1, F)    # (N*out_h*out_w, F)
        dW = self.cols.T @ dout_reshaped                         # (C*KH*KW, F)
        dW = dW.transpose(1,0).reshape(F, C, KH, KW)

        db = np.sum(dout_reshaped, axis=0).reshape(F,1)

        W_col = self.W.reshape(F, -1)
        dcols = dout_reshaped @ W_col                            # (N*out_h*out_w, C*KH*KW)

        dx = np.zeros((N, C, H + 2*self.padding, W + 2*self.padding))
        cols_reshaped = dcols.reshape(N, self.out_h, self.out_w, C, KH, KW).transpose(0,3,4,5,1,2)

        for y in range(KH):
            y_max = y + self.stride*self.out_h
            for x_ in range(KW):
                x_max = x_ + self.stride*self.out_w
                dx[:, :, y:y_max:self.stride, x_:x_max:self.stride] += cols_reshaped[:, :, y, x_, :, :]

        dx = dx[:, :, self.padding:H+self.padding, self.padding:W+self.padding]

        self.W -= lr * dW
        self.b -= lr * db
        return dx


conv1 = Conv2D(1, 8, 3)   # 1 input channel -> 8 filters
relu1 = ReLU()
flatten = Flatten()
fc1 = Dense(8*26*26, 128)
relu2 = ReLU()
fc2 = Dense(128, 10)
loss_fn = SoftmaxCrossEntropy()


lr = 0.01
batch_size = 64
epochs = 100

for epoch in range(epochs):
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    for i in range(0, len(X_train), batch_size):
        x_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # forward
        out = conv1.forward(x_batch)
        out = relu1.forward(out)
        out = flatten.forward(out)
        out = fc1.forward(out)
        out = relu2.forward(out)
        logits = fc2.forward(out)

        # loss
        loss = loss_fn.forward(logits, y_batch)

        # backward
        dout = loss_fn.backward()
        dout = fc2.backward(dout, lr)
        dout = relu2.backward(dout)
        dout = fc1.backward(dout, lr)
        dout = flatten.backward(dout)
        dout = relu1.backward(dout)
        dout = conv1.backward(dout, lr)

    # evaluate
    out = conv1.forward(X_test[:1000]) 
    out = relu1.forward(out)
    out = flatten.forward(out)
    out = fc1.forward(out)
    out = relu2.forward(out)
    logits = fc2.forward(out)

    acc = accuracy(logits, y_test[:1000])
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
