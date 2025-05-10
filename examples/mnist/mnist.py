from ntorch.model import NeuralNet
from ntorch.layers import Layer
import torch


print("building MLP ")
model = NeuralNet(
    Layer('Flatten'),           # 将输入图像展平成一维向量
    Layer('Linear', 28 * 28, 512),
    Layer('ReLU'),
    Layer('Linear', 512, 256),
    Layer('ReLU'),
    Layer('Linear', 256, 10)
)
print("MLP network built.")

print("inferring model structure...")
model.build(input_shape=(1, 28, 28))
print("model structure inferred.")


print("compiling model...")
model.compile(optimizer='Adam', lr=0.001, loss='CrossEntropyLoss',device='cuda')
print("model compiled.")


device = 'cuda'
print(f"Training on device: {device}")


print("training...")
model.fit(dataset='mnist', epochs=5, batch_size=64,device = device)
print("training finished.")


model.save("mnist_mlp_model.pth")

# 加载模型
# model = NeuralNet(...)  # 保持结构一致
# model.build(input_shape=(1, 28, 28))
# model.load("mnist_mlp_model.pth")


from ntorch.data import get_dataloader

# 使用已有的数据加载器
_, test_loader = get_dataloader('mnist', batch_size=1)
images, labels = next(iter(test_loader))
image, label = images[0], labels[0].item()

# 推理
with torch.no_grad():
    output = model(image.unsqueeze(0).to(model.device))
    predicted = torch.argmax(output, dim=1).item()

print(f"真实标签: {label}, 预测标签: {predicted}")