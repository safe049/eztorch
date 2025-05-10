from ntorch import NeuralNet

# 创建 GAN 模型
model = NeuralNet.GAN(latent_dim=100, img_shape=(3, 64, 64))

print("Generator:", model.generator)
print("Discriminator:", model.discriminator)