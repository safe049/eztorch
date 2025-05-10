from ntorch import NeuralNet

# 创建 DQN 模型
model = NeuralNet.RL(env_name='CartPole-v1', policy='MLP')

# 开始训练
model.train(episodes=200)