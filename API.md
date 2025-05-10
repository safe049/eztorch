

## 📘 类: NeuralNet

> 

### 方法列表

#### 🔧 __init__(*layers: any)

```python
def __init__(*layers: any)
```



#### 🔧 build(input_shape: any)

```python
def build(input_shape: any)
```



#### 🔧 compile(optimizer: any, lr: any, loss: any, device: any)

```python
def compile(optimizer: any, lr: any, loss: any, device: any)
```



#### 🔧 fit(dataset: any, epochs: any, batch_size: any, verbose: any, device: any)

```python
def fit(dataset: any, epochs: any, batch_size: any, verbose: any, device: any)
```



#### 🔧 plot(无参数)

```python
def plot(无参数)
```



#### 🔧 save(path: any)

```python
def save(path: any)
```



#### 🔧 load(path: any)

```python
def load(path: any)
```

从文件加载模型状态字典

#### 🔧 load_if_exists(path: any)

```python
def load_if_exists(path: any)
```



#### 🔧 RL(cls: any, env_name: any, policy: any, hidden_dim: any)

```python
def RL(cls: any, env_name: any, policy: any, hidden_dim: any)
```



#### 🔧 GAN(cls: any, latent_dim: any, img_shape: any, generator: any, discriminator: any)

```python
def GAN(cls: any, latent_dim: any, img_shape: any, generator: any, discriminator: any)
```



#### 🔧 device(无参数)

```python
def device(无参数)
```

返回模型所在的设备

## 📘 类: Layer

> 

### 方法列表

#### 🔧 __init__(layer_type: any, *args: any, **kwargs: any)

```python
def __init__(layer_type: any, *args: any, **kwargs: any)
```



## 📘 类: Flatten

> 

### 方法列表

#### 🔧 forward(x: any)

```python
def forward(x: any)
```



## 📘 类: Trainer

> 

### 方法列表

#### 🔧 __init__(model: any, optimizer: any, lr: any, loss: any, device: any)

```python
def __init__(model: any, optimizer: any, lr: any, loss: any, device: any)
```



#### 🔧 fit(dataset: any, epochs: any, batch_size: any, verbose: any, device: any)

```python
def fit(dataset: any, epochs: any, batch_size: any, verbose: any, device: any)
```

训练模型，支持指定设备、数据集、轮数和批量大小。

#### 🔧 plot_loss(无参数)

```python
def plot_loss(无参数)
```



## 🔧 函数: get_dataloader

```python
def get_dataloader(name: any, batch_size: any, root: any, download: any)
```

> 

**参数:**
- `name`: any
- `batch_size`: any
- `root`: any
- `download`: any

**返回值:** `None`

## 📘 类: RLModel

> 

### 方法列表

#### 🔧 __init__(env_name: any, policy: any, hidden_dim: any)

```python
def __init__(env_name: any, policy: any, hidden_dim: any)
```



#### 🔧 remember(state: any, action: any, reward: any, next_state: any, done: any)

```python
def remember(state: any, action: any, reward: any, next_state: any, done: any)
```



#### 🔧 act(state: any)

```python
def act(state: any)
```



#### 🔧 replay(batch_size: any)

```python
def replay(batch_size: any)
```



#### 🔧 train(episodes: any)

```python
def train(episodes: any)
```



## 📘 类: Generator

> 

### 方法列表

#### 🔧 __init__(latent_dim: any, img_shape: any)

```python
def __init__(latent_dim: any, img_shape: any)
```



#### 🔧 forward(z: any)

```python
def forward(z: any)
```



## 📘 类: Discriminator

> 

### 方法列表

#### 🔧 __init__(img_shape: any)

```python
def __init__(img_shape: any)
```



#### 🔧 forward(img: any)

```python
def forward(img: any)
```



## 📘 类: GANModel

> 

### 方法列表

#### 🔧 __init__(latent_dim: any, img_shape: any, generator: any, discriminator: any)

```python
def __init__(latent_dim: any, img_shape: any, generator: any, discriminator: any)
```



#### 🔧 train_step(real_images: any)

```python
def train_step(real_images: any)
```



## 🔧 函数: auto_infer

```python
def auto_infer(model: any, input_shape: any)
```

> 

**参数:**
- `model`: any
- `input_shape`: any

**返回值:** `None`

## 🔧 函数: register_activation_hook

```python
def register_activation_hook(model: any)
```

> 注册钩子以获取每层的输出

**参数:**
- `model`: any

**返回值:** `None`

## 🔧 函数: plot_activations

```python
def plot_activations(activations: any)
```

> 

**参数:**
- `activations`: any

**返回值:** `None`

## 📘 类: Config

> 

### 方法列表

#### 🔧 __init__(无参数)

```python
def __init__(无参数)
```



#### 🔧 _get_device(无参数)

```python
def _get_device(无参数)
```



#### 🔧 update(**kwargs: any)

```python
def update(**kwargs: any)
```

动态更新配置项

#### 🔧 show(无参数)

```python
def show(无参数)
```

显示当前配置信息

## 📘 类: MLP

> 

### 方法列表

#### 🔧 __init__(无参数)

```python
def __init__(无参数)
```



#### 🔧 forward(x: any)

```python
def forward(x: any)
```



## 🔧 函数: get_dataloader

```python
def get_dataloader(name: any, batch_size: any, root: any, download: any)
```

> 

**参数:**
- `name`: any
- `batch_size`: any
- `root`: any
- `download`: any

**返回值:** `None`

## 📘 类: Trainer

> 

### 方法列表

#### 🔧 __init__(model: any, optimizer: any, lr: any, loss: any)

```python
def __init__(model: any, optimizer: any, lr: any, loss: any)
```



#### 🔧 fit(dataset: any, epochs: any, batch_size: any, verbose: any)

```python
def fit(dataset: any, epochs: any, batch_size: any, verbose: any)
```



#### 🔧 plot_loss(无参数)

```python
def plot_loss(无参数)
```

