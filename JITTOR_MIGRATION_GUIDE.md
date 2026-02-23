# ATAS PyTorch → Jittor 迁移完全指南

## 目录
1. [Jittor 简介](#1-jittor-简介)
2. [环境准备](#2-环境准备)
3. [API 对应关系速查表](#3-api-对应关系速查表)
4. [迁移策略与步骤](#4-迁移策略与步骤)
5. [逐文件迁移指南](#5-逐文件迁移指南)
6. [关键难点攻克](#6-关键难点攻克)
7. [验证与对齐](#7-验证与对齐)
8. [常见问题 FAQ](#8-常见问题-faq)

---

## 1. Jittor 简介

### 1.1 什么是 Jittor？
Jittor（计图）是清华大学开发的深度学习框架，特点是：
- **元算子融合**：自动优化算子执行效率
- **动态图 + JIT 编译**：结合动态图的灵活性和静态图的高效性
- **API 高度兼容 PyTorch**：大部分代码只需修改 import 即可运行

### 1.2 为什么选择 Jittor？
- 国产框架，适合科研和教学
- 性能接近甚至超越 PyTorch（在某些场景下）
- API 设计借鉴 PyTorch，学习成本低

---

## 2. 环境准备

### 2.1 安装 Jittor
```bash
# 方案 A：使用 pip 安装（推荐）
pip install jittor

# 方案 B：从源码编译（如果需要 CUDA 支持）
git clone https://github.com/Jittor/jittor.git
cd jittor
python setup.py install
```

### 2.2 验证安装
在 Python 中测试：
```python
import jittor as jt
print(jt.__version__)

# 检查 CUDA 是否可用
print(jt.flags.use_cuda)  # 如果输出 1，说明可以用 GPU
```

### 2.3 创建迁移工作目录
```bash
cd ATAS-main
mkdir ATAS-jittor  # 创建 Jittor 版本的代码目录
```

---

## 3. API 对应关系速查表

### 3.1 基础模块导入
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `import torch` | `import jittor as jt` | 核心库 |
| `import torch.nn as nn` | `import jittor.nn as nn` | 神经网络模块 |
| `import torch.optim as optim` | `import jittor.optim as optim` | 优化器 |
| `from torch.utils.data import DataLoader` | `from jittor.dataset import Dataset` | 数据加载（有差异） |

### 3.2 常用 API 对应
| 功能 | PyTorch | Jittor |
|------|---------|--------|
| 创建张量 | `torch.tensor([1,2,3])` | `jt.array([1,2,3])` |
| 随机张量 | `torch.rand(2,3)` | `jt.rand(2,3)` |
| 全零张量 | `torch.zeros(2,3)` | `jt.zeros(2,3)` |
| 设备转移 | `.cuda()` / `.to(device)` | ~~删除~~（Jittor 自动管理） |
| 梯度计算 | `torch.autograd.grad(...)` | `jt.grad(...)` |
| 反向传播 | `loss.backward()` | `optimizer.backward(loss)` |
| 无梯度上下文 | `with torch.no_grad():` | `with jt.no_grad():` |
| 启用梯度 | `with torch.enable_grad():` | ~~无需显式声明~~（默认启用） |

### 3.3 神经网络模块
| PyTorch | Jittor | 说明 |
|---------|--------|------|
| `nn.Conv2d` | `nn.Conv2d` | ✅ 完全一致 |
| `nn.BatchNorm2d` | `nn.BatchNorm2d` | ✅ 完全一致 |
| `nn.ReLU` | `nn.ReLU` | ✅ 完全一致 |
| `nn.CrossEntropyLoss` | `nn.CrossEntropyLoss` | ✅ 完全一致 |
| `nn.DataParallel(model)` | ~~删除~~（Jittor 自动并行） | - |

---

## 4. 迁移策略与步骤

### 4.1 迁移原则
1. **先易后难**：从最简单的文件开始（如模型定义），最后攻克复杂的（如梯度攻击）
2. **逐层验证**：每迁移一个文件，立即验证其输出是否与 PyTorch 版本一致
3. **保留原文件**：不要覆盖 PyTorch 代码，创建新的 `-jittor.py` 文件

### 4.2 迁移顺序（推荐）
```
Step 1: models/normalize.py       → 最简单，只有一个 Normalize 类
Step 2: models/preact_resnet.py   → 纯模型定义，无复杂逻辑
Step 3: data.py                   → 数据加载（需要适配 Jittor 的 Dataset）
Step 4: data_aug.py               → 数据增强（张量操作为主）
Step 5: adv_attack.py             → 🔥 核心难点：梯度计算
Step 6: ATAS.py                   → 主训练脚本，整合所有模块
Step 7: attack.py                 → 评估脚本
```

---

## 5. 逐文件迁移指南

### 5.1 迁移 `models/normalize.py`

#### 原 PyTorch 代码 (核心部分)
```python
import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def forward(self, x):
        return (x - self.mean) / self.std
```

#### 迁移后 Jittor 代码
```python
import jittor as jt
import jittor.nn as nn

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        # torch.tensor → jt.array
        self.mean = jt.array(mean).view(1, 3, 1, 1)
        self.std = jt.array(std).view(1, 3, 1, 1)
    
    def execute(self, x):  # forward → execute
        return (x - self.mean) / self.std
```

#### 关键改动
1. `torch.tensor` → `jt.array`
2. `forward` → `execute`（Jittor 中所有 Module 的前向传播方法叫 `execute`）

---

### 5.2 迁移 `models/preact_resnet.py`

#### 关键改动
1. **全局替换**：
   - `import torch` → `import jittor as jt`
   - `import torch.nn as nn` → `import jittor.nn as nn`
   - `import torch.nn.functional as F` → `import jittor.nn as F`（Jittor 把 functional 整合到 nn 里了）

2. **forward → execute**：所有的 `def forward(self, x):` 改为 `def execute(self, x):`

3. **无需其他改动**：ResNet 的卷积、BN、ReLU 等层在 Jittor 中 API 完全一致。

---

### 5.3 迁移 `data.py`

这是第一个有点棘手的文件，因为 Jittor 的数据加载机制和 PyTorch 不完全一样。

#### 原 PyTorch 代码
```python
from torchvision import datasets, transforms
import torch.utils.data

train_data = datasets.CIFAR10(dir_, train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)
```

#### 迁移后 Jittor 代码
```python
from jittor.dataset import CIFAR10
import jittor.transform as transforms

# Jittor 自带 CIFAR10 数据集
train_data = CIFAR10(train=True, transform=transforms.Compose([
    transforms.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]))

# Jittor 的 Dataset 自带批处理功能，不需要 DataLoader
train_data.set_attrs(batch_size=128, shuffle=True)
```

#### ⚠️ 重要差异
- Jittor **没有 DataLoader**，`Dataset` 对象本身就支持迭代。
- 使用 `dataset.set_attrs(batch_size=..., shuffle=...)` 设置批次大小。
- 迭代时直接 `for batch in train_data:`。

---

### 5.4 迁移 `data_aug.py`

#### 关键改动
1. `torch.zeros` → `jt.zeros`
2. `torch.flip` → `jt.flip`（用法完全一致）
3. **注意**：`random` 库无需改动，Python 原生随机数生成与框架无关。

**示例**：
```python
# PyTorch
rst = torch.zeros((len(input_tensor), 3, 32, 32), dtype=torch.float32, device=input_tensor.device)

# Jittor
rst = jt.zeros((len(input_tensor), 3, 32, 32), dtype=jt.float32)  # Jittor 自动管理设备，无需指定 device
```

---

### 5.5 迁移 `adv_attack.py` 🔥 **核心难点**

这是整个迁移过程中最难的部分，因为涉及**手动梯度计算**。

#### 原 PyTorch 代码（自适应步长核心）
```python
def get_adv_adaptive_step_size(model, x_nat, x_adv, y, gdnorm, args, epsilon):
    model.eval()
    x_adv.requires_grad_()
    with torch.enable_grad():
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, [x_adv])[0]  # 🔥 关键：手动计算梯度
    
    with torch.no_grad():
        cur_gdnorm = torch.norm(grad.view(len(x_adv), -1), dim=1) ** 2
        step_sizes = 1 / (1 + torch.sqrt(cur_gdnorm) / args.c) * 2 * 8 / 255
        # ...
    return x_adv, cur_gdnorm
```

#### 迁移后 Jittor 代码
```python
def get_adv_adaptive_step_size(model, x_nat, x_adv, y, gdnorm, args, epsilon):
    model.eval()
    
    # Jittor 需要显式标记需要梯度的变量
    x_adv_var = jt.array(x_adv)
    x_adv_var.start_grad()  # 🔥 关键：启动梯度追踪
    
    logits = model(x_adv_var)
    loss = nn.cross_entropy(logits, y)
    
    # Jittor 的梯度计算方式
    grad = jt.grad(loss, x_adv_var)  # 🔥 关键：jt.grad 替代 torch.autograd.grad
    
    # 后续的 no_grad 在 Jittor 中也是 jt.no_grad()
    with jt.no_grad():
        cur_gdnorm = jt.norm(grad.view(len(x_adv_var), -1), dim=1) ** 2
        step_sizes = 1 / (1 + jt.sqrt(cur_gdnorm) / args.c) * 2 * 8 / 255
        # ...
    return x_adv_var, cur_gdnorm
```

#### 🔥 关键差异点
| 操作 | PyTorch | Jittor |
|------|---------|--------|
| 启动梯度追踪 | `x.requires_grad_()` | `x.start_grad()` |
| 计算梯度 | `torch.autograd.grad(loss, [x])[0]` | `jt.grad(loss, x)` |
| 梯度上下文 | `with torch.enable_grad():` | 无需（默认启用） |
| 停止梯度 | `x.detach()` | `x.detach()` 或 `x.stop_grad()` |

---

### 5.6 迁移 `ATAS.py`

#### 关键改动点
1. **删除所有设备相关代码**：
   ```python
   # PyTorch
   model = model.cuda()
   data = data.to(device)
   
   # Jittor：全部删除，Jittor 自动管理
   # （什么都不用写）
   ```

2. **优化器反向传播**：
   ```python
   # PyTorch
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   
   # Jittor
   optimizer.step(loss)  # 🔥 一行搞定！Jittor 把三步合并了
   ```

3. **DataParallel 删除**：
   ```python
   # PyTorch
   model = torch.nn.DataParallel(model)
   
   # Jittor：删除这一行，Jittor 自动并行
   ```

4. **模型保存/加载**：
   ```python
   # PyTorch
   torch.save(model.state_dict(), 'model.pth')
   
   # Jittor
   jt.save(model.state_dict(), 'model.pkl')  # 注意扩展名通常用 .pkl
   ```

---

## 6. 关键难点攻克

### 6.1 梯度计算的完整对比

**场景**：在对抗攻击中，我们需要计算 Loss 对输入图片的梯度。

#### PyTorch 写法
```python
x_adv = x_nat.clone().detach()
x_adv.requires_grad = True

with torch.enable_grad():
    output = model(x_adv)
    loss = criterion(output, y)

grad = torch.autograd.grad(loss, x_adv)[0]
x_adv = x_adv + step_size * grad.sign()
```

#### Jittor 写法
```python
x_adv = jt.array(x_nat)
x_adv.start_grad()  # 启动梯度追踪

output = model(x_adv)
loss = criterion(output, y)

grad = jt.grad(loss, x_adv)  # 直接计算梯度
x_adv = x_adv + step_size * grad.sign()
```

### 6.2 Jittor 的 `execute` vs PyTorch 的 `forward`

这是新手最容易踩的坑。

- **PyTorch**：所有 `nn.Module` 的前向传播方法叫 `forward`。
- **Jittor**：所有 `nn.Module` 的前向传播方法叫 `execute`。

**迁移时必须全局替换**：
```bash
# 在命令行中批量替换（慎用，先备份！）
sed -i 's/def forward(/def execute(/g' *.py
```

---

## 7. 验证与对齐

### 7.1 逐模块验证策略

迁移完一个文件后，立即写一个小脚本验证其输出是否与 PyTorch 一致。

#### 示例：验证 ResNet18 的输出
```python
import torch
import jittor as jt
from models.preact_resnet import PreActResNet18 as PyTorchResNet
from models_jittor.preact_resnet import PreActResNet18 as JittorResNet

# 创建相同的输入
x_torch = torch.randn(2, 3, 32, 32)
x_jittor = jt.array(x_torch.numpy())

# 创建模型
model_torch = PyTorchResNet(num_classes=10)
model_jittor = JittorResNet(num_classes=10)

# 同步权重（手动复制参数）
for (name_t, param_t), (name_j, param_j) in zip(model_torch.named_parameters(), model_jittor.named_parameters()):
    param_j.assign(jt.array(param_t.detach().numpy()))

# 前向传播
out_torch = model_torch(x_torch)
out_jittor = model_jittor(x_jittor)

# 对比输出
diff = abs(out_torch.detach().numpy() - out_jittor.numpy()).max()
print(f"最大误差: {diff}")  # 应该 < 1e-5
```

### 7.2 使用你的 PyTorch 模型权重作为标准答案

你之前训练好的 `last_autodl.pt` 就是最好的"标准答案"。

#### 步骤
1. 加载 PyTorch 模型权重到 Jittor 模型（需要手动转换）。
2. 用同一张图片输入两个模型。
3. 对比输出的 Logits，误差应 < 1e-4。

---

## 8. 常见问题 FAQ

### Q1: Jittor 报错 "CUDA not available"
**A**: 
```bash
# 检查 CUDA 配置
python -m jittor.test.test_cuda

# 如果没有 GPU，可以强制使用 CPU
export use_cuda=0
```

### Q2: `jt.grad` 返回 None
**A**: 确保变量调用了 `.start_grad()`，且 Loss 确实依赖这个变量。

### Q3: 训练速度比 PyTorch 慢
**A**: 
- Jittor 有"预热期"，前几轮会编译算子，后面会变快。
- 确保安装了 CUDA 版本（`pip install jittor-cuda`）。

### Q4: 如何调试 Jittor 代码？
**A**: 
```python
# 打印中间变量
print(x.shape, x.dtype)

# 查看是否在 GPU 上
print(jt.flags.use_cuda)
```

---

## 9. 完整迁移检查清单 (Checklist)

在提交作业前，确保完成以下所有项：

- [ ] 所有文件的 `import torch` 已改为 `import jittor as jt`
- [ ] 所有 `forward` 方法已改为 `execute`
- [ ] 删除了所有 `.cuda()` 和 `.to(device)` 调用
- [ ] `torch.autograd.grad` 已改为 `jt.grad`
- [ ] `optimizer.zero_grad() + loss.backward() + optimizer.step()` 已改为 `optimizer.step(loss)`
- [ ] 数据加载部分已适配 Jittor 的 Dataset
- [ ] 至少验证了一个模块的输出与 PyTorch 一致
- [ ] 能够成功跑完至少 1 个 Epoch 的训练

---

## 10. 推荐的工作流程

```
第 1 天: 搭建环境 + 迁移 models/
第 2 天: 迁移 data.py 和 data_aug.py
第 3 天: 攻克 adv_attack.py（最难）
第 4 天: 迁移 ATAS.py + 调试运行
第 5 天: 验证结果 + 撰写报告
```

---

## 附录：完整的模板文件

我会在后续为你生成几个关键文件的完整迁移版本，你可以直接参考或使用。

**祝你迁移顺利！如果遇到具体的报错或卡住的地方，随时告诉我。** 🚀
