# Differences from Original LeNet-5 Paper

This implementation is based on the architecture described in the paper:

**Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner — _Gradient-Based Learning Applied to Document Recognition_ (1998).**

While the overall structure of the network is preserved, several simplifications and modern practices were used to make the implementation compatible with modern deep learning frameworks.

---

# 1. Architecture Differences

| Component            | Original Paper                                               | This Implementation                          | Reason                                |
| -------------------- | ------------------------------------------------------------ | -------------------------------------------- | ------------------------------------- |
| Subsampling Layers   | Trainable average pooling (avg + weight + bias + activation) | `AvgPool2d()`                                | Modern CNNs use non-trainable pooling |
| C3 Connectivity      | Partial connectivity between feature maps                    | Fully connected convolution `Conv2d(6 → 16)` | Easier to implement in PyTorch        |
| Output Layer         | Radial Basis Function (RBF) classifier                       | Linear layer (`Linear(84 → 10)`)             | Standard practice in modern CNNs      |
| Activation Placement | After each convolution and fully connected layer             | Same but simplified                          | Matches modern frameworks             |

---

# 2. Training Differences

| Component              | Original Paper                                                              | This Implementation      | Reason                           |
| ---------------------- | --------------------------------------------------------------------------- | ------------------------ | -------------------------------- |
| Loss Function          | Mean Squared Error (MSE) / MAP criterion                                    | `CrossEntropyLoss`       | Better suited for classification |
| Batch Size             | 1                                                                           | Mini-batch training (64) | Faster training on GPUs          |
| Optimizer              | Stochastic Gradient Descent with diagonal Levenberg-Marquardt approximation | Standard SGD             | Simpler and widely supported     |
| Learning Rate Schedule | Hand-crafted schedule across dataset passes                                 | Constant LR              | Easier to reproduce              |

---

# 3. Input Preprocessing Differences

| Component       | Original Paper           | This Implementation          | Reason                     |
| --------------- | ------------------------ | ---------------------------- | -------------------------- |
| Digit Centering | Center of mass alignment | Resize / basic preprocessing | torchvision pipeline       |
| Anti-aliasing   | Used during resizing     | torchvision resize           | Equivalent modern approach |

---

# 4. Model Architecture (Identical Components)

Despite the differences above, the following key architectural properties remain identical:

| Property       | Value                       |
| -------------- | --------------------------- |
| Input size     | 32 × 32                     |
| C1 layer       | 6 feature maps, kernel 5×5  |
| S2 layer       | 2×2 subsampling             |
| C3 layer       | 16 feature maps, kernel 5×5 |
| S4 layer       | 2×2 subsampling             |
| C5 layer       | 120 feature maps            |
| F6 layer       | 84 neurons                  |
| Output classes | 10                          |

---

This implementation preserves the **core convolutional architecture of LeNet-5** while adopting several modern deep-learning practices to simplify training and integration with PyTorch.

---
