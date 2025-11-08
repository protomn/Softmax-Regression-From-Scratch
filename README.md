# Softmax Regression from Scratch

A complete, production-quality implementation of **Softmax Regression** (Multinomial Logistic Regression) built entirely from scratch in NumPy, featuring advanced regularization techniques, class imbalance handling, and comprehensive evaluation tools.

**Benchmark Result**: Achieved **90.07% accuracy** on MNIST, outperforming scikit-learn's implementation (86.27%) 

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Theory & Mathematics](#theory--mathematics)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Regularization Techniques](#regularization-techniques)
- [Class Imbalance Handling](#class-imbalance-handling)
- [Benchmark Results](#benchmark-results)
- [Usage Examples](#usage-examples)
- [Performance Analysis](#performance-analysis)

---

## Overview

This project implements **Softmax Regression** from first principles.

### Why This Project?

While libraries like scikit-learn provide optimized implementations, building from scratch:
- Deepened my understanding of gradient-based optimization
- Revealed numerical stability challenges and solutions
- Enabled me to implement custom loss functions and regularization

---

## Features

### Core Implementation
- [x] **Numerically Stable Softmax** - Using log-sum-exp trick to prevent overflow
- [x] **Cross-Entropy Loss** - With epsilon clipping for numerical stability
- [x] **Vectorized Gradients** - Efficient batch gradient computation
- [x] **One-Hot Encoding** - Support for both hard and soft labels

### Regularization
- [x] **L2 Regularization (Ridge)** - Prevents overfitting with weight decay
- [x] **L1 Regularization (Lasso)** - Induces sparsity in weights
- [x] **Elastic Net** - Combines L1 and L2 with configurable ratio

### Class Imbalance
- [x] **Weighted Cross-Entropy** - Automatic class balancing
- [x] **Focal Loss** - Focuses on hard-to-classify examples
- [x] **Custom Class Weights** - Manual weight specification

### Training & Evaluation
- [x] **Early Stopping** - Prevents overfitting with patience-based halting
- [x] **Train/Validation/Test Split** - Proper evaluation methodology
- [x] **Learning Curves** - Visualization of training progress
- [x] **Confusion Matrix** - Detailed error analysis
- [x] **Classification Reports** - Per-class metrics (precision, recall, F1)
- [x] **Model Comparison** - Side-by-side evaluation of different configurations

---

## Theory & Mathematics

### Softmax Function

The softmax function converts raw scores (logits) into a probability distribution:

```
σ(z)_i = exp(z_i) / Σ_j exp(z_j)
```

**Numerical Stability**: We subtract the maximum logit before exponentiation:

```
σ(z)_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
```

### Cross-Entropy Loss

For a single sample with one-hot encoded label **y** and prediction **ŷ**:

```
L(y, ŷ) = -Σ_i y_i * log(ŷ_i)
```

With one-hot labels, this simplifies to:

```
L(y, ŷ) = -log(ŷ_c)  where c is the true class
```

### Gradient Computation

The beauty of combining softmax with cross-entropy is the elegant gradient:

```
∂L/∂z = (ŷ - y) / n
```

Where:
- **z** are the logits (pre-softmax scores)
- **ŷ** are the softmax probabilities
- **y** are the one-hot encoded labels
- **n** is the batch size

This leads to simple weight updates:

```
∂L/∂W = (1/n) * X^T @ (ŷ - y)
∂L/∂b = (1/n) * Σ(ŷ - y)
```

### Regularization

**L2 (Ridge)**:
```
Loss = CrossEntropy + (λ/2) * ||W||²
Gradient: ∂L/∂W = ∂CE/∂W + λ * W
```

**L1 (Lasso)**:
```
Loss = CrossEntropy + λ * ||W||₁
Gradient: ∂L/∂W = ∂CE/∂W + λ * sign(W)
```

**Elastic Net**:
```
Loss = CrossEntropy + α * λ * ||W||₁ + (1-α) * (λ/2) * ||W||²
```

### Focal Loss

Addresses class imbalance by down-weighting easy examples:

```
FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
```

Where:
- **p_t** is the predicted probability for the true class
- **γ** (gamma) controls focusing (2.0 by default)
- **α** provides class-level balance

---

## Installation

### Prerequisites
```bash
Python 3.7+
NumPy >= 1.19.0
Matplotlib >= 3.3.0
scikit-learn >= 0.24.0 (for datasets and metrics)
```

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/softmax-regression-scratch.git
cd softmax-regression-scratch

#Make sure you have the required dependencies installed.

# Launch Jupyter Notebook
jupyter notebook softmax_regression.ipynb
```

---

## Quick Start

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as tts

# Generate synthetic data
X, y = make_classification(n_samples = 1000, n_features = 20, 
                          n_classes = 4, random_state = 42)

# Split data
X_train, X_test, y_train, y_test = tts(
    X, y, test_size = 0.3, random_state = 42
)

# One-hot encode labels
Y_train = one_hot_encoder(y_train, num_classes = 4)
Y_test = one_hot_encoder(y_test, num_classes = 4)

# Train model
W, b, history = train(
    X_train, Y_train,
    X_val, Y_val,
    n_classes = 4,
    learning_rate = 0.1,
    n_epochs = 200,
    reg_type = 'l2',
    lambda_reg = 0.01,
    early_stopping = EarlyStopping(patience = 20)
)

# Evaluate
_, probs = forward_pass(X_test, W, b)
predictions = np.argmax(probs, axis = 1)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## Architecture

### Core Components

```
softmax_regression/
│
├── Data Processing
│   ├── one_hot_encoder()          # Convert integer labels to one-hot
│   └── compute_class_weights()   # Calculate balanced class weights
│
├── Model Components
│   ├── init_params()   # Weights/Biases initialization
│   ├── softmax()                 # Numerically stable softmax
│   ├── forward_pass()            # Compute logits and probabilities
│   └── grad()       # Computes Gradient
│
├── Loss Functions
│   ├── cross_entropy_loss()           # Standard cross-entropy
│   ├── cross_entropy_loss_weighted()  # Class-weighted version
│   └── focal_loss()                   # Focal loss for imbalance
│
├── Regularization
│   └── compute_regularization()  # L1, L2, Elastic Net
│
├── Training
│   ├── train()                   # Main training loop
│   └── EarlyStopping             # Early stopping class
│
└── Evaluation
    ├── evaluate_model()          # Compute test metrics
    ├── plot_training_history()   # Visualize learning curves
    └── plot_confusion_matrix()   # Error analysis
```

---

## Regularization Techniques

### L2 Regularization (Ridge)

```python
W, b, history = train(
    X_train, Y_train, X_val, Y_val,
    n_classes = 4,
    reg_type = 'l2',
    lambda_reg = 0.01  # Regularization strength
)
```

**When to use**: Default choice for most problems. Prevents overfitting by penalizing large weights.

### L1 Regularization (Lasso)

```python
W, b, history = train(
    X_train, Y_train, X_val, Y_val,
    n_classes = 4,
    reg_type = 'l1',
    lambda_reg = 0.01
)
```

**When to use**: Feature selection needed. Drives some weights exactly to zero.

### Elastic Net

```python
W, b, history = train(
    X_train, Y_train, X_val, Y_val,
    n_classes = 4,
    reg_type = 'elastic_net',
    lambda_reg = 0.01,
    l1_ratio = 0.5  # 0 = pure L2, 1 = pure L1
)
```

**When to use**: Combines benefits of L1 and L2. Good for correlated features.

---

## Class Imbalance Handling

### Weighted Cross-Entropy

Automatically balances classes based on their frequencies:

```python
# Compute balanced weights
weights = compute_class_weights(y_train, method = 'balanced')

# Train with weighted loss
W, b, history = train(
    X_train, Y_train, X_val, Y_val,
    n_classes = 4,
    loss_type = 'weighted',
    class_weights = weights
)
```

**Effect**: Rare classes get higher weights, forcing the model to pay more attention.

### Focal Loss

Focuses on hard-to-classify examples:

```python
W, b, history = train(
    X_train, Y_train, X_val, Y_val,
    n_classes = 4,
    loss_type = 'focal',
    focal_gamma = 2.0,      
    focal_alpha = weights   
)
```

**When to use**: 
- Severe class imbalance (e.g., 1:100 ratio)
- Many easy examples dominate the loss
- Need to focus on hard negatives

---

## Benchmark Results

### MNIST Classification (10K subset)

| Model | Accuracy | Training Time | Notes |
|-------|----------|---------------|-------|
| **Our Implementation** | **90.07%** | 15.96s | L2 regularization, early stopping |
| Scikit-learn (LBFGS) | 86.27% | 7.37s | Default multinomial logistic regression |

**Key Observations**:
- **3.8% higher accuracy** than scikit-learn
- Strong performance across all digits (0-9)
- Best on digits 1 (98% recall) and 0 (95% recall)
- Slower due to Python implementation vs C-optimized LBFGS

### Per-Class Performance (Our Model)

| Digit | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.92 | 0.95 | 0.93 | 148 |
| 1 | 0.91 | **0.98** | 0.94 | 173 |
| 2 | 0.92 | 0.92 | 0.92 | 145 |
| 3 | 0.89 | 0.86 | 0.88 | 155 |
| 4 | 0.86 | 0.91 | 0.89 | 136 |
| 5 | 0.87 | 0.88 | 0.88 | 140 |
| 6 | 0.93 | 0.92 | 0.92 | 144 |
| 7 | 0.92 | 0.92 | 0.92 | 158 |
| 8 | 0.87 | 0.81 | 0.84 | 146 |
| 9 | 0.89 | 0.85 | 0.87 | 155 |

**Overall Accuracy**: 90.07%

---

## Usage Examples

### Example 1: Basic Training

```python
# Train with default settings
W, b, history = train(
    X_train, Y_train, X_val, Y_val,
    n_classes = 4,
    learning_rate = 0.1,
    n_epochs = 200
)
```

### Example 2: With Regularization

```python
# Add L2 regularization to prevent overfitting
W, b, history = train(
    X_train, Y_train, X_val, Y_val,
    n_classes = 4,
    learning_rate = 0.1,
    n_epochs = 200,
    reg_type = 'l2',
    lambda_reg = 0.01
)
```

### Example 3: Handle Class Imbalance

```python
# Compute balanced weights
weights = compute_class_weights(y_train, method = 'balanced')

# Train with focal loss
W, b, history = train(
    X_train, Y_train, X_val, Y_val,
    n_classes = 4,
    loss_type = 'focal',
    focal_gamma = 2.0,
    focal_alpha = weights,
    early_stopping = EarlyStopping(patience = 20)
)
```

### Example 4: Model Comparison

```python
# Train multiple configurations
configs = [
    {'reg_type': 'none', 'label': 'Baseline'},
    {'reg_type': 'l2', 'lambda_reg': 0.01, 'label': 'L2'},
    {'reg_type': 'l1', 'lambda_reg': 0.01, 'label': 'L1'},
    {'loss_type': 'focal', 'focal_gamma': 2.0, 'label': 'Focal'}
]

histories = []
for config in configs:
    label = config.pop('label')
    W, b, hist = train(X_train, Y_train, X_val, Y_val, 
                       n_classes = 4, **config)
    histories.append(hist)

# Compare visually
compare_models(histories, [c['label'] for c in configs])
```

---

## Performance Analysis

### Training Convergence

Our implementation shows:
- **Smooth convergence** with proper learning rate tuning
- **Early stopping** typically triggers around epoch 50-100
- **No overfitting** when regularization is applied
- **Stable gradients** thanks to numerical stability tricks

### Computational Complexity

- **Forward pass**: O(n * d * k) where n = samples, d = features, k = classes
- **Gradient computation**: O(n * d * k)
- **Parameter update**: O(d * k)
- **Total per epoch**: O(n * d * k)

### Memory Requirements

- **Weight matrix W**: (d, k) = 784 × 10 = 7,840 parameters for MNIST
- **Activations**: (n, k) stored during forward pass
- **Gradients**: Same shape as weights
- **Total**: ~O(n*k + d*k) for single batch

---

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for API changes
- Maintain backward compatibility

---

## Acknowledgments

- **Scikit-learn** - API design inspiration
- **NumPy Documentation** - Vectorization techniques
- **Papers**:
  - "Focal Loss for Dense Object Detection" (Lin et al., 2017)
  - "Regularization and variable selection via the elastic net" (Zou & Hastie, 2005)

---

## 📧 Contact

**Pratham** - Electrical Engineering Student @ RMIT University

- GitHub: [@protomn](https://github.com/protomn)

---

## 📚 References

1. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
2. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press.
3. **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
4. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning*. Springer.
5. **Lin, T. Y., et al.** (2017). "Focal Loss for Dense Object Detection." *ICCV*.

---

**Built with Python and NumPy**

[⬆ Back to Top](#-softmax-regression-from-scratch)
