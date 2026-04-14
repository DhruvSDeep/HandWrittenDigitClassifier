# HandWrittenDigitClassifier

A convolutional neural network trained on the **MNIST** dataset to classify handwritten digits (0–9). A clean, minimal PyTorch implementation that achieves high accuracy in just 10 epochs, with per-epoch accuracy tracking and a train-vs-test accuracy plot.

---

## Highlights

- **Classic CNN baseline** — two convolutional layers, two fully connected layers, and nothing more. A clean starting point for digit recognition.
- **High accuracy, fast convergence** — reaches strong test accuracy within 10 epochs on MNIST's 60 000 training / 10 000 test images.
- **Accuracy curve visualization** — plots training and test accuracy side-by-side across epochs to monitor generalization at a glance.

---

## Model Architecture

```
Input (1 × 28 × 28 grayscale image)
  │
  ├─ Conv2d(1 → 32, 3×3, padding=1) → ReLU → MaxPool(2×2)     → [32 × 14 × 14]
  ├─ Conv2d(32 → 64, 3×3, padding=1) → ReLU → MaxPool(2×2)     → [64 × 7 × 7]
  │
  ├─ Flatten                                                      → [3 136]
  ├─ Linear(3136 → 128) → ReLU
  └─ Linear(128 → 10)
```

---

## Project Structure

```
HandWrittenDigitClassifier/
├── HandDig.ipynb      # Full pipeline: data loading, model definition, training, evaluation, plotting
└── README.md
```

At runtime the notebook produces:
- `mnist_cnn.pth` — saved model weights (updated each epoch).
- An inline accuracy-vs-epochs plot comparing train and test performance.

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch, matplotlib

```bash
pip install torch torchvision matplotlib tqdm
```

### Dataset

The notebook expects the MNIST dataset in PNG format, organized by digit folder:

```
mnist_png/mnist_png/
├── training/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 9/
└── testing/
    ├── 0/
    ├── 1/
    ├── ...
    └── 9/
```

You can obtain this from the [MNIST-PNG](https://github.com/myleott/mnist_png) conversion or generate it yourself from the original MNIST data. The images are loaded via `torchvision.datasets.ImageFolder` and converted to single-channel grayscale tensors.

### Train & Evaluate

Open and run `HandDig.ipynb`. The notebook will:

1. Load training (60 000 images) and test (10 000 images) sets.
2. Train the CNN for 10 epochs, printing train and test accuracy each epoch.
3. Save the model weights to `mnist_cnn.pth`.
4. Plot a training-vs-test accuracy curve.

Training hyperparameters:

| Parameter | Value |
|---|---|
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻³ |
| Loss | Cross-entropy |
| Epochs | 10 |

---

## How It Works

Each 28×28 grayscale digit image passes through two convolutional layers with ReLU activation and 2×2 max pooling, reducing the spatial dimensions to 7×7. The resulting 64-channel feature map is flattened into a 3 136-dimensional vector, projected through a hidden layer of 128 units, and mapped to 10 class logits. The model is trained end-to-end with cross-entropy loss and Adam, and the weights are saved after every epoch.

---

## License

No license specified. Contact the repository owner for usage terms.
