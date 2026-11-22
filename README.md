# CNN Shortcut Learning & Grad-CAM Explainability

Four PyTorch experiments about what convolutional networks actually learn, and how to catch a model that looks accurate for the wrong reasons.

| # | Notebook | What it shows | Result |
|---|---|---|---|
| 1 | `01_cnn_mnist.ipynb` | Custom CNN on MNIST, loss curves, first-layer filter visualisation | **99.04%** test accuracy |
| 2 | `02_colored_mnist_shortcut_learning.ipynb` | Same CNN trained on a *biased* Colored-MNIST where colour correlates with the digit | **99.46%** on biased test, **92.95%** on unbiased test |
| 3 | `03_resnet18_transfer_learning_stl10.ipynb` | ImageNet-pretrained ResNet-18, frozen backbone, new 10-class head on STL-10 | **93.78%** test accuracy (3 epochs) |
| 4 | `04_gradcam_explainability.ipynb` | Grad-CAM heatmaps on correct and incorrect ResNet-18 predictions | Visual explanation of where the model looks |

## Key finding: high accuracy can hide a shortcut

In experiment 2 the training data is biased: each digit mostly appears in one colour. The network scores 99.46% on a test set with the same bias, but drops to 92.95% when colour no longer matches the digit.

The optimizer minimises loss using whichever feature is easiest to separate. Colour is a much simpler signal than stroke shape, so the model partly learns "red means 3" instead of the shape of a 3. A single headline accuracy number hides this. Only a test set designed to break the shortcut reveals it.

Ways to force the model onto shape (discussed in the notebook):
- convert inputs to grayscale so colour carries no information
- colour jitter / random recolouring augmentation
- re-balance the training data so every digit appears in every colour

## Transfer learning + explainability

Experiment 3 freezes all ResNet-18 layers pretrained on ImageNet and trains only a new final linear layer, which reaches 93.78% on STL-10 in 3 epochs. Experiment 4 uses Grad-CAM on the last residual block to show which image regions drove each prediction, comparing correct and misclassified examples.

## Tech

PyTorch, torchvision, pytorch-grad-cam, OpenCV, NumPy, Matplotlib

## Run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or: pip install torch torchvision pytorch-grad-cam opencv-python matplotlib
jupyter notebook notebooks/
```

Run the notebooks in order. MNIST and STL-10 download automatically into `data/`. Experiment 2 expects the Colored-MNIST tensors in `data/cmnist/` (`train_biased.pt`, `test_biased.pt`, `test_unbiased.pt`). Notebooks 1 and 3 save model weights to `models/` (not committed), which notebook 4 loads.

## Results

- `results/mnist_training_curves.png`: loss and accuracy curves for experiment 1
- `results/resnet18_stl10_accuracy.txt`: final ResNet-18 test accuracy
