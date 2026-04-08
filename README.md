# Deep Learning Assignment 3

## Student Info
- **Name:** Abdullah Qureshi
- **Course:** Deep Learning
- **Dataset Used:** MNIST, Colored-MNIST, STL-10

---

## Repository Structure
DL-Assignment-3/
├── notebooks/
│   ├── task1_partA.ipynb   # Custom CNN on MNIST
│   ├── task1_partB.ipynb   # CNN on Colored-MNIST
│   ├── task2_partA.ipynb   # ResNet-18 fine-tuning on STL-10
│   └── task2_partB.ipynb   # GradCAM Visualization
├── models/
│   ├── partA_cnn.pth       # Saved MNIST model
│   └── resnet18_stl10.pth  # Saved ResNet model
├── results/
│   ├── partA_training_curves.png
│   └── task2_accuracy.txt
└── README.md

---


## How to Run
1. Clone the repo
2. Install requirements: `pip install torch torchvision pytorch-grad-cam matplotlib`
3. Run notebooks in order from task1_partA to task2_partB

---

