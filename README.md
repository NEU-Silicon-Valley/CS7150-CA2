# Coding Assignment 2: Optimization, CNNs, and Transfer Learning

Now that you've learned the fundamental mechanics of backpropagation and autograd in CA1, it's time to make neural networks truly work! In this assignment, we will bridge the gap between theory and practice. You'll learn how to control the training process with optimizers, build your own Convolutional Neural Networks (CNNs) for image recognition, and leverage one of the most powerful techniques in modern AI: transfer learning.

By the end of this assignment, you will have built, trained, and analyzed several neural networks for different tasks, giving you a solid, practical foundation in deep learning.

## Learning Objectives

After completing this assignment, you will be able to:

* **Implement and understand optimizers** like SGD and ADAM from scratch.
* **Analyze the effect of key hyperparameters** such as learning rates and weight decay.
* **Understand the mechanics of a convolution operation** and its relationship to linear algebra.
* **Explain the concepts of parameter sharing and equivariance** in CNNs.
* **Build, train, and evaluate a CNN** from scratch for image classification using PyTorch.
* **Implement transfer learning** by fine-tuning a state-of-the-art, pre-trained model (VGG16) on a new dataset.
* **Apply CNNs to non-image data**, such as DNA sequences, demonstrating their versatility.

---

## Assignment Structure

This assignment is broken down into five notebooks, each building upon the last.

1.  **`CA2.1_OptimizationAndNetworkArchitecture.ipynb`**
    * You'll start by implementing optimizers like SGD and ADAM from scratch to build a deep intuition for how models learn. You will then use these to train simple neural networks and analyze the impact of different activation functions and loss functions.

2.  **`CA2.2_Convolutions.ipynb`**
    * This notebook demystifies CNNs by showing you that a convolution is just a special, highly efficient form of matrix multiplication. You will work through the mechanics of 1D and 2D convolutions, padding, and equivariance.

3.  **`CA2.3_CIFAR_classifier.ipynb`**
    * Time to apply your knowledge! You'll build your first CNN to classify images from the well-known CIFAR-10 dataset and compare its performance and efficiency against a standard multi-layer perceptron (MLP).

4.  **`CA2.4_Transfer_Learning.ipynb`**
    * **(Read-through only)** Training a large network from scratch is tough. This notebook is a guided tutorial on transfer learning. You will see how to adapt a pre-trained VGG16 model to a new task, achieving high accuracy with minimal training.

5.  **`CA2.5_CNN_for_DNA.ipynb`**
    * CNNs aren't just for images! In this final notebook, you'll see how to apply convolutional networks to sequence data. You will build a model to find specific patterns in DNA sequences and answer 14 questions as you go.

---

## File Descriptions

This repository contains the following files:

* **Notebooks (`.ipynb`):**
    * `CA2.1_OptimizationAndNetworkArchitecture.ipynb`: Part 1 of the assignment.
    * `CA2.2_Convolutions.ipynb`: Part 2 of the assignment.
    * `CA2.3_CIFAR_classifier.ipynb`: Part 3 of the assignment.
    * `CA2.4_Transfer_Learning.ipynb`: **Read-through tutorial, not for submission.**
    * `CA2.5_CNN_for_DNA.ipynb`: Part 5 of the assignment.

* **Helper Scripts (`.py`):**
    * `hw2utils.py`: A helper script containing utility classes and functions needed for `CA2.1`.

* **Datasets (`.npz`, `.zip`):**
    * `tiny-classification.npz`: Dataset for the training exercises in `CA2.1`.
    * `hard-classification.npz`: Dataset for the extra credit portion of `CA2.1`.
    * `cropdata.zip`: Zipped dataset for `CA2.4`. The notebook will handle unzipping this file automatically.

---

## Setup and Environment on Explorer HPC

The recommended platform for this assignment is the **Explorer HPC**. Using the HPC is essential for installing the correct package versions and for training your models on a GPU compute node instead of the login node.

**Helpful Links:**
* **Access Explorer:** <https://ood.explorer.northeastern.edu/>
* **Getting Started with the HPC:** <https://rc-docs.northeastern.edu/en/latest/>

Follow these steps carefully to set up your environment:

**1. Start an Interactive Session**
First, log in to Explorer and open a shell. Before creating any environment or installing packages, you **must** start an interactive session on a compute node. This ensures that the resource-intensive tasks of installation and training do not run on the shared login node.

```bash
srun -p short --pty bash
```

**2. Create and Activate a Conda Environment**
Create a dedicated Conda environment for this assignment to keep your packages isolated.

```bash
# Create the environment (you only need to do this once)
conda create --name myenv

# Activate the environment (do this every time you start a new session)
source activate myenv
```

**3. Install Core Packages with Conda**
Install the basic data science libraries using Conda.

```bash
conda install pandas numpy scikit-learn matplotlib
```

**4. Install a Specific PyTorch Version with CUDA**
To avoid a common `AcceleratorError` with CUDA, we need to install a specific version of PyTorch that is compatible with the HPC's drivers.

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

**5. Install Remaining Packages**
Finally, install any other required packages. e.g.:

```bash
pip install scipy
```

You are now ready to launch Jupyter and begin the assignment!

---

## Submission Instructions

1.  Complete the exercises in notebooks **`CA2.1`**, **`CA2.2`**, **`CA2.3`**, and **`CA2.5`**. This involves filling in the `TODO` sections and answering the inline questions.
2.  Run all cells in these four notebooks from top to bottom on the HPC so that the outputs are clearly visible.
3.  Notebook **`CA2.4_Transfer_Learning.ipynb`** is a read-through tutorial. You should run it to understand the concepts, but it **does not need to be submitted**.
4.  Download your four completed notebooks (`.ipynb` files) from the HPC to your local machine.
5.  Submit the four completed notebook files as a direct reply to the assignment module on Canvas.

## Important Notes

* Please adhere to the collaboration policy outlined in the course syllabus.
* Make sure to credit any external resources, discussions, or AI assistance you used to complete the assignment.
* Start early! These notebooks comprehensively tries to translate the theoretical knowledge to practical application. Good luck!