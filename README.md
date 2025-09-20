# Coding Assignment 2: Optimization, CNNs, and Transfer Learning

Now that you've learned the fundamental mechanics of backpropagation and autograd in CA1, it's time to make neural networks truly work! In this assignment, we will bridge the gap between theory and practice. You'll learn how to control the training process with optimizers and build your own Convolutional Neural Networks (CNNs) from scratch for image recognition.

By the end of this assignment, you will have built, trained, and analyzed neural networks for image classification, giving you a solid, practical foundation in deep learning.


## Learning Objectives

After completing this assignment, you will be able to:

* **Implement and understand optimizers** like SGD and ADAM from scratch.
* **Analyze the effect of key hyperparameters** such as learning rates and weight decay.
* **Understand the mechanics of a convolution operation** and its relationship to linear algebra.
* **Explain the concepts of parameter sharing and equivariance** in CNNs.
* **Build, train, and evaluate a CNN** from scratch for image classification using PyTorch.

</br>

## Assignment Structure

This assignment is broken down into five notebooks, each building upon the last.

1.  **`CA2.1_OptimizationAndNetworkArchitecture.ipynb`**
    * You'll start by implementing optimizers like SGD and ADAM from scratch to build a deep intuition for how models learn. You will then use these to train simple neural networks and analyze the impact of different activation functions and loss functions.

2.  **`CA2.2_Convolutions.ipynb`**
    * This notebook demystifies CNNs by showing you that a convolution is just a special, highly efficient form of matrix multiplication. You will work through the mechanics of 1D and 2D convolutions, padding, and equivariance.

3.  **`CA2.3_CIFAR_classifier.ipynb`**
    * Time to apply your knowledge! You'll build your first CNN to classify images from the well-known CIFAR-10 dataset and compare its performance and efficiency against a standard multi-layer perceptron (MLP).

</br>

## File Descriptions

This repository contains the following files:

* **Notebooks (`.ipynb`):**
    * `CA2.1_OptimizationAndNetworkArchitecture.ipynb`: Part 1 of the assignment.
    * `CA2.2_Convolutions.ipynb`: Part 2 of the assignment.
    * `CA2.3_CIFAR_classifier.ipynb`: Part 3 of the assignment.
    
* **Helper Scripts (`.py`):**
    * `hw2utils.py`: A helper script containing utility classes and functions needed for `CA2.1`.

* **Datasets (`.npz`, `.zip`):**
    * `tiny-classification.npz`: Dataset for the training exercises in `CA2.1`.
    * `hard-classification.npz`: Dataset for the extra credit portion of `CA2.1`.

---

### Setup and Environment on Explorer HPC

The recommended platform for this assignment is the **Explorer HPC**, and the easiest way to use it is through the [Open OnDemand (OOD) web interface](ttps://ood.explorer.northeastern.edu/).

**Helpful Links:**
* **Access Explorer:** You can access the interactive dashboard from [here](https://ood.explorer.northeastern.edu/).
* **Getting started with Explorer:** Refer to this [document](https://docs.google.com/document/d/1nsP4YUBajdM6j3R0tA4gRnwo9qPdRv5gTtXHYdiXCz8/) to create environments, manage files, and start sessions in our HPC. 
* **Official Documentaion:** For more info on the HPC, refer to the [Official Documentation](https://rc-docs.northeastern.edu/en/latest/).

Follow these steps carefully to set up your environment:

**1. Initial Environment Setup**

Before launching a notebook, you'll need a Conda environment with the correct packages. You can create one using a terminal within the OOD interface (`Clusters >_ Shell Access`). Once you have created and activated a new environment, install the required packages. 

You may refer to the *Creating your virtual environment* section from this [Guide](https://docs.google.com/document/d/1nsP4YUBajdM6j3R0tA4gRnwo9qPdRv5gTtXHYdiXCz8/).


Install the following packages. The specific PyTorch version is **crucial** for compatibility with the HPC's GPU drivers.

```bash
# First, install standard libraries with Conda
conda install pandas numpy scikit-learn matplotlib scipy

# Second, install the specific PyTorch version with pip
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

You are now ready to launch Jupyter and begin the assignment!

---

### Submission Instructions

1.  Complete the exercises in notebooks **`CA2.1`**, **`CA2.2`**, and **`CA2.3`**. This involves filling in the `TODO` sections and answering the inline questions.
2.  Run all cells in these three notebooks from top to bottom on the HPC so that the outputs are clearly visible.
3.  Download your four completed notebooks (`.ipynb` files) from the HPC to your local machine.
4. Zip the three completed notebook files, along with their PDF versions; and name the zip file as "*CA02-Your-Last-Name*".
5.  Submit the zip file as a direct **reply** to the [*coding assignment module of week 2*](https://northeastern.instructure.com/courses/226141/discussion_topics/2877295?module_item_id=12458419) on Canvas.

### Important Notes

* Please adhere to the collaboration policy outlined in the course syllabus.
* Make sure to credit any external resources, discussions, or AI assistance you used to complete the assignment.
* Start early! These notebooks comprehensively tries to translate the theoretical knowledge to practical application. Good luck!