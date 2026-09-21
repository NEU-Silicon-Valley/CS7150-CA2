# Coding Assignment 2: Optimization, Network Architecture, and CNNs

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

This assignment is broken down into three notebooks, each building upon the last.

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

## Grading: 60 points here, 20 from discussion

This assignment is worth **80 points total**.

| | Points |
|---|--------|
| Notebooks in this repository | 60 |
| Group discussion | 20 |
| **Total** | **80** |

### Points in the notebooks (60)

| Notebook | Section | Points |
|----------|---------|--------|
| `CA2.1` | Part 1 — Gradient descent and weight decay | 8 |
| `CA2.1` | Part 2 — The ADAM optimizer | 11 |
| `CA2.1` | Part 3 — Training neural networks | 16 |
| `CA2.1` | Part 4 — Weight initialization | 9 |
| `CA2.2` | Convolutions (9 exercises, 1 pt each) | 9 |
| `CA2.3` | CIFAR-10 classifier | 7 |
| | **Total** | **60** |

`CA2.1` also has two extra credit questions worth **5 points** on top of the 60.

**How to read the point markers.** Every graded item carries exactly one point value, written in
square brackets next to it. A section heading never carries a separate total on top of the items
underneath it, so you can add up the brackets as you go and get the right number. The subtotals in
the table above are the only place sections are summed.

Some closely related tasks are graded as a group — for example Tasks 3.1 to 3.3 all run the same
experiment with a different activation function, so they share a single point between them. Where
that happens the first task says `[Tasks 3.1-3.3: 1 point total]` and the others say
`[graded with Task 3.1]`. You still need to do all of them.

### About the discussion (20)

You'll be in a small group of about three students. We'll walk through parts of the assignment
together and ask you to talk through your own work: why you took a particular approach, what gave
you trouble, what you would do differently. You can also earn credit by weighing in on a question
put to someone else in your group.

Keep it relaxed. There's nothing extra to prepare beyond knowing what you submitted, and "I got
stuck on this part and looked it up" is a perfectly good answer. The point is to hear you reason
about your own code, not to catch anyone out.

---

### Setup and Environment on Explorer HPC

The recommended platform for this assignment is the **Explorer HPC**, and the easiest way to use it is through the [Open OnDemand (OOD) web interface](https://ood.explorer.northeastern.edu/).

**Helpful Links:**
* **Access Explorer:** You can access the interactive dashboard from [here](https://ood.explorer.northeastern.edu/).
* **Getting started with Explorer:** Refer to this [document](https://docs.google.com/document/d/1nsP4YUBajdM6j3R0tA4gRnwo9qPdRv5gTtXHYdiXCz8/) to create environments, manage files, and start sessions in our HPC. 
* **Official Documentation:** For more info on the HPC, refer to the [Official Documentation](https://rc-docs.northeastern.edu/en/latest/).

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
3.  Download your three completed notebooks (`.ipynb` files) from the HPC to your local machine.
4. Zip the three completed notebook files, along with their PDF versions; and name the zip file as "*CA2-Your-Last-Name*".
5.  Submit the zip file as a direct **reply** to the **Coding Assignment 2** discussion thread on Canvas, found in the **Coding Assignments** module for the week this assignment is due. Reply to the existing thread rather than starting a new one.

### Important Notes

* Please adhere to the collaboration policy outlined in the course syllabus.
* Make sure to credit any external resources, discussions, or AI assistance you used to complete the assignment.
* Start early! These notebooks comprehensively tries to translate the theoretical knowledge to practical application. Good luck!
