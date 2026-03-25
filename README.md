# Character-Level Bigram Language Model

This repository contains a Python implementation of a character-level language model using a **Bigram** approach. The model learns the probability of a character following another based on a dataset of names, allowing it to generate new, name-like sequences.

## 🚀 Overview
The core objective is to build a model that predicts the next suitable or high-probability character for a given word. We achieve this by:
1. **Counting Bigrams:** Analyzing pairs of consecutive characters (e.g., in "emma", the bigrams are `<S>e`, `em`, `mm`, `ma`, and `a<E>`).
2. **Frequency Mapping:** Storing these counts in a 2D matrix where rows represent the first character and columns represent the second.
3. **Probability Distribution:** Converting counts into probabilities to sample the next character.

## 📊 Dataset
The model uses a text file (`names.txt`) containing approximately **32,033 names**.
* **Shortest name length:** 2 characters.
* **Longest name length:** 15 characters.

## 🛠️ Implementation Details

### Data Preprocessing
* **Vocabulary:** A unique set of 26 English lowercase letters plus a special character `.` used to denote the start and end of a name.
* **Mapping:**
    * `stoi`: String-to-integer mapping (e.g., `.` $\rightarrow$ 0, `a` $\rightarrow$ 1).
    * `itos`: Integer-to-string mapping (reverses `stoi`).

### The Model Matrix (N)
Instead of a simple Python dictionary, a **PyTorch 2D Tensor (27x27)** is used for efficient computation. 
* **Rows ($i$):** Represent the current character.
* **Columns ($j$):** Represent the next character.
* **Value ($N[i, j]$):** The frequency of the bigram $ij$ in the dataset.

### Visualizing the Bigrams
The project includes a visualization script using `matplotlib` to display the 27x27 matrix, showing exactly how many times each character pair occurs across the entire training set.

## 💻 Tech Stack
* **Python**
* **PyTorch:** Used for tensor operations and matrix management.
* **Matplotlib:** Used for visualizing the bigram frequency distribution.

## 📖 References
This project was built following the instructional videos by **Andrej Karpathy**, specifically the "Neural Networks: Zero to Hero" series.

*** ### How to Use
1. Ensure you have the `names.txt` file available.
2. Run the cells in the `Next_Char_prediction.ipynb` notebook to train the model.
3. Use the sampling cell at the end of the notebook to generate new names.
