# Next Word Prediction using LSTM

This repository contains a neural language model developed to predict the next word in a sentence sequence. The model is built using Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN) capable of learning order dependence in sequence prediction problems.

## 📌 Overview

Next Word Prediction is a fundamental task in Natural Language Processing (NLP). It involves predicting the most probable word that follows a given sequence of words. This project demonstrates how to build such a model using Deep Learning techniques.

The model is trained on a custom corpus of text data. It learns the statistical structure of the language in the dataset to generate coherent word sequences.

## 🚀 Features

* **Data Preprocessing:** Tokenization, sequence generation, and padding.
* **Model Architecture:** Embedding layer followed by LSTM layers and a Dense output layer.
* **Prediction:** Generates the next word based on a seed text input.
* **Visualization:** Analysis of model accuracy and loss during training.

## 🛠️ Technologies Used

* **Python**
* **TensorFlow / Keras** (Deep Learning Framework)
* **NumPy** (Numerical operations)
* **NLTK** (Natural Language Toolkit for text processing)
* **Matplotlib** (For plotting training graphs)

## 📂 Repository Structure

```text
├── next_word_prediction_with_LSTM.ipynb   # Main Jupyter Notebook containing the code
├── README.md                              # Project Documentation

## 🧠 Model Architecture

The model generally follows this architecture:

1.  **Input Layer:** Takes in a sequence of words.
2.  **Embedding Layer:** Converts word indices into dense vectors of fixed size.
3.  **LSTM Layer(s):** Captures temporal dependencies and context from the sequence.
4.  **Dense Layer:** A fully connected layer with a **Softmax** activation function that outputs a probability distribution over the vocabulary.

## 🔧 Installation & Usage

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/stym01/Next_word_prediction_using_LSTM.git](https://github.com/stym01/Next_word_prediction_using_LSTM.git)
    cd Next_word_prediction_using_LSTM
    ```

2.  **Install the required dependencies:**
    Ensure you have Python installed. You can install the necessary libraries using pip:
    ```bash
    pip install tensorflow numpy nltk matplotlib notebook
    ```

3.  **Run the Notebook:**
    Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
    Open `next_word_prediction_with_LSTM.ipynb` and run the cells sequentially to train the model and see predictions.

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.




