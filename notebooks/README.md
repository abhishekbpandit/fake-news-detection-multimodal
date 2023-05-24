# Deep Learning Model for Fake News Detection

This project involves the development of a multi-input deep learning model for fake news detection, combining text, image and additional feature data. The model is built with TensorFlow and Keras and uses Optuna for hyperparameter tuning.

## Installation

To install the required libraries and dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used in this project consists of text, image and additional features. It's stored in a Google Cloud Storage (GCS) bucket named 'fake-news-data'.

## Model

The model takes three inputs:

Text input: This is passed through an embedding layer and then an LSTM layer.
Image input: This is processed using the MobileNet architecture (excluding the top layer) and then passed through a GlobalAveragePooling2D layer.
Additional features input: This is passed through a dense layer.
The outputs of these three branches are then concatenated and passed through several dense layers and dropout layers, before a final dense layer with softmax activation function which outputs the class probabilities.

## Training

The model is trained using the Adam optimizer and Sparse Categorical Crossentropy as the loss function. A callback for early stopping is added to prevent overfitting.

Hyperparameters of the model such as learning rate, dropout rate, and the number of LSTM and dense units are tuned using Optuna. The model with the best validation accuracy is saved in the GCS bucket.
