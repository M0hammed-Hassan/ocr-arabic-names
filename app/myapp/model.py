import torch
import torch.nn as nn


class CRNNModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        """
        Initialize the CRNN model for OCR tasks.

        The model consists of a Convolutional Neural Network (CNN) for feature extraction, followed by a Long Short-Term Memory (LSTM)
        recurrent neural network for sequence modeling, and a fully connected layer for classification.

        The CNN consists of two convolutional layers with ReLU activation and max pooling, followed by another two convolutional layers
        with ReLU activation and max pooling. The output of the CNN is reshaped and fed into the LSTM.

        The LSTM has 128 * 8 input features, 256 hidden units, and is bidirectional. The LSTM outputs are then passed through a fully connected
        layer with num_classes output units.

        The model is initialized with the provided num_classes and the defined CNN, LSTM, and fully connected layers.

        Parameters:
            num_classes (int): The number of output classes for the model. This is typically the number of unique characters in the dataset.
        """
        super(CRNNModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.rnn = nn.LSTM(128 * 24, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the CRNN model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, height, width, channels).
            The input tensor represents the preprocessed images for OCR.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, num_classes).
            The output tensor represents the predicted character probabilities for each time step in the sequence.
        """
        x = self.cnn(x)
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), x.size(2), -1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
