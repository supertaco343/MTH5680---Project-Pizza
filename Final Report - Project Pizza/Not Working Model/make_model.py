import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, input_size//2, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(input_size//2, input_size//4, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(input_size//4, input_size//8, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Calculate the size of the fully connected layer dynamically
        # based on the adjusted input channels and downsampling from convolutional layers
        self.fc_input_size = self.calculate_fc_input_size(input_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.fc_input_size, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 5)

    def calculate_fc_input_size(self, input_size):
        # Function to calculate the size of the fully connected layer
        # based on the adjusted input channels and downsampling from convolutional layers
        dummy_input = torch.ones(1, 3, input_size, input_size)
        dummy_output = self.forward_conv(dummy_input)
        return dummy_output.view(1, -1).size(1)
    
    def forward_conv(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

def make_cnn(input_size, learning_rate):
    model = CNN(input_size)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, loss_function, optimizer