import torch

class FullModel(torch.nn.Module):
    def __init__(self, len_input, num_hidden, num_output,
                 conv1=(16, 3, 1), 
                 pool1=(2, 2), 
                 conv2=(32, 4, 2), 
                 pool2=(2, 2), 
                 fc1=256, 
                 num_classes=3):
        super(FullModel, self).__init__()
        
        n = len_input
        # In-channels, Out-channels, Kernel_size, stride ...
        self.conv1 = torch.nn.Conv1d(1, conv1[0], conv1[1], stride=conv1[2])
        n = (n - conv1[1]) // conv1[2] + 1

        self.pool1 = torch.nn.MaxPool1d(pool1[0], stride=pool1[1] )
        n = (n - pool1[0]) // pool1[1] + 1
        
        self.conv2 = torch.nn.Conv1d(conv1[0], conv2[0], conv2[1], stride=conv2[2])
        n = (n - conv2[1]) // conv2[2] + 1
        
        self.pool2 = torch.nn.MaxPool1d(pool2[0], stride=pool2[1] )
        n = (n - pool2[0]) // pool2[1] + 1

        self.relu = torch.nn.LeakyReLU(0.1)
        self.features = torch.nn.Sequential( self.conv1, self.relu, self.pool1, self.conv2, self.relu, self.pool2 )
        self.fc1 = torch.nn.Linear(n*conv2[0], fc1)
        self.fc2 = torch.nn.Linear(fc1, num_classes)        
        self.regression_layer=torch.nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        class_output = self.fc2(x)
        regression_output = self.regression_layer(x)
        return class_output, regression_output
