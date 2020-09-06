import torch
import torch.nn as nn

class Dense_Block(nn.Module):
    def __init__(self, in_channels, block_size=32):

        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_channels=in_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=block_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=block_size, out_channels=block_size, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=block_size * 2, out_channels=block_size, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=block_size * 3, out_channels=block_size, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=block_size * 4, out_channels=block_size, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        # Concatenate in channel dimension
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))

        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return c5_dense