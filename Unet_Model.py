import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------
# Depth-wise Separable Convolution
# -------------------
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# -------------------
# SegNet Model
# -------------------
class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SegNet, self).__init__()

        # Encoder
        self.enc1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc1_2 = DepthwiseSeparableConv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc2_1 = DepthwiseSeparableConv2d(64, 128, kernel_size=3, padding=1)
        self.enc2_2 = DepthwiseSeparableConv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc3_1 = DepthwiseSeparableConv2d(128, 256, kernel_size=3, padding=1)
        self.enc3_2 = DepthwiseSeparableConv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc4_1 = DepthwiseSeparableConv2d(256, 512, kernel_size=3, padding=1)
        self.enc4_2 = DepthwiseSeparableConv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Bottleneck
        self.bottleneck = DepthwiseSeparableConv2d(512, 512, kernel_size=3, padding=1)

        # Decoder
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec4_1 = DepthwiseSeparableConv2d(512, 256, kernel_size=3, padding=1)
        self.dec4_2 = DepthwiseSeparableConv2d(256, 256, kernel_size=3, padding=1)

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec3_1 = DepthwiseSeparableConv2d(256, 128, kernel_size=3, padding=1)
        self.dec3_2 = DepthwiseSeparableConv2d(128, 128, kernel_size=3, padding=1)

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec2_1 = DepthwiseSeparableConv2d(128, 64, kernel_size=3, padding=1)
        self.dec2_2 = DepthwiseSeparableConv2d(64, 64, kernel_size=3, padding=1)

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec1_1 = DepthwiseSeparableConv2d(64, 64, kernel_size=3, padding=1)
        self.dec1_2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.enc1_1(x))
        x = F.relu(self.enc1_2(x))
        x, idx1 = self.pool1(x)

        x = F.relu(self.enc2_1(x))
        x = F.relu(self.enc2_2(x))
        x, idx2 = self.pool2(x)

        x = F.relu(self.enc3_1(x))
        x = F.relu(self.enc3_2(x))
        x, idx3 = self.pool3(x)

        x = F.relu(self.enc4_1(x))
        x = F.relu(self.enc4_2(x))
        x, idx4 = self.pool4(x)

        x = F.relu(self.bottleneck(x))

        x = self.unpool4(x, idx4)
        x = F.relu(self.dec4_1(x))
        x = F.relu(self.dec4_2(x))

        x = self.unpool3(x, idx3)
        x = F.relu(self.dec3_1(x))
        x = F.relu(self.dec3_2(x))

        x = self.unpool2(x, idx2)
        x = F.relu(self.dec2_1(x))
        x = F.relu(self.dec2_2(x))

        x = self.unpool1(x, idx1)
        x = F.relu(self.dec1_1(x))
        x = self.dec1_2(x)

        return x
