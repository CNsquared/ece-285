
#Has the overall structure of the model

import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO: RESNET WILL TEST THIS AFTER SIMPLE CNN
def conv3x3(in_planes, out_planes, stride=1):
    """3×3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1  # how many times we expand the channel count
    
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        """
        in_planes:   # input channels
        planes:      # output channels before expansion
        stride:      # stride on the first conv (for spatial downsampling)
        downsample:  # optional nn.Sequential to match dimensions of identity path
        """
        super().__init__()
        # First conv-BN-ReLU
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        
        # Second conv-BN (no ReLU here, since we add identity first)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        
        # If the input & output shapes differ (in channels or spatial size), 
        # we apply downsampling to the identity branch:
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        # First layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second layer
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Match dimensions on the skip path if needed:
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip (identity) and residual
        out += identity
        out = self.relu(out)
        
        return out
class SimpleResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        """
        block:       class of block to use (e.g. BasicBlock)
        layers:      list of 4 integers, number of blocks in each of the 4 layers
        num_classes: number of output classes
        """
        super().__init__()
        self.in_planes = 64
        
        # Initial convolution + BN + ReLU + max-pool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build the 4 ResNet layers
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final pooling and fully-connected Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights as in original ResNet
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def _make_layer(self, block, planes, blocks, stride):
        """
        planes:   number of channels in this stage
        blocks:   how many blocks to stack
        stride:   stride for the *first* block in this layer
        """
        downsample = None
        # If shape changes, create downsample module
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        # First block may downsample
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        
        # Remaining blocks keep stride=1
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Pool & classifier
        x = self.avgpool(x)               # shape: (batch, 512, 1, 1)
        x = torch.flatten(x, 1)           # shape: (batch, 512)
        x = self.fc(x)                    # shape: (batch, num_classes)
        return x

class simpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_dim=(1, 32, 32)):
        super(simpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim[0], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * input_dim[1] * input_dim[2] // 4, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        embedding = F.max_pool2d(x, 2)
        c = embedding.view(embedding.size(0), -1)  # Flatten
        classification = self.fc1(c)
        softmax = F.softmax(classification, dim=1)
        return softmax, embedding
    
    
    
class simleGAN(nn.Module):
    def __init__(self):
        super(simleGAN, self).__init__()
        pass

    def forward(self, classification, embedding):
        #GAN that takes the classification and embedding as conditions
        pass
    
class simpleOverallModel(nn.Module):
    def __init__(self, num_classes=10, input_dim=(1, 32, 32)):
        super(simpleOverallModel, self).__init__()
        self.cnn = SimpleResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.gan = simleGAN()

    def forward(self, x):
        classification, embedding = self.cnn(x)
        x = self.gan(classification, embedding)
        return x