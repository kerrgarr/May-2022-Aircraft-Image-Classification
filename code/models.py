import torch
import torch.nn.functional as F
from torchvision import models

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        return F.cross_entropy(input, target)
    
##############################################################################################
class VGG16(torch.nn.Module):
    def __init__(self, n_output_channels=6):
        super(VGG16, self).__init__()
        vgg = models.vgg16(pretrained=True)
        num_ftrs = vgg.classifier[0].in_features
        self.feature_extractor = torch.nn.Sequential(*list(vgg.children())[:-1])
        self.classifier = torch.nn.Linear(num_ftrs, n_output_channels)
        
    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.view(x.size(0),-1)
        return self.classifier(x)

    
#############################################################################################    
class ResNet(torch.nn.Module):
    def __init__(self, n_output_channels=6):
        super(ResNet, self).__init__()

        """
        ResNet 152
        """
        resnet = models.resnet152(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = torch.nn.Linear(resnet.fc.in_features, n_output_channels)
        
    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.view(x.size(0),-1)
        return self.classifier(x)

##############################################################################################
def down_conv(in_channels, out_channels, pad):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 3, padding=pad),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.Conv2d(out_channels, out_channels, 3, padding=pad),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(out_channels),
    )   

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.relu    = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, ceil_mode=True)         
        
        a = 32
        b = a*2 #64
        c = b*2 #128
        d = c*2 #256
        m = 12 
        
        self.adapt = torch.nn.AdaptiveMaxPool2d(m)        
        self.linear = torch.nn.Linear(d * m * m, 6)                 
        n_class = 6
        
        self.conv_down1 = down_conv(3, a, 1) # 3 --> 32
        self.conv_down2 = down_conv(a, b, 1)  # 32 --> 64
        self.conv_down3 = down_conv(b, c, 1)  # 64 --> 128
        self.conv_down4 = down_conv(c, d, 1)  # 128 --> 256
    
    def forward(self, x):        
        conv1 =  self.conv_down1(x)
        mx1 = self.maxpool(conv1)
        conv2 =  self.conv_down2(mx1)
        mx2 = self.maxpool(conv2) 
        conv3 =  self.conv_down3(mx2)
        mx3 = self.maxpool(conv3) 
        conv4 =  self.conv_down4(mx3)
        out = self.adapt(conv4)
        out = out.view(x.size(0), -1)
        out = self.linear(out)        
        return out   
    
    
model_factory = {
    'cnn': CNNClassifier,
    'resnet': ResNet,
    'vgg': VGG16,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
