import torch
import torch.nn.functional as F
from torchvision import models

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        """
        return F.cross_entropy(input, target)


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

class CNNClassifier1(torch.nn.Module):
    def __init__(self, layers=[16, 32, 64, 128], n_input_channels=3, n_output_channels=6, kernel_size=5):
        super().__init__()

        """
        Simple CNN
        """
        L = []
        c = n_input_channels
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size, stride=2, padding=kernel_size//2))
            L.append(torch.nn.ReLU())
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_output_channels)

    def forward(self, x):
        return self.classifier(self.network(x).mean(dim=[2, 3]))

    
class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()

            """
            Deeper CNN
            """
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, bias=False)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=6, kernel_size=3):
        super().__init__()
        self.input_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.input_std = torch.Tensor([0.229, 0.224, 0.2254])

        L = []
        c = 3
        for l in layers:
            L.append(self.Block(c, l, kernel_size, 2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_output_channels)

    def forward(self, x):
        z = self.network((x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device))
        return self.classifier(z.mean(dim=[2, 3]))    
    
    
    
    
    
model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
    'cnn': CNNClassifier,
    'resnet': ResNet,
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
