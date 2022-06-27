import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        """
        return F.cross_entropy(input, target)


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Linear Classifier
        """
        self.network = torch.nn.Linear(3 * 64 * 64, 70)

    def forward(self, x):
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,70))
        """
        return self.network(x.view(x.size(0), -1))


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        self.network = torch.nn.Sequential(
            torch.nn.Linear(3 * 64 * 64, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 70),
        )

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        return self.network(x.view(x.size(0), -1))

class CNNClassifier(torch.nn.Module):
    def __init__(self, layers=[16, 32, 64, 128], n_input_channels=3, n_output_channels=70, kernel_size=5):
        super().__init__()

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

model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
    'cnn': CNNClassifier,
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
