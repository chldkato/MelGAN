import torch.nn as nn
from torch.nn.utils import weight_norm


class residual_stack(nn.Module):
    def __init__(self, size, dilation):
        super().__init__()

        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            weight_norm(nn.Conv1d(size, size, kernel_size=3, dilation=dilation)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(size, size, kernel_size=1))
        )
        self.shortcut = weight_norm(nn.Conv1d(size, size, kernel_size=1))

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


def encoder_sequential(input_size, output_size, *args, **kwargs):
    return nn.Sequential(
        nn.LeakyReLU(0.2),
        weight_norm((nn.ConvTranspose1d(input_size, output_size, *args, **kwargs)))
    )


class Generator(nn.Module):
    def __init__(self, mel_dim):
        super().__init__()

        factor = [8, 8, 2, 2]

        layers = [
            nn.ReflectionPad1d(3), # 3+80+3 = 86
            weight_norm(nn.Conv1d(mel_dim, 512, kernel_size=7)),
        ]

        input_size = 512
        for f in factor:
            layers += [encoder_sequential(input_size,
                                          input_size // 2,
                                          kernel_size=f * 2,
                                          stride=f,
                                          padding=f // 2 + f % 2)]
            input_size //= 2
            for d in range(3):
                layers += [residual_stack(input_size, 3 ** d)]

        layers += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            weight_norm(nn.Conv1d(32, 1, kernel_size=7)),
            nn.Tanh(),
        ]

        self.generator = nn.Sequential(*layers)

    def forward(self, x):
        return self.generator(x)


def decoder_sequential(input_size, output_size, *args, **kwargs):
    return nn.Sequential(
        weight_norm((nn.Conv1d(input_size, output_size, *args, **kwargs))),
        nn.LeakyReLU(0.2, inplace=True)
    )


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # extract output from each module
        self.discriminator = nn.ModuleList([
            # Feature map x 1
            nn.Sequential(
                nn.ReflectionPad1d(7), # 7+1+7 = 15
                weight_norm(nn.Conv1d(1, 16, kernel_size=15)),
                nn.LeakyReLU(0.2, inplace=True) # modify the input
            ),
            # Downsampling layer Feature map x 4
            decoder_sequential(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
            decoder_sequential(64, 256, kernel_size=41, stride=4, padding=20, groups=16),
            decoder_sequential(256, 1024, kernel_size=41, stride=4, padding=20, groups=64),
            decoder_sequential(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256),
            # Feature map x 1
            nn.Sequential(
                weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, padding=2)),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Output x 1
            weight_norm(nn.Conv1d(1024, 1, kernel_size=3, padding=1))
        ])

    def forward(self, x):
        feature_map = []
        for module in self.discriminator:
            x = module(x)
            feature_map.append(x)
        return feature_map


class MultiScale(nn.Module):
    def __init__(self):
        super().__init__()

        # extract output from each discriminator block
        self.block = nn.ModuleList([
            Discriminator() for _ in range(3)
        ])

        self.avgpool = nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        result = []
        for idx, module in enumerate(self.block):
            result.append(module(x))
            if idx <= 1:
                x = self.avgpool(x)
        return result
