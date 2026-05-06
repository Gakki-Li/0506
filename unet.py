from collections import OrderedDict

import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_batchnorm=False, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_batchnorm=False, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

    def forward(self, x):
        return self.conv(self.pool(x))


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=False, use_batchnorm=False, dropout=0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            )
            up_channels = in_channels // 2
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            up_channels = in_channels // 2
        self.conv = ConvBlock(
            in_channels=up_channels + skip_channels,
            out_channels=out_channels,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = nn.functional.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        init_features=32,
        bilinear=False,
        use_batchnorm=False,
        dropout=0.0,
    ):
        super().__init__()
        features = [
            init_features,
            init_features * 2,
            init_features * 4,
            init_features * 8,
            init_features * 16,
        ]
        self.config = OrderedDict(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            bilinear=bilinear,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

        self.stem = ConvBlock(
            in_channels=in_channels,
            out_channels=features[0],
            use_batchnorm=use_batchnorm,
            dropout=0.0,
        )
        self.down1 = EncoderBlock(features[0], features[1], use_batchnorm=use_batchnorm, dropout=0.0)
        self.down2 = EncoderBlock(features[1], features[2], use_batchnorm=use_batchnorm, dropout=dropout)
        self.down3 = EncoderBlock(features[2], features[3], use_batchnorm=use_batchnorm, dropout=dropout)
        self.down4 = EncoderBlock(features[3], features[4], use_batchnorm=use_batchnorm, dropout=dropout)

        self.up1 = DecoderBlock(features[4], features[3], features[3], bilinear=bilinear, use_batchnorm=use_batchnorm, dropout=dropout)
        self.up2 = DecoderBlock(features[3], features[2], features[2], bilinear=bilinear, use_batchnorm=use_batchnorm, dropout=dropout)
        self.up3 = DecoderBlock(features[2], features[1], features[1], bilinear=bilinear, use_batchnorm=use_batchnorm, dropout=0.0)
        self.up4 = DecoderBlock(features[1], features[0], features[0], bilinear=bilinear, use_batchnorm=use_batchnorm, dropout=0.0)
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.stem(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)
        bottleneck = self.down4(enc4)

        dec1 = self.up1(bottleneck, enc4)
        dec2 = self.up2(dec1, enc3)
        dec3 = self.up3(dec2, enc2)
        dec4 = self.up4(dec3, enc1)

        logits = self.head(dec4)
        return self.activation(logits)
