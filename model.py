import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv3x3(in_planes, out_planes, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  padding=dilation, dilation=dilation),
        nn.ReLU(inplace=True))


def conv3x3_down(in_planes, out_planes):
    return nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.MaxPool2d(kernel_size=2, stride=2))


class Encoder(nn.Module):
    def __init__(self, in_planes=6, no_last_down=False):
        super(Encoder, self).__init__()
        self.convs = nn.ModuleList([
            conv3x3_down(in_planes, 32),
            conv3x3_down(32, 64),
            conv3x3_down(64, 128),
            conv3x3_down(128, 256),
            conv3x3_down(256, 512),
            conv3x3_down(512, 1024),
            conv3x3(1024, 2048) if no_last_down else conv3x3_down(1024, 2048)])

    def forward(self, x):
        conv_out = []
        for conv in self.convs:
            x = conv(x)
            conv_out.append(x)
        return conv_out


class Decoder(nn.Module):
    def __init__(self, skip_num=2, out_planes=3,
                 upsample='nearset', no_last_down=False,
                 dilation=1):
        super(Decoder, self).__init__()
        self.convs = nn.ModuleList([
            conv3x3(2048, 1024, 1 if no_last_down else dilation),
            conv3x3(1024 * skip_num, 512, dilation),
            conv3x3(512 * skip_num, 256, dilation),
            conv3x3(256 * skip_num, 128, dilation),
            conv3x3(128 * skip_num, 64, dilation),
            conv3x3(64 * skip_num, 32, dilation)])
        self.last_conv = nn.Conv2d(
            32 * skip_num, out_planes, kernel_size=3,
            padding=dilation, dilation=dilation)
        self.upsample = upsample
        self.no_last_down = no_last_down

    def forward(self, f_list):
        conv_out = []
        f_last = f_list[0]
        first = self.no_last_down
        for conv, f in zip(self.convs, f_list[1:]):
            if first:
                first = False
            else:
                f_last = F.interpolate(f_last, scale_factor=2,
                                       mode=self.upsample)
            f_last = conv(f_last)
            f_last = torch.cat([f_last, f], dim=1)
            conv_out.append(f_last)
        conv_out.append(self.last_conv(F.interpolate(
            f_last, scale_factor=2,
            mode=self.upsample)))
        return conv_out


if __name__ == '__main__':

    encoder = Encoder()
    edg_decoder = Decoder(skip_num=2, out_planes=3)
    cor_decoder = Decoder(skip_num=3, out_planes=1)

    with torch.no_grad():
        x = torch.rand(2, 6, 512, 1024)
        en_list = encoder(x)
        edg_de_list = edg_decoder(en_list[::-1])
        cor_de_list = cor_decoder(en_list[-1:] + edg_de_list[:-1])

    for f in en_list:
        print('encoder', f.size())
    for f in edg_de_list:
        print('edg_decoder', f.size())
    for f in cor_de_list:
        print('cor_decoder', f.size())
