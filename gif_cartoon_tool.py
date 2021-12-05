#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/5 18:10
# @Author  : 剑客阿良_ALiang
# @Site    : 
# @File    : gif_cartoon_tool.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/5 0:26
# @Author  : 剑客阿良_ALiang
# @Site    :
# @File    : video_cartoon_tool.py

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/4 22:34
# @Author  : 剑客阿良_ALiang
# @Site    :
# @File    : image_cartoon_tool.py

from PIL import Image, ImageEnhance, ImageSequence
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch import nn
import os
import torch.nn.functional as F
import uuid
import imageio


# -------------------------- hy add 01 --------------------------
class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))

        # dw
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # pw
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out


class Generator(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.block_a = nn.Sequential(
            ConvNormLReLU(3, 32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(64, 64)
        )

        self.block_b = nn.Sequential(
            ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(128, 128)
        )

        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )

        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64, 64),
            ConvNormLReLU(64, 32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input, align_corners=True):
        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)

        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out


# -------------------------- hy add 02 --------------------------

def handle(gif_path: str, output_dir: str, type: int, device='cpu'):
    _ext = os.path.basename(gif_path).strip().split('.')[-1]
    if type == 1:
        _checkpoint = './weights/paprika.pt'
    elif type == 2:
        _checkpoint = './weights/face_paint_512_v1.pt'
    elif type == 3:
        _checkpoint = './weights/face_paint_512_v2.pt'
    elif type == 4:
        _checkpoint = './weights/celeba_distill.pt'
    else:
        raise Exception('type not support')
    os.makedirs(output_dir, exist_ok=True)
    net = Generator()
    net.load_state_dict(torch.load(_checkpoint, map_location="cpu"))
    net.to(device).eval()
    result = os.path.join(output_dir, '{}.{}'.format(uuid.uuid1().hex, _ext))
    img = Image.open(gif_path)
    out_images = []
    for frame in ImageSequence.Iterator(img):
        frame = frame.convert("RGB")
        with torch.no_grad():
            image = to_tensor(frame).unsqueeze(0) * 2 - 1
            out = net(image.to(device), False).cpu()
            out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
            out = to_pil_image(out)
            out_images.append(out)
    # out_images[0].save(result, save_all=True, loop=True, append_images=out_images[1:], duration=100)
    imageio.mimsave(result, out_images, fps=15)
    return result


if __name__ == '__main__':
    print(handle('samples/gif/128.gif', 'samples/gif_result/', 3, 'cuda'))
