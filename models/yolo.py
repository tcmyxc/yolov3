import argparse
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import Conv, Bottleneck, SPP, Concat, autoShape
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import fuse_conv_and_bn, model_info, initialize_weights, select_device, copy_attr


class Detect(nn.Module):
    """检测头, 一般3个"""
    stride = None  # strides computed during build

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        # nc = 80,
        # anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        # ch = [256, 512, 1024]]
        super(Detect, self).__init__()
        self.nc = nc  # number of classes, 80, 类别数量
        self.no = nc + 5  # number of outputs per anchor, 85, 每个anchor的输出: 是否有感兴趣的物体+四个坐标+每个类别的概率
        self.nl = len(anchors)  # number of detection layers, 3, 检测层的数量(每一层都有对应匹配的anchor)
        self.na = len(anchors[0]) // 2  # number of anchors, 3, 一般是3个anchor
        self.grid = [torch.zeros(1)] * self.nl  # 初始化网格
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)  # 多加了一个维度, (3, 3, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)  # 注册到模型自身缓存（不会被更新）
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        # 下面是三个检测头
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # 推理阶段用到的输出
        for i in range(self.nl):
            # x[i] 是预测头的输出, 需要reshape
            x[i] = self.m[i](x[i])  # 把输入送到对应的卷积层，得到对应的输出
            bs, _, ny, nx = x[i].shape  # x(bs, 255, h, w)
            # bs, 3, 85, h, w -> bs, 3, h, w, 85; contiguous 变成内存连续的变量
            # 把特征图摞起来，纵向有255个点，对应3个锚框，每个点是锚框的一个预测信息
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # 如果是推理模式
                # x[i].shape[2:4] -> h, w
                # grid[i].shape[2:4]最开始的值是torch.Size([])
                # 如果预设的网格大小和特征图大小不匹配，则重新设置网格
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)  # 构造网格

                y = x[i].sigmoid()
                # 下面这部分变换和原始yolov3不一样，但是都是把预测出现的坐标信息映射到原始图片上
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        # xv, yv 对应每个网格左上角的横纵坐标
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()  # 多加了两个维度


class Model(nn.Module):
    def __init__(self, cfg='yolov3.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            logger.info('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value, 类别数

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist(创建模型)

        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names, 数字形式
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect(), 这个部分是三个检测头
        if isinstance(m, Detect):
            # 下面两行是计算下采样倍数 ==> (8, 16, 32), 小anchor对应大特征图
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward

            m.anchors /= m.stride.view(-1, 1, 1)  # 将锚框映射到特征图
            check_anchor_order(m)  # 检查anchor顺序和缩放比例的顺序对不对, 都是递增顺序
            self.stride = m.stride
            self._initialize_biases()  # only run once/
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x):
        # y 收集需要保存中间结果那些模块的输出
        # 需要保存就保存, 不需要就直接赋值 None
        y = []
        for m in self.model:
            if m.f != -1:  # 如果输入不是来自上一层
                # 获取当前模块的实际输入到底是什么
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            # .data 和 detach() 方法区别: https://www.jb51.net/article/177918.htm
            b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors, anchor数量
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5), 输出的数量

    # layers, savelist: 输出保存层的列表, ch out: 记录每一个模块输出channel数量
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        # from: 输入从哪一层来的, number: 本模块的数量, module: 模块名, args: 模块参数
        m = eval(m) if isinstance(m, str) else m  # eval strings, 解析 m 属于哪个具体的类
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain, 获取模块的深度(层数)

        if m in [Conv, Bottleneck, SPP]:
            c1, c2 = ch[f], args[0]  # 输入输出的channel

            # 如果输出通道的数量不等于检测头输出，则输出通道变成8的倍数。
            # 如果是最后的检测头，什么都不做
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]  # [out_channel, ...] -> [in_channel, out_channel, ...]
        elif m is Concat:
            # 如果需要拼接, 则输出channel应该是这些层channel数量的总和
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])  # f:[-1, 8]
        elif m is Detect:
            # args放的是nc和anchors的信息。之后再追加输出通道维度的信息
            # [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]]
            # -> [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 512, 1024]]
            args.append([ch[x + 1] for x in f])  # 加1是因为最开始有一个输入通道3
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        # module, 构造模块
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)

        # 下面是往模块里面添加属性的操作
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print

        # ([f] if isinstance(f, int) else f) 转list
        # x % i 是一个很神奇的操作, 涉及了负数取余,  (-2)%16 -> 14
        # 这里保存的都是需要存储中间结果的层数
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov3.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()
    # print(model.info())
    # Profile
    img = torch.rand(1, 3, 640, 640).to(device)
    y = model(img)

