from model import common
import torch.nn as nn


def make_model(args, parent=False):
    return MRIRSR(args)


class RIRBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, n_resblocks, res_scale, args, conv=common.default_conv):
        super(RIRBlock, self).__init__()
        act = nn.ReLU(True)
        body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*body)
        self.args = args

    def forward(self, x, origin):
        res = self.body(x)
        if self.args.cwo:
            res += origin
        else:
            res += x
        return res, origin


class MultiRIRBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, n_resblocks, res_scale, res_scale_factor, n_rirblocks, args, conv=common.default_conv):
        super(MultiRIRBlock, self).__init__()
        self.rirblocks = []
        for _ in range(n_rirblocks):
            self.rirblocks.append(RIRBlock(n_feats, kernel_size, n_resblocks, res_scale, args, conv))
            res_scale *= res_scale_factor
        self.n_rirblocks = n_rirblocks
        self.body = common.Sequential(*self.rirblocks)

    def forward(self, x):
        y, _ = self.body(x, x)
        return y


class MRIRSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MRIRSR, self).__init__()

        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        self.args = args

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            MultiRIRBlock(n_feats, kernel_size, n_resblocks=args.n_resblocks, res_scale=args.res_scale, res_scale_factor=args.res_scale_factor, n_rirblocks=args.n_rirblocks, args=args, conv=conv)
        ]

        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
