import torch
from torch import nn
import warnings
warnings.filterwarnings('ignore')
import numpy as np



try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r


    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
        return t.real


def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # Vc = torch.fft.rfft(v, 1, onesided=False)
    Vc = rfft(v, 1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

class facam_block(nn.Module):
    def __init__(self, channel):
        super(facam_block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(902, 902 * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=False),
            nn.Linear(902 * 2, 902, bias=False),
            nn.Sigmoid()
        )

        self.facam_norm = nn.LayerNorm([902], eps=1e-6)


    def forward(self, x):
        b, c, l = x.size()
        list = []
        device = torch.device('cuda:0')

        for i in range(c):
            freq = dct(x[:, i, :]).to(device)
            list.append(freq)

        stack_dct = torch.stack(list, dim=1).to(device)
        stack_dct = torch.tensor(stack_dct).to(device)
        lr_weight = self.facam_norm(stack_dct).to(device)
        lr_weight = self.fc(stack_dct).to(device)
        lr_weight = self.facam_norm(lr_weight).to(device)

        return x * lr_weight


class FA_CTGTrans(nn.Module):
    def __init__(self, configs, hparams):
        super(FA_CTGTrans, self).__init__()
        self.device = torch.device('cuda:0')

        filter_sizes = [5, 9, 11]
        self.conv1 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[0],
                               stride=configs.stride, bias=False, padding=(filter_sizes[0] // 2))
        self.conv2 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[1],
                               stride=configs.stride, bias=False, padding=(filter_sizes[1] // 2))
        self.conv3 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[2],
                               stride=configs.stride, bias=False, padding=(filter_sizes[2] // 2))
        self.bn = nn.BatchNorm1d(configs.mid_channels)

        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.do = nn.Dropout(configs.dropout)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.inplanes = 128
        self.crm = self._make_layer(FACAMBasicBlock, 128, 3)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=configs.trans_dim, nhead=configs.num_heads,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.aap = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(1 * 128, configs.num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_in):
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        x3 = self.conv3(x_in)
        x_concat = torch.mean(torch.stack([x1, x2, x3], 2), 2)
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))

        x = self.conv_block2(x_concat)
        x = self.conv_block3(x)

        x = self.crm(x)

        x1 = self.transformer_encoder(x)
        x2 = self.transformer_encoder(torch.flip(x, [2]))
        x = x1 + x2

        x = self.aap(x)
        x_flat = x.reshape(x.shape[0], -1)
        x_out = self.clf(x_flat)

        return x_out



class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)



class FACAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(FACAMBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.facam = facam_block(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.facam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

