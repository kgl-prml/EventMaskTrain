import torch
import torch.nn as nn
from .utils import weights_init_he
from .i3d import InceptionI3d as Encoder

key_mapping = {}

class _bn_relu(nn.Module):
    def __init__(self, conv, conv_out_channels):
        super(_bn_relu, self).__init__()
        self._conv = conv
        self.bn = nn.BatchNorm3d(conv_out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv4 = _bn_relu(nn.ConvTranspose3d(1024, 64, (2, 2, 2), stride=(2, 2, 2)), 64)
        self.conv4c = _bn_relu(nn.Conv3d(832, 448, (3, 3, 3), padding=(1, 1, 1)), 448)
        self.deconv3 = _bn_relu(nn.ConvTranspose3d(512, 64, (2, 2, 2), stride=(2, 2, 2)), 64)
        self.conv3c = _bn_relu(nn.Conv3d(480, 448, (3, 3, 3), padding=(1, 1, 1)), 448)
        self.deconv2 = _bn_relu(nn.ConvTranspose3d(512, 64, (1, 2, 2), stride=(1, 2, 2)), 64)
        self.conv2c = _bn_relu(nn.Conv3d(192, 128, (3, 3, 3), padding=(1, 1, 1)), 128)
        self.deconv1 = _bn_relu(nn.ConvTranspose3d(192, 48, (1, 2, 2), stride=(1, 2, 2)), 48)
        self.conv1c = _bn_relu(nn.Conv3d(64, 64, (3, 3, 3), padding=(1, 1, 1)), 64)

        #self.conv1x1a_aux = _bn_relu(nn.Conv3d(192, 512, (1, 1, 1)), 512)
        #self.conv1x1b_aux = nn.Conv3d(512, 2, (1, 1, 1))

        self.deconv0 = _bn_relu(nn.ConvTranspose3d(112, 512, (2, 2, 2), stride=(2, 2, 2)), 512)
        # TODO
        #self.conv1x1a = _bn_relu(nn.Conv3d(512, 512, (1, 1, 1)), 512)
        self.conv1x1a = nn.Conv3d(512, 512, (1, 1, 1))
        self.conv1x1b = nn.Conv3d(512, 2, (1, 1, 1))

    def forward(self, x, x4, x3, x2, x1):
        out4_0 = self.deconv4(x)
        out4_1 = self.conv4c(x4)
        out4 = torch.cat((out4_0, out4_1), dim=1)

        out3_0 = self.deconv3(out4)
        out3_1 = self.conv3c(x3)
        out3 = torch.cat((out3_0, out3_1), dim=1)

        out2_0 = self.deconv2(out3)
        out2_1 = self.conv2c(x2)
        out2 = torch.cat((out2_0, out2_1), dim=1)
        #out_aux = nn.functional.interpolate(out2, scale_factor=(1, 2, 2))
        #out_aux = self.conv1x1a_aux(out2)
        #out_aux = self.conv1x1b_aux(out_aux)

        out1_0 = self.deconv1(out2)
        out1_1 = self.conv1c(x1)
        out1 = torch.cat((out1_0, out1_1), dim=1)

        x = self.deconv0(out1)
        x = self.conv1x1a(x)
        x = self.conv1x1b(x)
        #print('Decoder final output shape:', x.size())
        return x #, out_aux

class MaskGenNet(nn.Module):
    def __init__(self):
        super(MaskGenNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x, x1, x2, x3, x4 = self.encoder(x)
        #x, out_aux = self.decoder(x, x4, x3, x2, x1)
        x = self.decoder(x, x4, x3, x2, x1)
        #x = torch.sigmoid(x)
        return x

def get_MaskGenNet(state_dict=None):
    net = MaskGenNet()
    if state_dict is not None:
        net.load_state_dict(state_dict)
    else:
        net.apply(weights_init_he)

        state_dict = torch.load('./rgb_charades.pt')
        new_state_dict = {}
        if len(key_mapping) == 0:
            new_state_dict = state_dict
        else:
            for key in key_mapping:
                new_state_dict[key] = state_dict[key_mapping[key]]

        net.encoder.load_state_dict(new_state_dict)
    return net
