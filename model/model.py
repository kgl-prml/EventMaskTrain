import torch
import torch.nn as nn
from .utils import weights_init_he

class _bn_relu(nn.Module):
    def __init__(self, conv, conv_out_channels):
        super(_bn_relu, self).__init__()
        self._conv = conv
        #self.bn = nn.BatchNorm3d(conv_out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._conv(x)
        #x = self.bn(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = _bn_relu(nn.Conv3d(3, 64, (3, 3, 3), padding=(1, 1, 1)), 64)
        self.maxpool1 = nn.MaxPool3d((1, 2, 2))

        self.conv2 = _bn_relu(nn.Conv3d(64, 128, (3, 3, 3), padding=(1, 1, 1)), 128)
        self.maxpool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3a = _bn_relu(nn.Conv3d(128, 256, (3, 3, 3), padding=(1, 1, 1)), 256)
        self.conv3b = _bn_relu(nn.Conv3d(256, 256, (3, 3, 3), padding=(1, 1, 1)), 256)
        self.maxpool3 = self.maxpool2

        self.conv4a = _bn_relu(nn.Conv3d(256, 512, (3, 3, 3), padding=(1, 1, 1)), 512)
        self.conv4b = _bn_relu(nn.Conv3d(512, 512, (3, 3, 3), padding=(1, 1, 1)), 512)
        self.maxpool4 = self.maxpool2

        self.conv5a = _bn_relu(nn.Conv3d(512, 512, (3, 3, 3), padding=(1, 1, 1)), 512)
        self.conv5b = _bn_relu(nn.Conv3d(512, 512, (3, 3, 3), padding=(1, 1, 1)), 512)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.maxpool1(x1)

        x2 = self.conv2(x)
        x = self.maxpool2(x2)

        x = self.conv3a(x)
        x3 = self.conv3b(x)
        x = self.maxpool3(x3)

        x = self.conv4a(x)
        x4 = self.conv4b(x)
        x = self.maxpool4(x4)

        x = self.conv5a(x)
        x = self.conv5b(x)
        #print('Encoder final output shape: ', x.size())
        return x, x1, x2, x3, x4

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv4 = _bn_relu(nn.ConvTranspose3d(512, 64, (2, 2, 2), stride=(2, 2, 2)), 64)
        self.conv4c = _bn_relu(nn.Conv3d(512, 448, (3, 3, 3), padding=(1, 1, 1)), 448)
        self.deconv3 = _bn_relu(nn.ConvTranspose3d(512, 64, (2, 2, 2), stride=(2, 2, 2)), 64)
        self.conv3c = _bn_relu(nn.Conv3d(256, 448, (3, 3, 3), padding=(1, 1, 1)), 448)
        self.deconv2 = _bn_relu(nn.ConvTranspose3d(512, 64, (2, 2, 2), stride=(2, 2, 2)), 64)
        self.conv2c = _bn_relu(nn.Conv3d(128, 128, (3, 3, 3), padding=(1, 1, 1)), 128)
        self.deconv1 = _bn_relu(nn.ConvTranspose3d(192, 48, (1, 2, 2), stride=(1, 2, 2)), 48)
        self.conv1c = _bn_relu(nn.Conv3d(64, 64, (3, 3, 3), padding=(1, 1, 1)), 64)

        self.conv6 = _bn_relu(nn.Conv3d(112, 512, (1, 1, 1)), 512)
        #self.conv6_aux = _bn_relu(nn.Conv3d(192, 2, (1, 1, 1)), 512)
        self.conv7 = nn.Conv3d(512, 2, (1, 1, 1))

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
        #out_aux = self.conv6_aux(out_aux)
        #out_aux = self.conv7(out_aux)

        out1_0 = self.deconv1(out2)
        out1_1 = self.conv1c(x1)
        out1 = torch.cat((out1_0, out1_1), dim=1)

        x = self.conv6(out1)
        x = self.conv7(x)
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
        return x

def get_MaskGenNet(state_dict=None):
    net = MaskGenNet()
    if state_dict is not None:
        net.load_state_dict(state_dict)
    else:
        net.apply(weights_init_he)

    state_dict = torch.load('c3d.pickle')
    new_state_dict = {}
    new_state_dict['conv1._conv.weight'] = state_dict['conv1.weight']
    new_state_dict['conv1._conv.bias'] = state_dict['conv1.bias']

    new_state_dict['conv2._conv.weight'] = state_dict['conv2.weight']
    new_state_dict['conv2._conv.bias'] = state_dict['conv2.bias']

    new_state_dict['conv3a._conv.weight'] = state_dict['conv3a.weight']
    new_state_dict['conv3a._conv.bias'] = state_dict['conv3a.bias']
    new_state_dict['conv3b._conv.weight'] = state_dict['conv3b.weight']
    new_state_dict['conv3b._conv.bias'] = state_dict['conv3b.bias']

    new_state_dict['conv4a._conv.weight'] = state_dict['conv4a.weight']
    new_state_dict['conv4a._conv.bias'] = state_dict['conv4a.bias']
    new_state_dict['conv4b._conv.weight'] = state_dict['conv4b.weight']
    new_state_dict['conv4b._conv.bias'] = state_dict['conv4b.bias']

    new_state_dict['conv5a._conv.weight'] = state_dict['conv5a.weight']
    new_state_dict['conv5a._conv.bias'] = state_dict['conv5a.bias']
    new_state_dict['conv5b._conv.weight'] = state_dict['conv5b.weight']
    new_state_dict['conv5b._conv.bias'] = state_dict['conv5b.bias']

    net.encoder.load_state_dict(new_state_dict)
    return net
