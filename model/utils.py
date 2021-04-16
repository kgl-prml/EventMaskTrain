import torch
from torch.nn import init

def weights_init_he(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if 'weight' in m.state_dict().keys():
            m.weight.data.normal_(1.0, 0.02)
        if 'bias' in m.state_dict().keys():
            m.bias.data.fill_(0)
    else:
        if 'weight' in m.state_dict().keys():
            init.kaiming_normal_(m.weight)
        if 'bias' in m.state_dict().keys():
            m.bias.data.fill_(0)


