import torch
import torch.nn.functional as F

def set_param_groups(net, lr_mult_dict):
    params = []
    modules = net.module._modules
    for name in modules:
        module = modules[name]
        if name in lr_mult_dict:
            params += [{'params': module.parameters(), 'lr_mult': lr_mult_dict[name]}]
        else:
            params += [{'params': module.parameters(), 'lr_mult': 1.0}]

    return params

def adjust_learning_rate_exp(lr, optimizer, iters, decay_rate=0.1, decay_step=25):
    lr = lr * (decay_rate ** (iters // decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def adjust_learning_rate_RevGrad(lr, optimizer, max_iter, cur_iter, alpha=10, beta=0.75):
    p = 1.0 * cur_iter / (max_iter - 1)
    lr = lr / pow(1.0 + alpha * p, beta)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def adjust_learning_rate_inv(lr, optimizer, iters, alpha=0.001, beta=0.75):
    lr = lr / pow(1.0 + alpha * iters, beta)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def IoU(x, target):
    eps = 1e-5
    intersection = torch.sum(x * target, dim=1)
    union = torch.sum(x, dim=1) + torch.sum(target, dim=1) - intersection
    iou = torch.mean((intersection + eps) / (union + eps))
    return iou

def dice_loss(probs, target):
    # pos: N x D x H x W
    N = probs.size(0)
    pos = probs[:, 1, :, :, :].view(N, -1)
    neg = 1.0 - pos

    if target.is_cuda:
        target = target.type(torch.cuda.FloatTensor)
    else:
        target = target.type(torch.FloatTensor)

    target = target.view(N, -1)
    pos_iou = IoU(pos, target)
    #neg_iou = IoU(neg, target)
    return 1.0 - pos_iou #- neg_iou

def BF_loss(probs, target):
    # get the foreground probs
    probs = probs.narrow(1, 1, 1).squeeze(1)
    # get the boundary
    N, D, H, W = probs.size()
    assert(probs.size() == target.size())
    invert_probs = 1.0 - probs.reshape(-1, H, W)
    invert_target = 1.0 - target.reshape(-1, H, W).float()
    pb = F.max_pool2d(invert_probs, kernel_size=(3, 3), stride=1, padding=1) - invert_probs
    gb = F.max_pool2d(invert_target, kernel_size=(3, 3), stride=1, padding=1) - invert_target

    # get the extended boundary
    epb = F.max_pool2d(pb, kernel_size=(3, 3), stride=1, padding=1)
    egb = F.max_pool2d(gb, kernel_size=(3, 3), stride=1, padding=1)

    # compute the precision
    pb = pb.reshape(N, D, H, W).reshape(N, -1)
    gb = gb.reshape(N, D, H, W).reshape(N, -1)
    epb = epb.reshape(N, D, H, W).reshape(N, -1)
    egb = egb.reshape(N, D, H, W).reshape(N, -1)

    eps = 1e-5
    prec = (torch.sum(pb * egb, dim=1) + eps) / (torch.sum(pb) + eps)
    recall = (torch.sum(gb * epb, dim=1) + eps) / (torch.sum(gb) + eps)

    metric = 2 * prec * recall / (prec + recall)
    loss = 1.0 - torch.mean(metric)
    return loss




