import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from . import utils as solver_utils
from utils.utils import to_cuda
import utils.utils as gen_utils
from torch import optim
from config.config import cfg
#from time import time

class Solver:
    def __init__(self, net, dataloader, resume=None, **kwargs):
        self.opt = cfg
        self.net = net
        self.init_data(dataloader)

        self.CEWeight = to_cuda(torch.tensor([1.0 - cfg.TRAIN.WPOS, cfg.TRAIN.WPOS]))
        self.CELoss = nn.CrossEntropyLoss(weight=self.CEWeight)
        self.BCELoss = nn.BCELoss()
        if torch.cuda.is_available():
            self.CELoss.cuda()
            self.BCELoss.cuda()

        self.iters = 0
        self.epochs = 0
        self.iters_per_epoch = None

        self.base_lr = self.opt.TRAIN.BASE_LR 
        self.momentum = self.opt.TRAIN.MOMENTUM
        self.optim_state_dict = None

        self.resume = False
        if resume is not None:
            self.resume = True
            self.epochs = resume['epochs']
            self.iters = resume['iters']
            self.optim_state_dict = resume['optimizer_state_dict']
            print('Resume Training from iters %d, %d.' % \
                     (self.epochs, self.iters))

        self.build_optimizer()

    def init_data(self, dataloader):
        self.train_data = {}
        self.test_data = {}
        self.train_data['loader'] = dataloader['train']
        self.test_data['loader'] = dataloader['test']
        self.train_data['iterator'] = iter(self.train_data['loader'])
        #self.test_data['iterator'] = iter(self.test_data['loader]'])

    def build_optimizer(self):
        opt = self.opt
        param_groups = solver_utils.set_param_groups(self.net, {'decoder': opt.TRAIN.LR_MULT})

        assert opt.TRAIN.OPTIMIZER in ["Adam", "SGD"], \
            "Currently do not support your specified optimizer."

        if opt.TRAIN.OPTIMIZER == "Adam":
            self.optimizer = optim.Adam(param_groups,
                        lr=self.base_lr, betas=[opt.ADAM.BETA1, opt.ADAM.BETA2],
                        weight_decay=opt.TRAIN.WEIGHT_DECAY)

        elif opt.TRAIN.OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(param_groups,
                        lr=self.base_lr, momentum=self.momentum,
                        weight_decay=opt.TRAIN.WEIGHT_DECAY)

        if self.optim_state_dict is not None:
            self.optimizer.load_state_dict(self.optim_state_dict)

    def test(self):
        vout_all, vlabel_all = [], []
        self.net.eval()
        for sample in iter(self.test_data['loader']):
            _, _, vclip, vlabel = sample
            vclip = to_cuda(vclip)
            vlabel = to_cuda(vlabel)
            vout = self.net(vclip)
            vout_all += [vout]
            vlabel_all += [vlabel]

        vout_all = torch.cat(vout_all, dim=0)
        vlabel_all = torch.cat(vlabel_all, dim=0)
        iou, iou_pos, accu = self.model_eval(vout_all, vlabel_all)
        return iou, iou_pos, accu

    def get_samples(self):
        assert('loader' in self.train_data and \
               'iterator' in self.train_data)

        data_loader = self.train_data['loader']
        data_iterator = self.train_data['iterator']
        assert data_loader is not None and data_iterator is not None, \
            'Check your training dataloader.'

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data['iterator'] = data_iterator
        return sample

    def update_lr(self):
        iters = self.iters
        if self.opt.TRAIN.LR_SCHEDULE == 'exp':
            solver_utils.adjust_learning_rate_exp(self.base_lr,
                        self.optimizer, iters,
                        decay_rate=self.opt.EXP.LR_DECAY_RATE,
                        decay_step=self.opt.EXP.LR_DECAY_STEP)

        elif self.opt.TRAIN.LR_SCHEDULE == 'inv':
            solver_utils.adjust_learning_rate_inv(self.base_lr, self.optimizer,
                    iters, self.opt.INV.ALPHA, self.opt.INV.BETA)

        else:
            raise NotImplementedError("Currently don't support the specified \
                    learning rate schedule: %s." % self.opt.TRAIN.LR_SCHEDULE)

    def model_eval(self, preds, gts):
        # 1) IoU
        iou = gen_utils.iou(preds, gts)
        iou_pos = gen_utils.iou(preds, gts, True)
        # TODO: 2) Recall
        # 3) pixel-level accuracy (may not be suitable)
        accuracy = gen_utils.accuracy(preds, gts)
        return iou, iou_pos, accuracy
        
    def save_ckpt(self):
        save_path = self.opt.SAVE_DIR
        ckpt_resume = os.path.join(save_path, 'ckpt_%d_%d.resume' % (
                                   self.epochs, self.iters))
        ckpt_weights = os.path.join(save_path, 'ckpt_%d_%d.weights' % (
                                   self.epochs, self.iters))

        torch.save({'epochs': self.epochs,
                    'iters': self.iters,
                    'model_state_dict': self.net.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, ckpt_resume)

        torch.save({'parameters': self.net.module.state_dict()}, ckpt_weights)

    def solve(self):
        if self.resume:
            self.iters += 1
            self.epochs += 1

        self.compute_iters_per_epoch()
        print('Start Training.')
        while True:
            if self.epochs > self.opt.TRAIN.MAX_EPOCHS: 
                break
            self.update_network()
            self.epochs += 1
        print('Training Done!')

    def compute_iters_per_epoch(self):
        self.iters_per_epoch = len(self.train_data['loader'])
        print('Iterations per epoch: %d' % self.iters_per_epoch)

    def update_network(self):
        stop = False
        update_iters = 0

        while not stop:
            self.net.train()
            self.net.zero_grad()
          
            if self.opt.TRAIN.OPTIMIZER != "Adam":
                self.update_lr()

            # get the video clip and corresponding mask
            #start = time()
            _, _, vclip, vlabel = self.get_samples()
            #end = time()
            #print('Time: %f' % (end-start))
            vclip = to_cuda(vclip)
            vlabel = to_cuda(vlabel)
            # forward and get the predictions
            # N x C x D x H x W
            #vout, vout_aux = self.net(vclip)
            vout = self.net(vclip)
            vprobs = F.softmax(vout, dim=1)

            #vout_aux = F.interpolate(vout_aux, scale_factor=(2, 4, 4))
            #vprobs_aux = F.softmax(vout_aux, dim=1)
            ## reshape and compute the cross-entropy loss
            #ch = vout.size(1)
            #vpreds = vout.transpose(0, 1).reshape(ch, -1).transpose(0, 1).squeeze(-1)
            ##vout0 = F.interpolate(vout0, scale_factor=(1, 2, 2))
            ##small_vpreds = vout0.transpose(0, 1).reshape(ch, -1).transpose(0, 1)

            #vgt = vlabel.view(-1)
            #loss = self.CELoss(vpreds, vgt) #self.BCELoss(vpreds, vgt)
            alpha = 0.3 
            loss = (1.0 - alpha) * solver_utils.dice_loss(vprobs, vlabel)
            loss += alpha * solver_utils.BF_loss(vprobs, vlabel)

            #loss_aux = (1.0 - alpha) * solver_utils.dice_loss(vprobs_aux, vlabel)
            #loss_aux += alpha * solver_utils.BF_loss(vprobs_aux, vlabel)

            #beta = 0.5
            #loss = beta * loss + (1.0 - beta) * loss_aux

            # downsample the mask by scale 2
            #small_mask = F.interpolate(vlabel.type(torch.cuda.FloatTensor), scale_factor=(0.5, 0.5))
            #small_vgt = small_mask.view(-1).type(torch.cuda.LongTensor)
            #loss += self.CELoss(small_vpreds, vgt)
            
            loss.backward()
            self.optimizer.step()

            if self.opt.TRAIN.LOGGING and (update_iters+1) % \
                      (max(1, self.iters_per_epoch // 
                      self.opt.TRAIN.NUM_LOGGING_PER_EPOCH)) == 0:

                iou, iou_pos, accuracy = self.model_eval(vout, vlabel)
                print('Training at (epoch %d, iters: %d) with loss, iou, iou_pos, accuracy: %.4f, %.4f, %.4f, %.4f.' % (
                      self.epochs, self.iters, loss, iou, iou_pos, accuracy))

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
                (self.iters+1) % int(self.opt.TRAIN.TEST_INTERVAL * 
                self.iters_per_epoch) == 0:

                with torch.no_grad():
                    iou, iou_pos, accuracy = self.test()
                    print('Test at (epoch %d, iters: %d): %.4f, %.4f, %.4f.' % (self.epochs,
                              self.iters, iou, iou_pos, accuracy))

            if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
                (self.iters+1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * 
                self.iters_per_epoch) == 0:

                self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_epoch:
                stop = True
            else:
                stop = False

