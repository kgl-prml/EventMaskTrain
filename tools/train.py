import torch
import argparse
import os
import numpy as np
from torch.backends import cudnn
from solver.solver import Solver as Solver
from model import model_i3d as model
from config.config import cfg, cfg_from_file, cfg_from_list
import sys
from utils import utils
import pprint
import data.video_transforms as video_transforms
from torchvision import transforms
from data.meva_dataset import MEVA as Dataset
import data.annotation as annot
from yaml2json import get_yml_paths, Converter
import json

SUFFIX = ['avi', 'mp4']
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with specified model parameters',
                        default=None, type=str)
    parser.add_argument('--resume', dest='resume',
                        help='initialize with saved solver status',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name', 
                        default='exp', type=str)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_video_names(video_list):
    with open(video_list, 'r') as f:
        vids = f.readlines()
        vids = [vid.strip() for vid in vids]

    vnames = []
    for vid in vids:
        if vid.split(".")[-1] in SUFFIX:
            with_suffix = True
        else:
            with_suffix = False

        vname = vid if with_suffix else vid + '.' + cfg.DATASET.VIDEO_FORMAT
        vnames += [vname]
    return vnames

def prepare_data():
    # split the video list
    vnames = get_video_names(cfg.DATASET.VIDEO_LIST)
    train_split = vnames
    test_split = vnames[:int(0.2*len(train_split))]
    
    train_transforms = transforms.Compose([
        video_transforms.Resize(cfg.DATA_TRANSFORM.LOADSIZE),
        video_transforms.RandomCrop(cfg.DATA_TRANSFORM.FINESIZE)])
        #video_transforms.RandomHorizontalFlip()])

    test_transforms = transforms.Compose([
        video_transforms.Resize(cfg.DATA_TRANSFORM.LOADSIZE),
        video_transforms.CenterCrop(cfg.DATA_TRANSFORM.FINESIZE)])

    video_root = cfg.DATASET.VIDEO_ROOT
    annot_path = cfg.DATASET.ANNOT_JSON_SAVE_PATH
    dataset = Dataset(train_split, video_root, annot_path, train_transforms)
    train_dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, 
            num_workers=cfg.NUM_WORKERS, drop_last=True, pin_memory=True)

    video_root = cfg.DATASET.VIDEO_ROOT
    annot_path = cfg.DATASET.ANNOT_JSON_SAVE_PATH
    test_dataset = Dataset(test_split, video_root, annot_path, test_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
            batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, 
            num_workers=cfg.NUM_WORKERS, drop_last=False, pin_memory=True)

    dataloaders = {'train': train_dataloader, 'test': test_dataloader}
    return dataloaders

def convert_annotations(annot_path, out_json_path):
    # convert yaml files to json files
    vid2path = get_yml_paths(annot_path)
    converter = Converter(vid2path)
    reference = converter.get_official_format()

    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(reference, f, indent=4)

def train(args):
    convert_annotations(cfg.DATASET.ANNOT_PATH, 
                      cfg.DATASET.ANNOT_JSON_SAVE_PATH)

    assert(os.path.exists(cfg.DATASET.ANNOT_JSON_SAVE_PATH)) 
    # prepare the data
    dataloaders = prepare_data()

    # initialize model
    model_state_dict = None
    resume_dict = None

    if cfg.RESUME != '':
        resume_dict = torch.load(cfg.RESUME)
        model_state_dict = resume_dict['model_state_dict']
    elif cfg.WEIGHTS != '':
        param_dict = torch.load(cfg.WEIGHTS)
        model_state_dict = param_dict['parameters']

    net = model.get_MaskGenNet(state_dict=model_state_dict)

    net = torch.nn.DataParallel(net)
    if torch.cuda.is_available():
       net.cuda()

    # initialize solver
    train_solver = Solver(net, dataloaders, resume=resume_dict)

    # train 
    train_solver.solve()
    print('Finished!')

if __name__ == '__main__':
    cudnn.benchmark = True 
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.resume is not None:
        cfg.RESUME = args.resume 
    if args.weights is not None:
        cfg.MODEL = args.weights
    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name 

    print('Using config:')
    pprint.pprint(cfg)

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.EXP_NAME)
    print('Output will be saved to %s.' % cfg.SAVE_DIR)

    train(args)
