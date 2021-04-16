import torch
import numpy as np
from diva_io.video import VideoReader
import data.annotation as annot
import os
import cv2
import numpy as np

SUFFIX = ["avi", "mp4"]

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_onehot(label, num_classes):
    identity = to_cuda(torch.eye(num_classes))
    onehot = torch.index_select(identity, 0, label)
    return onehot

def mean_accuracy(preds, target):
    num_classes = preds.size(1)
    preds = torch.max(preds, dim=1).indices
    accu_class = []
    for c in range(num_classes):
        mask = (target == c)
        c_count = torch.sum(mask).item()
        if c_count == 0: continue
        preds_c = torch.masked_select(preds, mask)
        accu_class += [1.0 * torch.sum(preds_c == c).item() / c_count]
    return 100.0 * np.mean(accu_class)

def accuracy(preds, target):
    preds = torch.max(preds, dim=1).indices
    corrects = (preds == target).type(torch.cuda.FloatTensor)
    #return 100.0 * torch.sum(preds == target).item() / preds.size(0)
    return 100.0 * torch.mean(corrects)

def iou(preds, target, exclude_neg=False):
    N = preds.size(0)
    assert(N == target.size(0))
    pred_labels = torch.max(preds, dim=1).indices
    # compute the intersection 
    inter = pred_labels * target
    # compute the union
    union = (pred_labels + target) > 0
    # compute average iou
    total_inter = torch.sum(inter.view(N, -1), dim=1).type(torch.cuda.FloatTensor)
    total_union = torch.sum(union.view(N, -1), dim=1).type(torch.cuda.FloatTensor)
    eps = 1e-5
    iou = total_inter / (total_union + eps)
    mask = (total_union == 0).type(torch.cuda.FloatTensor)
    if not exclude_neg:
        iou = (1.0 - mask) * iou + mask * 1.0
        iou = torch.mean(iou)
    else:
        neg_N = torch.sum(mask).item()
        N = iou.size(0) - neg_N
        iou = 1.0 * torch.sum(iou) / N
    return 100.0 * iou

def gen_mask_gt(video_list, video_root_path, annot_path, annot_out_path, video_suffix='avi'):
    mask_annot = annot.MaskAnnotation(annot_path)
    
    with open(video_list, 'r') as f:
        lines = f.readlines()
        vids = [line.strip() for line in lines]
    
    height, width = -1, -1
    event_bbox = mask_annot.event_bbox
    if vids[0].split(".")[-1] in SUFFIX:
        with_suffix = True
    else:
        with_suffix = False

    for vid in vids:
        vname = vid + '.' + video_suffix if not with_suffix else vid
        video_path = os.path.join(video_root_path, vname)
        cap = VideoReader(video_path)
        if vname not in event_bbox:
            continue
    
        for frame in cap.get_iter(1):
            img = frame.numpy()
            assert(len(img.shape) > 1)
            height, width, _ = img.shape
    
        cur_bboxes = event_bbox[vname]
        for fid in cur_bboxes:
            img = np.zeros([height, width])
            bboxes = cur_bboxes[fid]
            for bbox in bboxes:
                print(vid, fid, bbox, height, width)
                x1, y1, w, h = bbox
                img[y1:y1+h, x1:x1+w] = 255
    
            save_path = os.path.join(annot_out_path, vid)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            cv2.imwrite(os.path.join(save_path, "frame_%d_mask.png"%fid), img)

