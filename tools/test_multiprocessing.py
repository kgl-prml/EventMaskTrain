import torch
import argparse
from math import ceil as ceil
import os
import numpy as np
import torch.nn.functional as F
import cv2
import pydensecrf.densecrf as dcrf
import multiprocessing

from utils.utils import to_cuda
from torch.backends import cudnn
from solver.solver import Solver as Solver
#from model import model as model
from model import model_i3d as model
from config.config import cfg, cfg_from_file, cfg_from_list
import sys
import pprint
import data.video_transforms as video_transforms
from torchvision import transforms
from data.meva_dataset import MEVATest as Dataset
from diva_io.video import VideoReader

sys.setrecursionlimit(80000)
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

def prepare_data():
    test_transforms = transforms.Compose([
        video_transforms.Resize(cfg.DATA_TRANSFORM.FINESIZE)])
        #video_transforms.CenterCrop(cfg.DATA_TRANSFORM.FINESIZE)])

    test_dataset = Dataset(cfg.DATASET.TEST_SPLIT_NAME, cfg.DATASET.DATAROOT, test_transforms, with_mask=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
            batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, 
            num_workers=cfg.NUM_WORKERS, drop_last=False, pin_memory=True)

    return test_dataloader

# resized_img: np.uint8; w x h x c (rgb)
def dense_crf(probs, resized_img):
    unary = probs
    assert(len(unary.shape) == 3)
    unary = -np.log(unary)
    unary = unary.transpose(2, 1, 0)
    w, h, c = unary.shape
    unary = unary.transpose(2, 0, 1).reshape(2, -1)
    unary = np.ascontiguousarray(unary)
    resized_img = resized_img.transpose(2, 1, 0)
    resized_img = np.ascontiguousarray(resized_img)
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)
    q = d.inference(50)
    mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
    return mask

def find_seeds(image, unvisited, pixel_val_thres):
    seeds = set()
    H, W = image.shape
    for h in range(H):
        for w in range(W):
            if image[h, w] >= pixel_val_thres and unvisited[h, w] > 0:
                pos = h * W + w
                seeds.add(pos)
    return seeds

def growing(pos, image, seeds, unvisited, region_id, labels, pixel_val_thres):
    H, W = pos
    max_H, max_W = image.shape
    if image[H, W] < pixel_val_thres or unvisited[H, W] == 0:
        unvisited[H, W] = 0
        return
    else:
        labels[H, W] = region_id
        unvisited[H, W] = 0
        pos = H * max_W + W
        if pos in seeds:
            seeds.remove(pos)

    for h in range(max(H-1, 0), min(H+2, max_H)):
        for w in range(max(W-1, 0), min(W+2, max_W)):
            if h == H and w == W:
                continue
            else:
                growing((h, w), image, seeds, unvisited, region_id, labels, pixel_val_thres)
    return

def regional_growing(image, pixel_val_thres=0.7):
    H, W = image.shape
    unvisited = np.ones([H, W])
    labels = np.zeros([H, W]) - 1.0
    count = 0
    seeds = find_seeds(image, unvisited, pixel_val_thres)
    while np.sum(unvisited) > 0:
        if len(seeds) == 0:
            break
        seed_pos = seeds.pop()
        h = seed_pos // W
        w = seed_pos - h * W
        #print('pos: %d, %d; prob: %f.' % (h, w, image[h, w]))
        # growing
        growing((h, w), image, seeds, unvisited, count, labels, pixel_val_thres)
        #print('unvisited: %d' % np.sum(unvisited))
        count += 1
    return labels, count

def IoU(A, B):
    xA1, yA1, xA2, yA2 = A
    xB1, yB1, xB2, yB2 = B
    
    intersection = max(min(xA2, xB2) - max(xA1, xB1), 0.0) * max(min(yA2, yB2) - max(yA1, yB1), 0.0)
    union = (max(xA2, xB2) - min(xA1, xB1)) * (max(yA2, yB2) - min(yA1, yB1))
    return 1.0 * intersection / union

def associate_bboxes(cur_bbox, next_bboxes, thres=0.3):
    max_iou = -1.0
    max_nid = 0
    nid = 0
    for nb in next_bboxes:
        iou = IoU(cur_bbox, nb)
        if iou > max_iou:
            max_iou = iou
            max_nid = nid

        nid += 1

    if max_iou > thres:
        return max_nid
    else:
        return None

def smoothing(mask, threshold=0.8, len_thres=5):
    h, w = mask.shape
    for i in range(h):
        start = -1
        end = -1
        for j in range(w):
            if mask[i, j] >= threshold and start == -1:
                start = j
            if mask[i, j] < threshold and start > 0:
                end = j

            if start > 0 and end > 0:
                length = end - start + 1
                if length < len_thres:
                    #print('Start %d, end %d' % (start, end))
                    mask[i, start:end+1] = 0
                start = end = -1

    for i in range(w):
        start = -1
        end = -1
        for j in range(h):
            if mask[j, i] >= threshold and start == -1:
                start = j
            if mask[j, i] < threshold and start > 0:
                end = j

            if start > 0 and end > 0:
                length = end - start + 1
                if length < len_thres:
                    #print('Start %d, end %d' % (start, end))
                    mask[start:end+1, i] = 0
                start = end = -1

def filtering(image, labels, num_regions, threshold=10):
    H, W = image.shape
    label_count = [0] * num_regions
    region_min_x = [W] * num_regions
    region_max_x = [-1] * num_regions
    region_min_y = [H] * num_regions 
    region_max_y = [-1] * num_regions
    #print('number of regions: %d' % num_regions)

    mask = np.zeros([H, W])
    for h in range(H):
        for w in range(W):
            if labels[h, w] == -1:
                continue
            #print(labels[h, w])
            region_id = int(labels[h, w])
            label_count[region_id] += 1
            region_min_x[region_id] = min(w, region_min_x[region_id])
            region_min_y[region_id] = min(h, region_min_y[region_id])
            region_max_x[region_id] = max(w, region_max_x[region_id])
            region_max_y[region_id] = max(h, region_max_y[region_id])

    eps = 1e-5
    region_to_remove = []
    for r in range(num_regions):
        # 1) occupation should be larger
        metric1 = label_count[r]
        if metric1 < threshold:
            region_to_remove += [r]
            continue

        # 2) more like a rectangle
        width = region_max_x[r] - region_min_x[r] + 1
        height = region_max_y[r] - region_min_y[r] + 1
        if width < 3 or height < 3:
            region_to_remove += [r]
            continue

        metric2 = 1.0 * label_count[r] / (width * height)
        if metric2 < 0.3:
            region_to_remove += [r]
            continue

        #metric3 = 1.0 * width / height
        #if metric3 > 2 or metric3 < 0.5:
        #    region_to_remove += [r]
        #    continue

    bboxes = []
    for r in range(num_regions):
        if r not in region_to_remove:
            x1, y1, x2, y2 = region_min_x[r], region_min_y[r], region_max_x[r], region_max_y[r]
            bboxes += [(x1, y1, x2, y2)]
            #print(bboxes[-1])
   
    #print('Remaining %d regions' % (num_regions - len(region_to_remove)))
    for h in range(H):
        for w in range(W):
            if labels[h, w] >= 0 and labels[h, w] not in region_to_remove:
                mask[h, w] = 1.0
    return mask, bboxes

def draw_heatmaps(imgs, masks):
    N = len(imgs)
    assert(N == len(masks))
    heatmaps = []
    for i in range(N):
        h, w, _ = imgs[i].shape
        mask = np.uint8(255 * cv2.resize(masks[i], (w, h)))
        # apply the heatmap on the image
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        mix = np.uint8(heatmap * 0.3 + imgs[i] * 0.7)
        heatmaps += [mix]
    return heatmaps

def draw_bboxes(imgs, proposals):
    N = len(proposals) * 1.0
    R = np.linspace(0, 255, ceil(N ** (1/3)))
    G = np.linspace(0, 255, ceil(N ** (1/3)))
    B = np.linspace(0, 255, ceil(N ** (1/3)))
    colors = [(r, g, b) for r in R for g in G for b in B]
    count = 0
    for prop in proposals:
        for p in prop:
            frame_id, bbox = p
            if bbox is None:
                continue

            color = colors[count]
            imgs[frame_id] = cv2.rectangle(imgs[frame_id], bbox[0:2], bbox[2:], color, 4)
        count += 1

    return imgs

def save_proposals(proposals, name, vid, start_f):
    save_path = os.path.join(cfg.SAVE_DIR, name, vid)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'props.txt'), 'a') as f:
        end_f = start_f + cfg.DATASET.CLIP_LEN * cfg.DATASET.CLIP_STRIDE - 1
        for prop in proposals:
            _, bbox = prop[0]
            x1, y1, x2, y2 = bbox
            f.write('%d, %d: %d, %d, %d, %d\n' % (start_f, end_f, x1, y1, x2, y2))

def save_visualizations(imgs, name, vid, start_f):
    save_path = os.path.join(cfg.SAVE_DIR, name, vid)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    clip_hp_full_path = os.path.join(save_path, '%d.avi'%start_f)
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 
    fps = 8
    h, w, _ = imgs[0].shape
    out = cv2.VideoWriter(clip_hp_full_path, codec, fps, (w, h))
    for f in imgs:
        out.write(f)

    out.release()

def test_and_save_mask(net, test_dataloader):
    clip_len = cfg.DATASET.CLIP_LEN
    clip_stride = cfg.DATASET.CLIP_STRIDE

    dataroot = cfg.DATASET.DATAROOT
    with open(os.path.join(dataroot, cfg.DATASET.TEST_SPLIT_NAME), 'r') as f:
        lines = f.readlines()
        vids = [vid.strip() for vid in lines]

    num_workers = len(vids)
    #print('The number of workers: %d' % num_workers)
    vid_meta = multiprocessing.Manager().dict()
    ps = []
    for n in range(num_workers):
        p1 = multiprocessing.Process(target=get_vid_meta, args=[vids[n], vid_meta])
        p1.start()
        ps += [p1]

    for p in ps:
        p.join()

    #for vid in vid_meta:
        #print('%s: (%d, %d).' % (vid, vid_meta[vid][0], vid_meta[vid][1]))

    for sample in iter(test_dataloader):
        vid, start_f, clips, vmask = sample
        if clips.size(2) < clip_len: continue
        # forward and get the prediction result
        vpred = net(to_cuda(clips))
        # N x D x H x W
        probs = F.softmax(vpred, dim=1)
        pos_probs = probs[:, 1, :, :, :].cpu().numpy()

        start_f = start_f.numpy()
        N = len(vid)
        assert(N == len(start_f))

        frame_bboxes = {v: multiprocessing.Manager().dict() for v in vid}
        num_workers = N * clip_len
        ps = []
        for n in range(num_workers):
            clip_id = n // clip_len
            frame_id = n - clip_id * clip_len
            p1 = multiprocessing.Process(target=split_regions, args=(clip_id, frame_id, start_f, vid, pos_probs, frame_bboxes))
            p1.start()
            ps += [p1]

        for p in ps:
            p.join()

        vid_list = list(frame_bboxes.keys())
        associate = {}
        starts = {}
        for v in vid_list:
            associate[v] = {}
            starts[v] = {}
            for f in sorted(frame_bboxes[v]):
                nbboxes = len(frame_bboxes[v][f])
                if nbboxes == 0:
                    continue
                associate[v][f] = multiprocessing.Manager().list([None] * nbboxes)
                starts[v][f] = multiprocessing.Manager().dict({i: 1 for i in range(nbboxes)})

        ps = []
        for n in range(num_workers):
            clip_id = n // clip_len
            frame_id = n - clip_id * clip_len
            p1 = multiprocessing.Process(target=associate_bboxes_adjacent, args=(clip_id, frame_id, start_f, vid, frame_bboxes, associate, starts))
            p1.start()
            ps += [p1]

        for p in ps:
            p.join()

        
        #for v in associate:
        #    if len(associate[v]) == 0:
        #        continue
        #    for f in associate[v]:
        #        if len(associate[v][f]) == 0:
        #            continue
        #        print(v, f, associate[v][f])

        proposals = gen_proposals(associate, starts, frame_bboxes, vid_meta)
        #print(proposals)

def get_vid_meta(vid, vid_meta):
    video_path = os.path.join(cfg.DATASET.DATAROOT, 'videos', '%s.%s'%(vid, cfg.DATASET.VIDEO_FORMAT))
    video = VideoReader(video_path)
    h, w = video.height, video.width
    vid_meta[vid] = (h, w)

def split_regions(clip_id, frame_id, start_f, vid, pos_probs, frame_bboxes):
    cur_vid = vid[clip_id]
    cur_pos_probs = pos_probs[clip_id, frame_id, :, :]
    smoothing(cur_pos_probs)
    labels, num_regions = regional_growing(cur_pos_probs, pixel_val_thres=0.3)
    cur_pos_probs, bboxes = filtering(cur_pos_probs, labels, num_regions, 5)
    real_frame_id =  start_f[clip_id] + frame_id * cfg.DATASET.CLIP_STRIDE
    frame_bboxes[cur_vid][real_frame_id] = bboxes
    #print('Finish splitting %s at %d' % (cur_vid, real_frame_id))

# TODO: to set thres as a hyper-parameter in the configure file
def associate_bboxes_adjacent(clip_id, frame_id, start_f, vid, frame_bboxes, associate, starts):
    clip_len, clip_stride = cfg.DATASET.CLIP_LEN, cfg.DATASET.CLIP_STRIDE
    real_frame_id = start_f[clip_id] + frame_id * clip_stride

    cur_vid = vid[clip_id]
    if cur_vid not in frame_bboxes or real_frame_id not in frame_bboxes[cur_vid]:
        return

    if frame_id >= clip_len - 1:
        return 

    next_real_frame_id = real_frame_id + clip_stride
    #if next_real_frame_id not in frame_bboxes[cur_vid]:
    #    return 

    next_bboxes = frame_bboxes[cur_vid][next_real_frame_id]

    bbox_id = 0
    cur_bboxes = frame_bboxes[cur_vid][real_frame_id]
    for cur_bbox in cur_bboxes:
        nid = associate_bboxes(cur_bbox, next_bboxes, thres=0.3)
        associate[cur_vid][real_frame_id][bbox_id] = nid

        if nid is not None and nid in starts[cur_vid][next_real_frame_id]:
            del starts[cur_vid][next_real_frame_id][nid]
        bbox_id += 1

    #print('Finish association %s at %d' % (cur_vid, real_frame_id))

def resize_proposals(prop, stride_x, stride_y, W, H):
    new = []
    min_x1 = 1e5
    max_x2 = -1
    min_y1 = 1e5
    max_y2 = -1
    for bbox in prop:
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        min_x1 = min(int(x1 * stride_x), min_x1)
        max_x2 = max(int(x2 * stride_x), max_x2)
        min_y1 = min(int(y1 * stride_y), min_y1)
        max_y2 = max(int(y2 * stride_y), max_y2)

    new_bbox = (max(min_x1, 0), max(min_y1, 0), min(max_x2, W-1), min(max_y2, H-1))
    new += [new_bbox]
    return new

def gen_proposals(associate, starts, frame_bboxes, vid_meta):
    proposals = {}
    clip_len, clip_stride = cfg.DATASET.CLIP_LEN, cfg.DATASET.CLIP_STRIDE
    vid_list = list(starts.keys())
    real_clip_len = clip_len * clip_stride

    prop_id = 0
    for vid in vid_list:
        proposals[vid] = {}
        fbboxes = starts[vid]
        for f in sorted(fbboxes):
            cur_starts = list(fbboxes[f].keys())
            #print('frame: ', f, cur_starts)
            #print(len(frame_bboxes[vid][f]))
            if len(cur_starts) == 0:
                continue

            for cur in cur_starts:
                bbox_list = []
                cur_bbox = frame_bboxes[vid][f][cur]
                bbox_list.append(cur_bbox)
                nf = f
                while True:
                    nid = associate[vid][nf][cur]
                    if nid is None:
                        break
                    nf += clip_stride
                    cur_bbox = frame_bboxes[vid][nf][nid]
                    bbox_list.append(cur_bbox)
                    cur = nid

                if (nf - f) // clip_stride < 7:
                    continue

                start = (f // real_clip_len) * real_clip_len
                end = start + real_clip_len - 1
                proposals[vid][prop_id] = {'start': start}
                proposals[vid][prop_id]['end'] = end

                h, w = vid_meta[vid]
                stride_x = 1.0 * w / cfg.DATA_TRANSFORM.FINESIZE
                stride_y = 1.0 * h / cfg.DATA_TRANSFORM.FINESIZE
                proposals[vid][prop_id]['bbox'] = resize_proposals(bbox_list, stride_x, stride_y, w, h)
                prop_id += 1

    return proposals

def test(args):
    # prepare the data
    test_dataloader = prepare_data()

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
    with torch.no_grad():
        test_and_save_mask(net, test_dataloader)

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

    test(args)
