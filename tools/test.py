import torch
import argparse
from math import ceil as ceil
import os
import numpy as np
import torch.nn.functional as F
import cv2
import pydensecrf.densecrf as dcrf

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

    test_dataset = Dataset(cfg.DATASET.TEST_SPLIT_NAME, cfg.DATASET.DATAROOT, test_transforms, with_mask=cfg.TEST.WITH_MASK)
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

def associate_bboxes(frame_id, cur, proposal, thres=0.3):
    used = [False] * len(cur)
    for prop in proposal:
        _, last_bbox = prop[-1]
        if last_bbox is None:
            continue

        max_iou = -1.0
        max_cb = []
        max_cid = 0
        cid = 0
        for cb in cur:
            iou = IoU(last_bbox, cb)
            if iou > max_iou:
                max_iou = iou
                max_cb = cb
                max_cid = cid

            cid += 1

        if max_iou > thres:
            prop.append((frame_id, max_cb))
            used[max_cid] = True
        else:
            prop.append((frame_id, None))

    for i in range(len(used)):
        if not used[i]:
            #print(frame_id)
            proposal.append([(frame_id, cur[i])])

    return proposal

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
    print('number of regions: %d' % num_regions)

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
   
    print('Remaining %d regions' % (num_regions - len(region_to_remove)))
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

    for sample in iter(test_dataloader):
        if cfg.TEST.WITH_MASK:
            vid, start_f, clips, vmask = sample
        else:
            vid, start_f, clips = sample
        if clips.size(2) < 8: continue
        # forward and get the prediction result
        vpred = net(to_cuda(clips))
        # N x D x H x W
        probs = F.softmax(vpred, dim=1)
        pos_probs = probs[:, 1, :, :, :]

        start_f = start_f.numpy()
        N = len(vid)
        assert(N == len(start_f))
        for i in range(N):
            cur_vid = vid[i]
            cur_video_path = os.path.join(cfg.DATASET.DATAROOT, 'videos', '%s.%s'%(cur_vid, cfg.DATASET.VIDEO_FORMAT))
            print('video: %s, start_f: %d' % (cur_video_path, start_f[i]))
            # TODO: for debugging
            #if start_f[i] < 5344:
            #    continue
            cur_video = VideoReader(cur_video_path)
            #cur_video.seek(int(start_f[i]))

            frame_count = 0
            proposals = []
            clip_imgs = []
            #masks = []
            #for frame in cur_video.get_iter(clip_len * clip_stride):
            #    if frame_count % clip_stride:
            #        frame_count += 1
            #        continue

            #    # read the image
            #    img = frame.numpy()
            #    assert(len(img.shape) > 1)
            #    clip_imgs += [img]
            #    #img = img[:, :, [2, 1, 0]]
            #    #img = (img / 255.) * 2 - 1

            for fid in range(clip_len * clip_stride):
                if frame_count % clip_stride:
                    frame_count += 1
                    continue

                count = frame_count // clip_stride
                if not cfg.TEST.WITH_DENSE_CRF:
                    cur_pos_probs = pos_probs[i, count, :, :].cpu().numpy()
                else:
                    cur_probs = probs[i, :, count, :, :].cpu().numpy()
                    # TODO: need normalize or not?
                    resized_img = clips[i, :, count, :, :].cpu().numpy()
                    resized_img = np.uint8(255 * (resized_img + 1.) / 2.0)
                    cur_pos_probs = 1.0 * dense_crf(cur_probs, resized_img)

                smoothing(cur_pos_probs)
                labels, num_regions = regional_growing(cur_pos_probs, pixel_val_thres=0.3)
                cur_pos_probs, bboxes = filtering(cur_pos_probs, labels, num_regions, 5)
                #masks.append(cur_pos_probs)
                if len(proposals) == 0:
                    proposals = [[(count, bbox)] for bbox in bboxes]
                else:
                    associate_bboxes(count, bboxes, proposals)

                frame_count += 1

            #heatmaps = draw_heatmaps(clip_imgs, masks)
            #save_visualizations(heatmaps, 'heatmaps', cur_vid, start_f[i])

            #h, w, _ = clip_imgs[0].shape
            h, w = cur_video.height, cur_video.width
            new_proposals = [prop for prop in proposals if len(prop) >= 7]
            print('Number of proposals before and after filtering: %d, %d' % (len(proposals), len(new_proposals)))
            if len(new_proposals) == 0:
                continue
            #for prop in new_proposals:
            #    print(prop)

            stride_x = 1.0 * w / cfg.DATA_TRANSFORM.FINESIZE
            stride_y = 1.0 * h / cfg.DATA_TRANSFORM.FINESIZE
            new_proposals = resize_proposals(new_proposals, stride_x, stride_y, w, h)
            save_proposals(new_proposals, 'proposals', cur_vid, start_f[i])
            #print('H, W: %d, %d' % (h, w))
            #for prop in new_proposals:
            #    print('Resized proposal:')
            #    print(prop)

            #img_bboxes = draw_bboxes(clip_imgs, new_proposals)
            #save_visualizations(img_bboxes, 'bboxes', cur_vid, start_f[i])

def resize_proposals(proposals, stride_x, stride_y, W, H):
    new_proposals = []
    for prop in proposals:
        frame_ids = []
        min_x1 = 1e5
        max_x2 = -1
        min_y1 = 1e5
        max_y2 = -1
        for p in prop:
            frame_id, bbox = p
            if bbox is None:
                continue
            frame_ids += [frame_id]

            x1, y1, x2, y2 = bbox
            #new_bbox = (int(x1 * stride_x), int(y1 * stride_y), int(x2 * stride_x), int(y2 * stride_y))
            min_x1 = min(int(x1 * stride_x), min_x1)
            max_x2 = max(int(x2 * stride_x), max_x2)
            min_y1 = min(int(y1 * stride_y), min_y1)
            max_y2 = max(int(y2 * stride_y), max_y2)

        new_bbox = (max(min_x1, 0), max(min_y1, 0), min(max_x2, W-1), min(max_y2, H-1))
        new_proposals += [[(frame_id, new_bbox) for frame_id in frame_ids]]
    return new_proposals

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
