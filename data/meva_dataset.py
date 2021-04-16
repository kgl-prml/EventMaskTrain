import torch
from torch.utils.data import Dataset as Dataset
from  config.config import cfg as cfg
import os
import cv2
import numpy as np
from math import ceil as ceil
import random
from diva_io.video import VideoReader
import data.annotation as annot

SUFFIX = ['avi', 'mp4']

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    if len(pic.shape) == 4:
        return torch.from_numpy(pic.transpose([3, 0, 1, 2]))
    else:
        return torch.from_numpy(pic)

def load_masks(cur_mask, start, num, size, stride=1):

    frames = []
    H, W = size
    for i in range(start, start + num):
        #cur_frame_path = os.path.join(root, frame_format_str%(i*stride))
        cur_frame_id = i * stride
        #if not os.path.exists(cur_frame_path):
        img = np.zeros(size)
        if cur_frame_id in cur_mask:
            for (x, y, w, h) in cur_mask[cur_frame_id]:
                img[y:min(y+h, H), x:min(x+w, W)] = 255.0

        img = (img / 255.) * 2 - 1
        frames.append(img)

    return np.asarray(frames, dtype=np.float32)


def load_frames(root, start, num, frame_format_str='', rgb=True, size=None, stride=1):
    frames = []
    for i in range(start, start + num):
        cur_frame_path = os.path.join(root, frame_format_str%(i*stride))
        if not os.path.exists(cur_frame_path):
            #print(cur_frame_path)
            img = np.zeros(size)
        else:
            if rgb:
                img = cv2.imread(cur_frame_path, cv2.IMREAD_COLOR)
            else:
                img = cv2.imread(cur_frame_path, cv2.IMREAD_GRAYSCALE)

        assert(len(img.shape) > 1)
        if rgb:
            img = img[:, :, [2, 1, 0]]

        img = cv2.resize(img, size)
        # TODO
        #if w < 226 or h < 226:
        #    d = 226. - min(w, h)
        #    sc = 1 + d /min(w, h)
        #    img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        img = (img / 255.) * 2 - 1
        frames.append(img)

    return np.asarray(frames, dtype=np.float32)

def load_frames_from_video(video_path, start, num, stride=1):
    frames = []

    cap = VideoReader(video_path)
    start_frame_id = start * stride
    video_len = cap.length
    
    length = num * stride 
    if length > video_len - start_frame_id:
        start_frame_id = video_len - length

    cap.seek(start_frame_id)

    count = 0
    for frame in cap.get_iter(length):
        if count % stride:
            count += 1
            continue

        img = frame.numpy()

        assert(len(img.shape) > 1)
        img = img[:, :, [2, 1, 0]]
        h, w, c = img.shape
        #print('shape: w: %d, h: %d, c: %d' % (w, h, c))

        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        img = (img / 255.) * 2 - 1
        frames.append(img)
        count += 1

    return np.asarray(frames, dtype=np.float32), start_frame_id

def make_dataset(vnames, root):
    dataset = []

    i = 0
    for vname in vnames:
        video_path = os.path.join(root, vname)
        if not os.path.exists(video_path):
            print('Warning: %s not exist!' % video_path)
            continue

        try:
            cap = VideoReader(video_path)
        except:
            print('Error in reading %s' % video_path)
            continue

        num_frames = cap.length 
        if num_frames < cfg.DATASET.CLIP_LEN:
            print('Skipping %s due to the short length %d.' % (video_path, num_frames))
            continue

        dataset.append((vname, num_frames))
        i += 1
    
    return dataset

#def make_dataset(split_file, root, with_mask=True):
#    dataset = []
#    num_frames_collect = []
#    with open(split_file, 'r') as f:
#        vids = f.readlines()
#        vids = [vid.strip() for vid in vids]
#
#    i = 0
#    for vid in vids:
#        video_path = os.path.join(root, 'videos', vid+'.'+cfg.DATASET.VIDEO_FORMAT)
#        try:
#            cap = VideoReader(video_path)
#        except:
#            print('Error in reading %s' % video_path)
#            continue
#
#
#        if not os.path.exists(video_path):
#            print('Warning: %s not exist!' % video_path)
#            continue
#
#        num_frames = cap.length #len(os.listdir(video_path))
#        if with_mask:
#            mask_path = os.path.join(root, 'event_mask', vid)
#            if not os.path.exists(mask_path):
#                print('Warning: %s not exist!' % mask_path)
#                continue
#
#            assert(num_frames >= len(os.listdir(mask_path)))
#
#        if num_frames < cfg.DATASET.CLIP_LEN:
#            print('Skipping %s due to the short length %d.' % (video_path, num_frames))
#            continue
#
#        dataset.append((vid, num_frames))
#        num_frames_collect.append(num_frames)
#
#        #print(vid, num_frames)
#        i += 1
#    
#    return dataset, num_frames_collect

def parse_annotations(annot_path):
    mask_annot = annot.MaskAnnotation(annot_path)
    event_bbox = mask_annot.event_bbox
    return event_bbox

def split_pos_neg_clips(data, mask, clip_len=8, stride=4, pos_thres=0.5, neg_thres=0.1):
    pos_indices =  {}
    neg_indices = {}
    for (vname, length) in data:
        max_index = int(length // stride)

        pos_index, neg_index = [], []
        for i in range(max_index-clip_len+1):
            valid_count = 0
            for j in range(i, i+clip_len):
                if vname in mask and j*stride in mask[vname]:
                    valid_count += 1

            valid_prop = 1.0 * valid_count / clip_len
            if valid_prop >= pos_thres:
                pos_index += [i]
            elif valid_prop >= neg_thres:
                neg_index += [i]
            elif valid_prop == 0.0:
                neg_index += [i]

        pos_indices[vname] = pos_index
        neg_indices[vname] = neg_index

        #print('%s number of positive: %d' % (vname, len(pos_index)))
        #print('%s number of negtive: %d' % (vname, len(neg_index)))
    return pos_indices, neg_indices
    
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

class MEVA(Dataset):
    def __init__(self, vnames, video_root, annot_path, 
            transforms=None):

        self.split_filepath = split_filepath

        #self.video_root = os.path.join(root, cfg.DATASET.VIDEO_DIR)
        self.video_root = video_root
        self.annot_path = annot_path

        #if train:
        #    self.mask_root = cfg.ANNOT.TRAIN_OUT_PATH
        #else:
        #    self.mask_root = cfg.ANNOT.TEST_OUT_PATH
        #assert(os.path.isdir(self.mask_root)), self.mask_root

        #split_file = os.path.join(root, split)
        self.vnames = vnames #get_video_names(self.split_filepath)
        self.data = make_dataset(self.vnames, self.video_root)
        self.mask = parse_annotations(self.annot_path)

        self.clip_len = cfg.DATASET.CLIP_LEN
        self.stride = cfg.DATASET.CLIP_STRIDE
        pos_thres = cfg.DATASET.POS_THRES
        neg_thres = cfg.DATASET.NEG_THRES
        self.pos_indices, self.neg_indices = split_pos_neg_clips(
                self.data, self.mask, clip_len=self.clip_len, 
                stride=self.stride, pos_thres=pos_thres, 
                neg_thres=neg_thres)

        print('%s: %d' % (self.split_filepath, len(self.data)))
        self.transforms = transforms

    def get_index(self, type, vid):
        index_file = os.path.join('./KF1_pos_neg_split', vid, type+'.txt')
        #print('index_file: %s' % index_file)
        with open(index_file, 'r') as f:
            lines = f.readlines()
            indices = [int(line.strip()) for line in lines if line.strip() != ""]
        return indices

    def __getitem__(self, index):
        vname, nf = self.data[index]
        r = np.random.rand(1)[0]
        
        #pos_inds = self.get_index('pos', vid)
        #neg_inds = self.get_index('neg', vid)
        pos_inds = self.pos_indices[vname]
        neg_inds = self.neg_indices[vname]

        #print('pos ratio: %f' % cfg.TRAIN.POS_RATIO)
        assert(len(pos_inds) > 0 or len(neg_inds) > 0)
        if len(pos_inds) > 0 and len(neg_inds) == 0:
            #print('Choose positive.')
            inds = pos_inds
        elif len(pos_inds) == 0 and len(neg_inds) > 0:
            #print('Choose negative (1).')
            inds = neg_inds
        elif r < cfg.TRAIN.POS_RATIO:
            #print('Choose positive.')
            inds = pos_inds
        else:
            #print('Choose negative (2).')
            inds = neg_inds

        assert(len(inds) > 0), 'Please check %s.' % vname
        start_f = inds[random.randint(0, len(inds)-1)]
        clip_len = self.clip_len
        stride = self.stride

        cur_video = os.path.join(self.video_root, vname)
        #print(cur_video)

        try:
            imgs, _ = load_frames_from_video(cur_video, start_f, clip_len, stride=stride)
        except:
            assert(False), 'video %s, start_f %d, clip_len %d, stride %d' % (cur_video, start_f, clip_len, stride)

        h, w = imgs.shape[1], imgs.shape[2]
        #print('load image shape: w: %d, h: %d' % (w, h))
        #mask = load_frames(cur_mask, start_f, clip_len, 
        #                   frame_format_str='frame_%d_mask.png', 
        #                   rgb=False, size=(w, h), stride=stride)

        #assert(False), vname + ' ' + list(self.mask.keys())[0]
        if vname not in self.mask:
            cur_mask = []
        else:
            #print('Right')
            cur_mask = self.mask[vname]

        mask = load_masks(cur_mask, start_f, clip_len, size=(h, w), stride=stride)

        if self.transforms is not None:
            # TODO: to implement in a more efficient way in future
            T = imgs.shape[0]
            ch = imgs.shape[3]
            mask = np.expand_dims(mask, 3)
            mask = np.repeat(mask, ch, axis=3)
            #print('shape of img: ', imgs.shape)
            #print('shape of mask: ', mask.shape)
            img_mask = np.concatenate((imgs, mask), axis=0)
            img_mask = self.transforms(img_mask)
            imgs = img_mask[:T, :, :, :]
            mask = img_mask[T:, :, :, 0]
        
        imgs = video_to_tensor(imgs)
        mask = (video_to_tensor(mask) > 0).type(torch.LongTensor)
        #print('shape of imgs: ', imgs.size())
        #print('prop of foreground: %4f' % (1.0 * torch.sum(mask).item() / mask.view(-1).size(0)))

        return vname, start_f, imgs, mask

    def __len__(self):
        return len(self.data)

class MEVATest(Dataset):
    def __init__(self, split, root, transforms=None, with_mask=False):
        self.split = split
        self.root = root
        self.video_root = os.path.join(root, 'videos')

        self.with_mask = with_mask
        self.mask_root = ""
        if with_mask:
            self.mask_root = os.path.join(root, 'event_mask')

        split_file = os.path.join(root, split)
        self.data, self.num_frames = make_dataset(split_file, self.root, self.with_mask)
        print('%s: %d' % (self.split, len(self.data)))
        self.efinds = []
        self.get_efinds()
        self.transforms = transforms

    def get_efinds(self):
        self.efinds = []
        stride = cfg.DATASET.CLIP_LEN * cfg.DATASET.CLIP_STRIDE
        for nf in self.num_frames:
            last = self.efinds[-1] if len(self.efinds) > 0 else -1
            num_ind = ceil(1.0 * nf / stride)
            self.efinds.append(last + num_ind)
        return

    def get_vid_and_fid(self, index):
        vind = 0
        for ind in self.efinds:
            if ind >= index:
                break
            vind += 1

        start_find = self.efinds[vind - 1] + 1 if vind > 0 else 0
        start = index - start_find
        return vind, start

    def __getitem__(self, index):
        vind, start = self.get_vid_and_fid(index)
        vid, nf = self.data[vind]

        clip_len = cfg.DATASET.CLIP_LEN
        stride = cfg.DATASET.CLIP_STRIDE

        cur_video = os.path.join(self.video_root, vid+'.'+cfg.DATASET.VIDEO_FORMAT)
        start_f = clip_len * start
        try:
            imgs, real_start = load_frames_from_video(cur_video, start_f, clip_len, stride=stride)
        except:
            assert(False), 'video %s, start_f %d, clip_len %d, stride %d' % (cur_video, start_f, clip_len, stride)

        if self.with_mask:
            cur_mask = os.path.join(self.mask_root, vid, 'bbox')
            h, w = imgs.shape[1], imgs.shape[2]
            mask = load_frames(cur_mask, start_f, clip_len, 
                               frame_format_str=vid+'.ts0-%d.bbox.png', 
                               rgb=False, size=(w, h), stride=stride)

        if self.transforms is not None:
            if self.with_mask:
                # TODO: to implement in a more efficient way in future
                T = imgs.shape[0]
                ch = imgs.shape[3]
                mask = np.expand_dims(mask, 3)
                mask = np.repeat(mask, ch, axis=3)
                #print('shape of img: ', imgs.shape)
                #print('shape of mask: ', mask.shape)
                img_mask = np.concatenate((imgs, mask), axis=0)
                img_mask = self.transforms(img_mask)
                imgs = img_mask[:T, :, :, :]
                mask = img_mask[T:, :, :, 0]
            else:
                imgs = self.transforms(imgs)
        
        imgs = video_to_tensor(imgs)
        if self.with_mask:
            mask = (video_to_tensor(mask) > 0).type(torch.LongTensor)
            return vid, real_start, imgs, mask
        else:
            return vid, real_start, imgs

    def __len__(self):
        return self.efinds[-1] + 1
