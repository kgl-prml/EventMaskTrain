import argparse
import logging
from collections import defaultdict
import json
import os.path as osp
import yaml
import os

class YamlAnnotation(object):

    FIELDS = ['activities', 'geom', 'types']

    def __init__(self, video_name: str, annotation_dir: str):
        self.video_name = video_name
        self.raw_data = self._load_raw_data(video_name, annotation_dir)

    def _split_meta(self, contents, key):
        meta = []
        i = 0
        while i < len(contents) and 'meta' in contents[i]:
            assert key not in contents[i]
            meta.append(contents[i]['meta'])
            i += 1
        data = [content[key] for content in contents[i:]]
        return meta, data

    def _load_file(self, video_name, annotation_dir, field):
        path = osp.join(annotation_dir, video_name + ".%s.yml"%field)
        if not osp.exists(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            contents = yaml.load(f, Loader=yaml.FullLoader)
        return contents

    def _load_raw_data(self, video_name, annotation_dir):
        raw_data = {'meta': {}}
        for field in self.FIELDS:
            contents = self._load_file(video_name, annotation_dir, field)
            key = field if field != 'activities' else 'act'
            raw_data['meta'][field], raw_data[field] = self._split_meta(
                contents, key)
        objs = defaultdict(dict)
        for obj in raw_data['geom']:
            obj['g0'] = [int(x) for x in obj['g0'].split()]
            objs[obj['id1']][obj['ts0']] = obj
        for obj in raw_data['types']:
            objs[obj['id1']]['type'] = [*obj['cset3'].keys()][0]
        for act in raw_data['activities']:
            for actor in act.get('actors', []):
                obj = objs[actor['id1']]
                geoms = []
                for ts in actor['timespan']:
                    start, end = ts['tsr0']
                    for time in range(start, end + 1):
                        geoms.append(obj[time])
                actor['geoms'] = geoms
                actor['type'] = obj['type']
        return raw_data

    def get_activities_official(self):
        video = self.video_name + '.avi'
        activities = []
        for act in self.raw_data['activities']:
            act_id = act['id2']
            act_type = [*act['act3'].keys()][0]
            if act_type.startswith('empty'):
                continue
            start, end = act['timespan'][0]['tsr0']
            objects = []
            for actor in act['actors']:
                actor_id = actor['id1']
                bbox_history = {}
                for geom in actor['geoms']:
                    frame_id = geom['ts0']
                    x1, y1, x2, y2 = geom['g0']
                    bbox_history[frame_id] = {
                        'presenceConf': 1,
                        'boundingBox': {
                            'x': min(x1, x2), 'y': min(y1, y2),
                            'w': abs(x2 - x1), 'h': abs(y2 - y1)}}
                for frame_id in range(start, end + 1):
                    if frame_id not in bbox_history:
                        bbox_history[frame_id] = {}
                obj = {'objectType': actor['type'], 'objectID': actor_id,
                       'localization': {video: bbox_history}}
                objects.append(obj)
            activity = {
                'activity': act_type, 'activityID': act_id,
                'presenceConf': 1, 'alertFrame': start,
                'localization': {video: {start: 1, end + 1: 0}},
                'objects': objects}
            activities.append(activity)
        return activities

class Converter(object):

    def __init__(self, vid2path):
        self.vid2path = vid2path
        self.video_list = list(vid2path.keys())

    def _convert_worker(self, video_name):
        if video_name in self.vid2path:
            annotation_dir = self.vid2path[video_name]
        else:
            annotation_dir = ""
        annotation = YamlAnnotation(video_name, annotation_dir)
        return annotation.get_activities_official()

    def get_official_format(self):
        activities = []
        for vid in self.video_list:
            result = self._convert_worker(vid)
            activities.extend(result)
        files_processed = [v + '.avi' for v in self.video_list]
        reference = {'filesProcessed': files_processed,
                     'activities': activities}
        return reference

def parse_args(argv=None):
    parser = argparse.ArgumentParser('Annotation converter from yml to json')
    parser.add_argument('--annotation_root', default="../actev-data-repo/annotation/DIVA-phase-2/MEVA/KF1-examples/")
    parser.add_argument('--output_file', default="./meva.json")
    args = parser.parse_args(argv)
    return args

def get_yml_paths(annotation_root):
    vid2path = {}
    for root, _, files in os.walk(annotation_root): 
        if files == []: continue
        for f in files:
            if f.split(".")[-1] != 'yml':
                continue
            yml_type = f.split(".")[-2]
            vid = f.replace(".%s.yml"%yml_type, "")
            vid2path[vid] = root

    return vid2path

def main(args):
    logging.basicConfig(level=logging.INFO)
    logging.info('Loading yml annotation from %s', args.annotation_root)

    vid2path = get_yml_paths(args.annotation_root)
    logging.info('Total video number: %d' % len(list(vid2path.keys())))
 
    converter = Converter(vid2path)
    reference = converter.get_official_format()
    logging.info('Writing json reference to %s', args.output_file)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(reference, f, indent=4)


if __name__ == '__main__':
    main(parse_args())
