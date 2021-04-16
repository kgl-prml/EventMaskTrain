import json

class MaskAnnotation(object):
    def __init__(self, path=""):
        self.filepath = path
        self.parse()
        self.get_event_bbox()

    def parse(self):
        if self.filepath == "":
            self.data = None
            return 
        with open(self.filepath) as f:
            self.data = json.load(f)

    def get_event_bbox(self):
        self.event_bbox = {}
        assert(self.data is not None)
        actvs = self.data['activities']
        for act in actvs:
            vname = list(act['localization'].keys())[0]
            if vname not in self.event_bbox:
                self.event_bbox[vname] = {}

            objs = act['objects']
            for obj in objs:
                bboxes = obj['localization'][vname] 
                for fid in bboxes:
                    if "boundingBox" not in bboxes[fid]:
                        continue
                    x = int(bboxes[fid]["boundingBox"]['x'])
                    y = int(bboxes[fid]["boundingBox"]['y'])
                    w = int(bboxes[fid]["boundingBox"]['w'])
                    h = int(bboxes[fid]["boundingBox"]['h'])
                    if int(fid) not in self.event_bbox[vname]:
                        self.event_bbox[vname][int(fid)] = []

                    self.event_bbox[vname][int(fid)] += [(x, y, w, h)]
 
