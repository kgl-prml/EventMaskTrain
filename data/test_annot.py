import annotation as annot

mask_annot = annot.MaskAnnotation("../meva.json")
event_bbox = mask_annot.event_bbox
vids = list(event_bbox.keys())
vid = vids[10]

max_len = 0
max_vid = -1
max_fid = -1
for vid in vids:
    for fid in event_bbox[vid]:
        if len(event_bbox[vid][fid]) > 1:
            #print(vid, fid, event_bbox[vid][fid])
            if len(event_bbox[vid][fid]) > max_len:
                max_len = len(event_bbox[vid][fid])
                max_vid = vid
                max_fid = fid
print(max_len)
print(max_vid, max_fid, event_bbox[max_vid][max_fid])
