import argparse
import os, json
from pycocotools.coco import COCO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_path", type=str, required=True)
    parser.add_argument("--out_fp", type=str, default='sub.json')
    parser.add_argument("--n_samples", type=int, required=True)

    args = parser.parse_args()
    return args

def create_coco_subset(coco, img_ids):
    # has 3 keys: 'images', 'annotations' and 'categories'
    sub_coco = {
        'images': [],
        'annotations' : [],
        'categories' : [],
    }

    # doesn't change format from ori json, it's just how coco returns the value in dict
    sub_coco['categories'] = [cat for key,cat in coco.cats.items()]

    # parse and append annotations
    for img_id in img_ids:
        img_info = coco.imgs[img_id]
        anns_ids = coco.getAnnIds(img_id)
        img_anns = coco.loadAnns(anns_ids)  # returns list of dict

        sub_coco['images'].append(img_info)
        for img_ann in img_anns:
            sub_coco['annotations'].append(img_ann)
    
    return sub_coco

if __name__=='__main__':
    # argparse
    args = parse_args()
    print(args.coco_path, args.n_samples, args.out_fp)
    print(os.getcwd())
    coco = COCO(args.coco_path)

    # grab first n samples
    sub_image_ids = sorted(coco.getImgIds()[:args.n_samples])
    sub_coco = create_coco_subset(coco, sub_image_ids)
    print(sub_coco.keys())

    with open(args.out_fp, 'w') as f:
        json.dump(sub_coco, f)