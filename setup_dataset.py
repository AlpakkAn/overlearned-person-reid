import argparse
import json
import itertools as it
import os
import random
import shutil

import numpy as np
from torchvision.utils import save_image
from torchvision.transforms.functional import crop

from itertools import groupby, chain

import torch
import torchvision.transforms as transforms
from PIL import Image
from pycocotools import mask


def to_coco_img(path):
    with open(path, 'r') as f:
        video_json = json.load(f)

    img_counter = it.count(1)
    annotation_counter = it.count(1)

    img_json = {'images': [{'id': next(img_counter),
                            'video_id': video['id'],
                            'frame_id': frame_id,
                            'width': video['width'],
                            'height': video['height'],
                            'file_name': file_name}
                           for video in video_json['videos'] for frame_id, file_name in enumerate(video['file_names'])],
                'categories': video_json['categories']}

    img_json['annotations'] = [{'id': next(annotation_counter),
                                'object_id': object_id,
                                'video_id': obj['video_id'],
                                'image_id': next(filter(lambda image: image['video_id'] == obj['video_id']
                                                        and image['frame_id'] == seg_no, img_json['images']))['id'],
                                'category_id': obj['category_id'],
                                'segmentation': segmentation,
                                'area': obj['areas'][seg_no],
                                'bbox': obj['bboxes'][seg_no],
                                'iscrowd': obj['iscrowd']}
                               for object_id, obj in enumerate(video_json['annotations'], start=1)
                               for seg_no, segmentation in enumerate(obj['segmentations'])
                               if segmentation is not None]

    out_path = path.rsplit('/', 1)[0] + '/img_instances.json'
    with open(out_path, 'w') as outfile:
        json.dump(img_json, outfile)

    return out_path


def split(data_path, train_percentage=0.8, val_percentage=0.1):
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Shuffle videos
    random.seed(123)
    videos = [list(video[1]) for video in groupby(data['images'], key=lambda x: x['video_id'])]
    annotations = [list(ann[1]) for ann in groupby(data['annotations'], key=lambda x: x['video_id'])]
    shuffled = list(zip(videos, annotations))
    random.shuffle(shuffled)
    videos, annotations = zip(*shuffled)

    split_1 = int(train_percentage * len(videos))
    split_2 = int((train_percentage + val_percentage) * len(videos))
    train_json = {'images': list(chain.from_iterable(videos[:split_1])),  # flatten list
                  'categories': data['categories'],
                  'annotations': list(chain.from_iterable(annotations[:split_1]))}
    val_json = {'images': list(chain.from_iterable(videos[split_1:split_2])),
                'categories': data['categories'],
                'annotations': list(chain.from_iterable(annotations[split_1:split_2]))}
    test_json = {'images': list(chain.from_iterable(videos[split_2:])),
                 'categories': data['categories'],
                 'annotations': list(chain.from_iterable(annotations[split_2:]))}

    root_dir = data_path.rsplit('/', 1)[0]
    out_paths = [os.path.join(root_dir, 'train', 'instances.json'),
                 os.path.join(root_dir, 'valid', 'instances.json'),
                 os.path.join(root_dir, 'test', 'instances.json')]
    for path in out_paths:
        if not os.path.exists(path.rsplit('/', 1)[0]):
            os.makedirs(path.rsplit('/', 1)[0])
    with open(out_paths[0], 'w') as outfile:
        json.dump(train_json, outfile)
    with open(out_paths[1], 'w') as outfile:
        json.dump(val_json, outfile)
    with open(out_paths[2], 'w') as outfile:
        json.dump(test_json, outfile)

    return out_paths


def setup_foreground_data(root, crop_bbx=True, ovis=False, num_yvis_objects=8430,
                          coco=False, num_ovis_objects=5223):
    if ovis or coco:
        ann_paths = [os.path.join(root, 'train', 'instances.json')]
        if not os.path.exists(os.path.join(root, 'train')):
            os.makedirs(os.path.join(root, 'train'))
        shutil.copyfile(os.path.join(root, 'img_instances.json'), ann_paths[0])
    else:
        ann_paths = [os.path.join(root, 'train', 'instances.json'),
                     os.path.join(root, 'valid', 'instances.json'),
                     os.path.join(root, 'test', 'instances.json')]

    for path in ann_paths:
        with open(path, 'r') as f:
            data = json.load(f)

        if crop_bbx:
            path_fg = os.path.join(path.rsplit('/', 1)[0], 'foregroundImages_cropped')
        else:
            path_fg = os.path.join(path.rsplit('/', 1)[0], 'foregroundImages')
        if not os.path.exists(path_fg):
            os.makedirs(path_fg)

        # Write each foreground object in each frame to an image
        images = []
        annotations = []
        for img_id, annotation in enumerate(data['annotations'], start=1):
            old_img = next(image for image in data['images'] if image['id'] == annotation['image_id'])
            file_name = old_img['file_name']
            old_img_path = os.path.join(root, 'JPEGImages', file_name)
            if crop_bbx:
                image = Image.open(old_img_path)
                transform = transforms.ToTensor()
                img_tensor = transform(image)
                bbox = annotation['bbox']
                # Crop img
                if coco:
                    cropped = transform(np.array(image)[int(bbox[1]):int(bbox[1]) + int(bbox[3]),
                                        int(bbox[0]):int(bbox[0]) + int(bbox[2])])
                else:
                    cropped = crop(img_tensor, int(annotation['bbox'][1]), int(annotation['bbox'][0]),
                                   int(annotation['bbox'][3]), int(annotation['bbox'][2]))
                # Convert segmentation to binary mask
                if ovis or coco:
                    segm = annotation['segmentation']
                    if type(segm) == list:
                        # polygon -- a single object might consist of multiple parts
                        # we merge all parts into one mask rle code
                        rles = mask.frPyObjects(segm, old_img['height'], old_img['width'])
                        rle = mask.merge(rles)
                    elif type(segm['counts']) == list:
                        # uncompressed RLE
                        rle = mask.frPyObjects(segm, old_img['height'], old_img['width'])
                    else:
                        # rle
                        rle = segm
                else:
                    rle = mask.frPyObjects(annotation['segmentation'], annotation['segmentation'].get('size')[0],
                                           annotation['segmentation'].get('size')[1])
                binary_mask = torch.from_numpy(mask.decode(rle))
                # Crop binary mask
                binary_cropped = crop(binary_mask, int(annotation['bbox'][1]), int(annotation['bbox'][0]),
                                      int(annotation['bbox'][3]), int(annotation['bbox'][2]))

                # Get foreground
                fg = get_foreground_from_tensors(cropped, binary_cropped)

                # Convert mask to RLE
                rle_mask = binary_mask_to_rle(binary_cropped.detach().numpy().astype('uint8'))
            else:
                fg = get_foreground(old_img_path, annotation['segmentation'])

            if ovis:
                new_file_name = file_name.replace('img_', '').rsplit('.', 1)[0] + '_' + str(
                    annotation['object_id'] + num_yvis_objects) + '.' + file_name.rsplit('.', 1)[1]
            elif coco:
                object_id = img_id + num_yvis_objects + num_ovis_objects
                old_file_name = file_name.rsplit('.', 1)[0]
                new_file_name = old_file_name + '_' + str(object_id) + '.' + file_name.rsplit('.', 1)[1]
            else:
                new_file_name = file_name.replace('img_', '').rsplit('.', 1)[0] + '_' + str(
                    annotation['object_id']) + '.' + file_name.rsplit('.', 1)[1]

            # Write to foreground dir
            new_img_path = os.path.join(path_fg, new_file_name)
            if not os.path.exists(new_img_path.rsplit('/', 1)[0]):
                os.makedirs(new_img_path.rsplit('/', 1)[0])
            try:
                save_image(fg, new_img_path)
            except ValueError:
                continue

            images.append({'id': img_id,
                           'video_id': old_img['video_id'] if not coco else 0,
                           'frame_id': old_img['frame_id'] if not coco else 0,
                           'width': int(annotation['bbox'][2]) if crop_bbx else old_img['width'],
                           'height': int(annotation['bbox'][3]) if crop_bbx else old_img['height'],
                           'file_name': new_file_name})

            annotations.append({'id': annotation['id'],
                                'object_id': annotation['object_id'] if not coco else object_id,
                                'video_id': annotation['video_id'] if not coco else 0,
                                'image_id': img_id,
                                'category_id': annotation['category_id'],
                                'segmentation': rle_mask if crop_bbx else annotation['segmentation'],
                                'area': annotation['area'],
                                'bbox': [0.0, 0.0, annotation['bbox'][2], annotation['bbox'][3]] if crop_bbx
                                else annotation['bbox'],
                                'bbox_prev': annotation['bbox'],
                                'iscrowd': annotation['iscrowd']})

        fg_json = {'images': images,
                   'categories': data['categories'],
                   'annotations': annotations}

        if crop_bbx:
            out_path = path.rsplit('/', 1)[0] + '/fg_instances_cropped.json'
        else:
            out_path = path.rsplit('/', 1)[0] + '/fg_instances.json'
        with open(out_path, 'w') as outfile:
            json.dump(fg_json, outfile)

        # When crop_bbx=True, also save fg_instances.json with uncropped metadata
        if crop_bbx:
            uncropped_images = []
            uncropped_annotations = []
            for img, ann in zip(images, annotations):
                uncropped_images.append({
                    'id': img['id'],
                    'video_id': img['video_id'],
                    'frame_id': img['frame_id'],
                    'width': next(image['width'] for image in data['images'] if image['id'] == next(a['image_id'] for a in data['annotations'] if a['id'] == ann['id'])),
                    'height': next(image['height'] for image in data['images'] if image['id'] == next(a['image_id'] for a in data['annotations'] if a['id'] == ann['id'])),
                    'file_name': img['file_name']
                })
                uncropped_annotations.append({
                    'id': ann['id'],
                    'object_id': ann['object_id'],
                    'video_id': ann['video_id'],
                    'image_id': ann['image_id'],
                    'category_id': ann['category_id'],
                    'segmentation': next(a['segmentation'] for a in data['annotations'] if a['id'] == ann['id']),
                    'area': ann['area'],
                    'bbox': ann['bbox_prev'],
                    'iscrowd': ann['iscrowd']
                })

            uncropped_json = {
                'images': uncropped_images,
                'categories': data['categories'],
                'annotations': uncropped_annotations
            }
            uncropped_path = path.rsplit('/', 1)[0] + '/fg_instances.json'
            with open(uncropped_path, 'w') as outfile:
                json.dump(uncropped_json, outfile)


def remove_empty_annotations(path):
    with open(path, 'r') as f:
        data = json.load(f)

    new_data = data
    new_data['annotations'] = [annotation for annotation in data['annotations'] if
                               annotation['segmentation'] is not None]

    with open(path, 'w') as outfile:
        json.dump(new_data, outfile)


def get_foreground(image_path, rle_mask):
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(image)

    compressed_rle = mask.frPyObjects(rle_mask, rle_mask.get('size')[0], rle_mask.get('size')[1])
    binary_mask = torch.from_numpy(mask.decode(compressed_rle))
    return torch.mul(img_tensor, binary_mask)


def get_foreground_from_tensors(img_tensor: torch.Tensor, binary_mask: torch.Tensor):
    return torch.mul(img_tensor, binary_mask)


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(int(len(list(elements))))
    return rle


def setup(vis_root, ovis_root):
    vis_img_instances = to_coco_img(os.path.join(vis_root, 'instances.json'))
    ovis_img_instances = to_coco_img(os.path.join(ovis_root, 'annotations_train.json'))

    remove_empty_annotations(vis_img_instances)
    split(vis_img_instances)

    setup_foreground_data(vis_root)
    setup_foreground_data(ovis_root, ovis=True)

    os.rename(os.path.join(ovis_root, 'train'), os.path.join(ovis_root, 'test'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup instance search dataset')
    parser.add_argument('--vis', type=str, default='datasets/vis',
                        help='Root directory of youtube-vis dataset')
    parser.add_argument('--ovis', type=str, default='datasets/ovis',
                        help='Root directory of ovis distractors')
    args = parser.parse_args()

    # SET UP FOREGROUND DATA
    setup(args.vis, args.ovis)
