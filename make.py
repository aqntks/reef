import warnings
warnings.filterwarnings("ignore")

import ast
import os
import json
import pandas as pd
import torch
import importlib
import cv2 

from shutil import copyfile
from tqdm.notebook import tqdm
tqdm.pandas()
from sklearn.model_selection import GroupKFold
from PIL import Image
from string import Template
from IPython.display import display

TRAIN_PATH = './'

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

def get_path(row):
    row['image_path'] = f'{TRAIN_PATH}/train_images/video_{row.video_id}/{row.video_frame}.jpg'
    return row

df = pd.read_csv("train.csv")

# Taken only annotated photos
df["num_bbox"] = df['annotations'].apply(lambda x: str.count(x, 'x'))
df_train = df[df["num_bbox"]>0]

#Annotations
df_train['annotations'] = df_train['annotations'].progress_apply(lambda x: ast.literal_eval(x))
df_train['bboxes'] = df_train.annotations.progress_apply(get_bbox)

#Images resolution
df_train["width"] = 1280
df_train["height"] = 720

#Path of images
df_train = df_train.progress_apply(get_path, axis=1)

kf = GroupKFold(n_splits = 5)
df_train = df_train.reset_index(drop=True)
df_train['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, y = df_train.video_id.tolist(), groups=df_train.sequence)):
    df_train.loc[val_idx, 'fold'] = fold

HOME_DIR = './'
DATASET_PATH = 'yolox_images'

SELECTED_FOLD = 4

for i in tqdm(range(len(df_train))):
    row = df_train.loc[i]
    if row.fold != SELECTED_FOLD:
        copyfile(f'{row.image_path}', f'{HOME_DIR}{DATASET_PATH}/train2017/{row.image_id}.jpg')
    else:
        copyfile(f'{row.image_path}', f'{HOME_DIR}{DATASET_PATH}/val2017/{row.image_id}.jpg')

print(f'Number of training files: {len(os.listdir(f"{HOME_DIR}{DATASET_PATH}/train2017/"))}')
print(f'Number of validation files: {len(os.listdir(f"{HOME_DIR}{DATASET_PATH}/val2017/"))}')

def save_annot_json(json_annotation, filename):
    with open(filename, 'w') as f:
        output_json = json.dumps(json_annotation)
        f.write(output_json)

annotion_id = 0


def dataset2coco(df, dest_path):
    global annotion_id

    annotations_json = {
        "info": [],
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    info = {
        "year": "2021",
        "version": "1",
        "description": "COTS dataset - COCO format",
        "contributor": "",
        "url": "https://kaggle.com",
        "date_created": "2021-11-30T15:01:26+00:00"
    }
    annotations_json["info"].append(info)

    lic = {
        "id": 1,
        "url": "",
        "name": "Unknown"
    }
    annotations_json["licenses"].append(lic)

    classes = {"id": 0, "name": "starfish", "supercategory": "none"}

    annotations_json["categories"].append(classes)

    for ann_row in df.itertuples():

        images = {
            "id": ann_row[0],
            "license": 1,
            "file_name": ann_row.image_id + '.jpg',
            "height": ann_row.height,
            "width": ann_row.width,
            "date_captured": "2021-11-30T15:01:26+00:00"
        }

        annotations_json["images"].append(images)

        bbox_list = ann_row.bboxes

        for bbox in bbox_list:
            b_width = bbox[2]
            b_height = bbox[3]

            # some boxes in COTS are outside the image height and width
            if (bbox[0] + bbox[2] > 1280):
                b_width = bbox[0] - 1280
            if (bbox[1] + bbox[3] > 720):
                b_height = bbox[1] - 720

            image_annotations = {
                "id": annotion_id,
                "image_id": ann_row[0],
                "category_id": 0,
                "bbox": [bbox[0], bbox[1], b_width, b_height],
                "area": bbox[2] * bbox[3],
                "segmentation": [],
                "iscrowd": 0
            }

            annotion_id += 1
            annotations_json["annotations"].append(image_annotations)

    print(f"Dataset COTS annotation to COCO json format completed! Files: {len(df)}")
    return annotations_json


# Convert COTS dataset to JSON COCO
train_annot_json = dataset2coco(df_train[df_train.fold != SELECTED_FOLD], f"{HOME_DIR}{DATASET_PATH}/train2017/")
val_annot_json = dataset2coco(df_train[df_train.fold == SELECTED_FOLD], f"{HOME_DIR}{DATASET_PATH}/val2017/")

# Save converted annotations
save_annot_json(train_annot_json, f"{HOME_DIR}{DATASET_PATH}/annotations/train.json")
save_annot_json(val_annot_json, f"{HOME_DIR}{DATASET_PATH}/annotations/valid.json")