## import ##

from itertools import groupby
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import os
import pickle
import cv2
from multiprocessing import Pool
import matplotlib.pyplot as plt
# import cupy as cp
import ast
import glob

import shutil
import sys
sys.path.append('./')

from joblib import Parallel, delayed

from IPython.display import display, HTML

from matplotlib import animation, rc
rc('animation', html='jshtml')

## config ##

FOLD      = 4 # which fold to train
REMOVE_NOBBOX = True # remove images with no bbox
ROOT_DIR  = './'
IMAGE_DIR = './yolo_img/images' # directory to save images
LABEL_DIR = './yolo_img/labels' # directory to save labels

## create directories ##

# !mkdir -p {IMAGE_DIR}
# !mkdir -p {LABEL_DIR}


## get paths ##

def get_path(row):
    row['old_image_path'] = f'{ROOT_DIR}/train_images/video_{row.video_id}/{row.video_frame}.jpg'
    row['image_path'] = f'{IMAGE_DIR}/video_{row.video_id}_{row.video_frame}.jpg'
    row['label_path'] = f'{LABEL_DIR}/video_{row.video_id}_{row.video_frame}.txt'
    return row

# Train Data
df = pd.read_csv(f'{ROOT_DIR}/train.csv')
df = df.progress_apply(get_path, axis=1)
df['annotations'] = df['annotations'].progress_apply(lambda x: ast.literal_eval(x))
display(df.head(2))

## number of bboxes ##

df['num_bbox'] = df['annotations'].progress_apply(lambda x: len(x))
data = (df.num_bbox>0).value_counts(normalize=True)*100
print(f"No BBox: {data[0]:0.2f}% | With BBox: {data[1]:0.2f}%")

## clean data ##

if REMOVE_NOBBOX:
    df = df.query("num_bbox>0")

## write images ##
def make_copy(path):
    data = path.split('/')
    filename = data[-1]
    video_id = data[-2]
    new_path = os.path.join(IMAGE_DIR,f'{video_id}_{filename}')
    shutil.copy(path, new_path)
    return

image_paths = df.old_image_path.tolist()
_ = Parallel(n_jobs=-1, backend='threading')(delayed(make_copy)(path) for path in tqdm(image_paths))

## Helper ##

def voc2yolo(image_height, image_width, bboxes):
    """
    voc  => [x1, y1, x2, y1]
    yolo => [xmid, ymid, w, h] (normalized)
    """

    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / image_height

    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]

    bboxes[..., 0] = bboxes[..., 0] + w / 2
    bboxes[..., 1] = bboxes[..., 1] + h / 2
    bboxes[..., 2] = w
    bboxes[..., 3] = h

    return bboxes


def yolo2voc(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]

    """
    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * image_height

    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]] / 2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]

    return bboxes


def coco2yolo(image_height, image_width, bboxes):
    """
    coco => [xmin, ymin, w, h]
    yolo => [xmid, ymid, w, h] (normalized)
    """

    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    # normolizinig
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / image_height

    # converstion (xmin, ymin) => (xmid, ymid)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]] / 2

    return bboxes


def yolo2coco(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    coco => [xmin, ymin, w, h]

    """
    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    # denormalizing
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * image_height

    # converstion (xmid, ymid) => (xmin, ymin)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]] / 2

    return bboxes


def load_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_bboxes(img, bboxes, classes, class_ids, colors=None, show_classes=None, bbox_format='yolo', class_name=False,
                line_thickness=2):
    image = img.copy()
    show_classes = classes if show_classes is None else show_classes
    colors = (0, 255, 0) if colors is None else colors

    if bbox_format == 'yolo':

        for idx in range(len(bboxes)):

            bbox = bboxes[idx]
            cls = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors

            if cls in show_classes:
                x1 = round(float(bbox[0]) * image.shape[1])
                y1 = round(float(bbox[1]) * image.shape[0])
                w = round(float(bbox[2]) * image.shape[1] / 2)  # w/2
                h = round(float(bbox[3]) * image.shape[0] / 2)

                voc_bbox = (x1 - w, y1 - h, x1 + w, y1 + h)
                plot_one_box(voc_bbox,
                             image,
                             color=color,
                             label=cls if class_name else str(get_label(cls)),
                             line_thickness=line_thickness)

    elif bbox_format == 'coco':

        for idx in range(len(bboxes)):

            bbox = bboxes[idx]
            cls = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors

            if cls in show_classes:
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                w = int(round(bbox[2]))
                h = int(round(bbox[3]))

                voc_bbox = (x1, y1, x1 + w, y1 + h)
                plot_one_box(voc_bbox,
                             image,
                             color=color,
                             label=cls if class_name else str(cls_id),
                             line_thickness=line_thickness)

    elif bbox_format == 'voc_pascal':

        for idx in range(len(bboxes)):

            bbox = bboxes[idx]
            cls = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors

            if cls in show_classes:
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                x2 = int(round(bbox[2]))
                y2 = int(round(bbox[3]))
                voc_bbox = (x1, y1, x2, y2)
                plot_one_box(voc_bbox,
                             image,
                             color=color,
                             label=cls if class_name else str(cls_id),
                             line_thickness=line_thickness)
    else:
        raise ValueError('wrong bbox format')

    return image


def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes


def get_imgsize(row):
    row['width'], row['height'] = imagesize.get(row['image_path'])
    return row


# https://www.kaggle.com/diegoalejogm/great-barrier-reefs-eda-with-animations
def create_animation(ims):
    fig = plt.figure(figsize=(16, 12))
    plt.axis('off')
    im = plt.imshow(ims[0])

    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000 // 12)


np.random.seed(32)
colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) \
          for idx in range(1)]

## create bbox ##
df['bboxes'] = df.annotations.progress_apply(get_bbox)
df.head(2)

## get image size ##

df['width']  = 1280
df['height'] = 720
display(df.head(2))

## create labels ##

cnt = 0
all_bboxes = []
for row_idx in tqdm(range(df.shape[0])):
    row = df.iloc[row_idx]
    image_height = row.height
    image_width  = row.width
    bboxes_coco  = np.array(row.bboxes).astype(np.float32).copy()
    num_bbox     = len(bboxes_coco)
    names        = ['cots']*num_bbox
    labels       = [0]*num_bbox
    ## Create Annotation(YOLO)
    with open(row.label_path, 'w') as f:
        if num_bbox<1:
            annot = ''
            f.write(annot)
            cnt+=1
            continue
        bboxes_yolo  = coco2yolo(image_height, image_width, bboxes_coco)
        bboxes_yolo  = np.clip(bboxes_yolo, 0, 1)
        all_bboxes.extend(bboxes_yolo)
        for bbox_idx in range(len(bboxes_yolo)):
            annot = [str(labels[bbox_idx])]+ list(bboxes_yolo[bbox_idx].astype(str))+(['\n'] if num_bbox!=(bbox_idx+1) else [''])
            annot = ' '.join(annot)
            annot = annot.strip(' ')
            f.write(annot)
print('Missing:',cnt)

## BBox Distribution ##
# from scipy.stats import gaussian_kde
#
# all_bboxes = np.array(all_bboxes)
#
# x_val = all_bboxes[...,0]
# y_val = all_bboxes[...,1]
#
# # Calculate the point density
# xy = np.vstack([x_val,y_val])
# z = gaussian_kde(xy)(xy)
#
# fig, ax = plt.subplots(figsize = (10, 10))
# ax.axis('off')
# ax.scatter(x_val, y_val, c=z, s=100, cmap='viridis')
# # ax.set_xlabel('x_mid')
# # ax.set_ylabel('y_mid')
# plt.show()
##################################################
# x_val = all_bboxes[...,2]
# y_val = all_bboxes[...,3]
#
# # Calculate the point density
# xy = np.vstack([x_val,y_val])
# z = gaussian_kde(xy)(xy)
#
# fig, ax = plt.subplots(figsize = (10, 10))
# ax.axis('off')
# ax.scatter(x_val, y_val, c=z, s=100, cmap='viridis')
# # ax.set_xlabel('bbox_width')
# # ax.set_ylabel('bbox_height')
# plt.show()
##################################################
# import seaborn as sns
# sns.set(style='white')
# areas = all_bboxes[...,2]*all_bboxes[...,3]*720*1280
# plt.figure(figsize=(12,8))
# sns.kdeplot(areas,shade=True,palette='viridis')
# plt.axis('OFF')
# plt.show()

## visualization ##

# df2 = df[(df.num_bbox>0)].sample(100) # takes samples with bbox
# for seq in df.sequence.unique()[:2]:
#     seq_df = df.query("sequence==@seq")
#     images = []
#     for _, row in tqdm(seq_df.iterrows(), total=len(seq_df), desc=f'seq_id-{seq} '):
#         img           = load_image(row.image_path)
#         image_height  = row.height
#         image_width   = row.width
#         bboxes_coco   = np.array(row.bboxes)
#         bboxes_yolo   = coco2yolo(image_height, image_width, bboxes_coco)
#         names         = ['cots']*len(bboxes_coco)
#         labels        = [0]*len(bboxes_coco)
#         img = draw_bboxes(img = img,
#                                bboxes = bboxes_yolo,
#                                classes = names,
#                                class_ids = labels,
#                                class_name = True,
#                                colors = colors,
#                                bbox_format = 'yolo',
#                                line_thickness = 2)
#         images.append(img)
#     display(HTML(f"<h2>Sequence ID: {seq}</h2>"))
#     display(create_animation(images))

## create folds ##

from sklearn.model_selection import GroupKFold
kf = GroupKFold(n_splits = 5)
df = df.reset_index(drop=True)
df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(df, y = df.video_id.tolist(), groups=df.sequence)):
    df.loc[val_idx, 'fold'] = fold
display(df.fold.value_counts())

## dataset ##

train_files = []
val_files   = []
train_df = df.query("fold!=@FOLD")
valid_df = df.query("fold==@FOLD")
train_files += list(train_df.image_path.unique())
val_files += list(valid_df.image_path.unique())
len(train_files), len(val_files)

## configuration ##

import yaml

cwd = './'

with open(os.path.join(cwd, 'train.txt'), 'w') as f:
    for path in train_df.image_path.tolist():
        f.write(path + '\n')

with open(os.path.join(cwd, 'val.txt'), 'w') as f:
    for path in valid_df.image_path.tolist():
        f.write(path + '\n')

data = dict(
    path='./',
    train=os.path.join(cwd, 'train.txt'),
    val=os.path.join(cwd, 'val.txt'),
    nc=1,
    names=['cots'],
)

with open(os.path.join(cwd, 'bgr.yaml'), 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

f = open(os.path.join(cwd, 'bgr.yaml'), 'r')
print('\nyaml:')
print(f.read())