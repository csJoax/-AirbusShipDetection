# %% import
from config import *

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from imageio import imread
import matplotlib.pyplot as plt

import os
# import gc

# %% data dir
print('data dir(%s):' % (data_root))
print(os.listdir(data_root))


def show_train_test():
    print('train images dir(%s):' % (train_dir))
    train = os.listdir(train_dir)
    print(len(train))

    print('test images dir(%s):' % (test_dir))
    test = os.listdir(test_dir)
    print(len(test))


# %% submission
# submission = pd.read_csv(submission_csv)
# submission.head()


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def show_example(image_id):
    masks = pd.read_csv(train_csv)

    img = imread(train_dir + image_id)
    img_masks = masks.loc[masks['ImageId'] ==
                          image_id, 'EncodedPixels'].tolist()
    print(img_masks)

    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768))
    for mask in img_masks:
        if mask is np.nan:
            print('no ships.')
            return
        all_masks += rle_decode(mask)

    fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(all_masks)
    axarr[2].imshow(img)
    axarr[2].imshow(all_masks, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.show()


# %%
show_example('000fd9827.jpg')

# %%
pass


def write_small_csv():
    all_train = pd.read_csv(train_csv)
    imageids = os.listdir(train_dir)
    small_train = all_train.loc[all_train['ImageId'].isin(imageids)]
    small_train.to_csv('train_ship_segmentations_v2_small.csv', index=False)


# write_small_csv()

# %%
