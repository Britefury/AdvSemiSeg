import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from PIL import Image
import settings



class AbstractAccessor (data.Dataset):
    def __init__(self, ds, files, crop_size, scale, mirror, mean=(128, 128, 128), std=(1, 1, 1)):
        super(AbstractAccessor, self).__init__()
        self.ds = ds
        self.files = files
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.mirror = mirror
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label


class AccessorXY (AbstractAccessor):
    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image = (image - self.mean) / self.std
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ds.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name


class AccessorY (AbstractAccessor):
    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]

        attempt = 0
        while attempt < 10 :
            if self.scale:
                image, label = self.generate_scale_label(image, label)

            img_h, img_w = label.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                attempt += 1
                continue
            else:
                break

        if attempt == 10 :
            image = cv2.resize(image, (self.crop_w, self.crop_h), interpolation = cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.crop_w, self.crop_h), interpolation = cv2.INTER_NEAREST)


        image = np.asarray(image, np.float32)
        image = (image - self.mean) / self.std

        img_h, img_w = label.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(image[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name


class VOCDataSet(object):
    def __init__(self, ignore_label=255):
        self.root = settings.get_config_dir('pascal_voc')
        self.ignore_label = ignore_label
        self._train_files = self.file_list(os.path.join(self.root, 'ImageSets', 'SegmentationAug', 'train_aug.txt'))
        self._val_files = self.file_list(os.path.join(self.root, 'ImageSets', 'SegmentationAug', 'val.txt'))
        self.num_classes = 21

        self.class_names = ['background',  # always index 0
                            'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog', 'horse',
                            'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor']

    def file_list(self, list_path):
        img_ids = [i_id.strip() for i_id in open(list_path)]
        files = []
        # for split in ["train", "trainval", "val"]:
        for name in img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        return files

    def train_xy(self, crop_size=(321, 321), scale=True, mirror=True, mean=(128, 128, 128), std=(1, 1, 1)):
        return AccessorXY(self, self._train_files, crop_size=crop_size, scale=scale, mirror=mirror, mean=mean, std=std)

    def train_y(self, crop_size=(321, 321), scale=True, mirror=True, mean=(128, 128, 128), std=(1, 1, 1)):
        return AccessorY(self, self._train_files, crop_size=crop_size, scale=scale, mirror=mirror, mean=mean, std=std)

    def val_xy(self, crop_size=(321, 321), scale=False, mirror=False, mean=(128, 128, 128), std=(1, 1, 1)):
        return AccessorXY(self, self._train_files, crop_size=crop_size, scale=scale, mirror=mirror, mean=mean, std=std)



if __name__ == '__main__':
    dst = VOCDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
