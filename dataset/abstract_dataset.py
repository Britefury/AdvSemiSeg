import numpy as np
import random
import zipfile
import threading
import cv2
from torch.utils import data
from PIL import Image



class AbstractAccessor (data.Dataset):
    def __init__(self, ds, files, crop_size, scale, mirror, range01=False, mean=(128, 128, 128), std=(1, 1, 1)):
        super(AbstractAccessor, self).__init__()
        self.ds = ds
        self.files = files
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.mirror = mirror
        self.range01 = range01
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
        image = self.ds.read_input_image(datafiles)
        label = self.ds.read_label_image(datafiles)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        if self.range01:
            image = (image.astype(np.float32) / 255.0).astype(np.float32)
        else:
            image = image.astype(np.float32)
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

        h_off = random.randint(0, img_h - self.crop_h) if img_h > self.crop_h else 0
        w_off = random.randint(0, img_w - self.crop_w) if img_w > self.crop_w else 0
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))
        if self.mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name


class AccessorY (AbstractAccessor):
    def __getitem__(self, index):
        datafiles = self.files[index]
        image = self.ds.read_input_image(datafiles)
        label = self.ds.read_label_image(datafiles)
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

        if self.range01:
            image = (image.astype(np.float32) / 255.0).astype(np.float32)
        else:
            image = image.astype(np.float32)
        image = (image - self.mean) / self.std

        img_h, img_w = label.shape
        h_off = random.randint(0, img_h - self.crop_h) if img_h > self.crop_h else 0
        w_off = random.randint(0, img_w - self.crop_w) if img_w > self.crop_w else 0
        image = np.asarray(image[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))
        if self.mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name




class AbstractDataset (object):
    def read_input_image(self, file_list_entry):
        raise NotImplementedError('Abstract')

    def read_label_image(self, file_list_entry):
        raise NotImplementedError('Abstract')




class FileSystemDataset (AbstractDataset):
    def __init__(self, read_labels_with_pillow=True):
        self.read_labels_with_pillow =read_labels_with_pillow

    def read_input_image(self, file_list_entry):
        return cv2.imread(file_list_entry["img"], cv2.IMREAD_COLOR)

    def read_label_image(self, file_list_entry):
        path = file_list_entry['label']
        if self.read_labels_with_pillow:
            img = Image.open(path)
            img.load()
            return np.array(img)
        else:
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)



class ZipDataset(AbstractDataset):
    def __init__(self, zip_path):
        self.zip_path = zip_path
        self.zip_file = zipfile.ZipFile(self.zip_path, 'r')

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['zip_file']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.zip_file = zipfile.ZipFile(self.zip_path, 'r')

    def get_pil_image(self, name):
        f_img = self.zip_file.open(name)
        img = Image.open(f_img)
        img.load()
        f_img.close()
        return img

    def read_input_image(self, file_list_entry):
        return np.array(self.get_pil_image(file_list_entry["img"]))

    def read_label_image(self, file_list_entry):
        return np.array(self.get_pil_image(file_list_entry["label"]))
