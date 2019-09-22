import os
import io
import numpy as np
from scipy.io import loadmat
import settings
from dataset import abstract_dataset, cityscapes_dataset



def _get_gtav_path(exists=False):
    return settings.get_config_dir('gtav', exists=exists)

class GTAVDataSet(abstract_dataset.ZipDataset):
    GTA_CLASS_MAPPING = {
        # 'ground': 'terrain',
        # 'parking': 'road',
        # 'bridge': 'building',
        # 'tunnel': 'building',
        # 'polegroup': 'pole',
        # 'caravan': 'truck',
        # 'trailer': 'truck',
        # 'license plate': 'car',
        'traffic sign': 'traffic_sign',
        'traffic light': 'traffic_light',
    }

    def __init__(self, ignore_label=255, n_val=0, rng=None):
        super(GTAVDataSet, self).__init__(_get_gtav_path())
        self.ignore_label = ignore_label

        sample_names = set()

        for filename in self.zip_file.namelist():
            x_name, ext = os.path.splitext(filename)
            if x_name.endswith('_x') and ext.lower() == '.png':
                sample_name = x_name[:-2]
                sample_names.add(sample_name)

        sample_names = list(sample_names)
        sample_names.sort()

        if rng is None:
            rng = np.random
        ndx = rng.permutation(len(sample_names))
        if n_val > 0:
            train_ndx, val_ndx = ndx[:-n_val], ndx[-n_val:]
        else:
            train_ndx = ndx
            val_ndx = np.zeros((0,), dtype=int)

        train_names = [sample_names[int(i)] for i in train_ndx]
        val_names = [sample_names[int(i)] for i in val_ndx]

        self._train_files = [dict(img='{}_x.png'.format(name), label='{}_y.png'.format(name), name=name)
                             for name in train_names]
        self._val_files = [dict(img='{}_x.png'.format(name), label='{}_y.png'.format(name), name=name)
                           for name in val_names]

        self.class_names = cityscapes_dataset.CLASS_NAMES
        self.num_classes = len(self.class_names)

        mapping_bytes = self.zip_file.read('mapping.mat')
        mapping_bytes_io = io.BytesIO(mapping_bytes)
        mapping_data = loadmat(mapping_bytes_io)
        cls_name_to_index = {cls_name: i for i, cls_name in enumerate(cityscapes_dataset.CLASS_NAMES)}
        mapping = []
        for v in mapping_data['classes'][0]:
            class_name = str(v[0])
            class_name = self.GTA_CLASS_MAPPING.get(class_name, class_name)
            cls_index = cls_name_to_index.get(class_name, 255)
            # print('Mapping GTA class {} to {}'.format(class_name, cls_index))
            mapping.append(cls_name_to_index.get(class_name, 255))

        self.class_map = np.array(mapping, dtype=int)


    def read_label_image(self, file_list_entry):
        y = super(GTAVDataSet, self).read_label_image(file_list_entry)
        return self.class_map[y]


    def train_xy(self, crop_size=(512, 1024), scale=True, mirror=True, range01=False, mean=(128, 128, 128), std=(1, 1, 1)):
        return abstract_dataset.AccessorXY(self, self._train_files, crop_size=crop_size, scale=scale, mirror=mirror, range01=range01, mean=mean, std=std)

    def train_y(self, crop_size=(512, 1024), scale=True, mirror=True, range01=False, mean=(128, 128, 128), std=(1, 1, 1)):
        return abstract_dataset.AccessorY(self, self._train_files, crop_size=crop_size, scale=scale, mirror=mirror, range01=range01, mean=mean, std=std)

    def val_xy(self, crop_size=(512, 1024), scale=False, mirror=False, range01=False, mean=(128, 128, 128), std=(1, 1, 1)):
        return abstract_dataset.AccessorXY(self, self._val_files, crop_size=crop_size, scale=scale, mirror=mirror, range01=range01, mean=mean, std=std)
