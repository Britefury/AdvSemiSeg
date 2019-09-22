import os
import numpy as np
import settings
from dataset import abstract_dataset


CLASS_NAMES_WITH_VOID = [
    'unlabeled', 'ego_vehicle', 'rectification_border', 'out_of_roi', 'static', 'dynamic', 'ground',

    'road', 'sidewalk', 'parking', 'rail_track',

    'building', 'wall', 'fence', 'guard_rail', 'bridge', 'tunnel',

    'pole', 'pole_group', 'traffic_light', 'traffic_sign',

    'vegetation', 'terrain', 'sky',

    'person', 'rider',

    'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle',

    'license_plate'
]

VOID_CLASS_NAMES = [
    'unlabeled', 'ego_vehicle', 'rectification_border', 'out_of_roi', 'static', 'dynamic', 'ground',

    'parking', 'rail_track',

    'guard_rail', 'bridge', 'tunnel',

    'pole_group',

    'caravan', 'trailer',

    'license_plate'
]

VOID_CLASS_INDICES = np.array([CLASS_NAMES_WITH_VOID.index(name) for name in VOID_CLASS_NAMES])

CLASS_NAMES = [name for name in CLASS_NAMES_WITH_VOID if name not in VOID_CLASS_NAMES]



def _get_cityscapes_path(exists=False):
    return settings.get_config_dir('cityscapes', exists=exists)

class CityscapesDataSet(abstract_dataset.ZipDataset):
    def __init__(self, ignore_label=255):
        super(CityscapesDataSet, self).__init__(_get_cityscapes_path())
        self.ignore_label = ignore_label

        sample_names = set()

        for filename in self.zip_file.namelist():
            x_name, ext = os.path.splitext(filename)
            if x_name.endswith('_x') and ext.lower() == '.png':
                sample_name = x_name[:-2]
                sample_names.add(sample_name)

        sample_names = list(sample_names)
        sample_names.sort()

        self._train_files = [dict(img='{}_x.png'.format(name), label='{}_y.png'.format(name), name=name)
                             for name in sample_names if name.startswith('train/')]
        self._val_files = [dict(img='{}_x.png'.format(name), label='{}_y.png'.format(name), name=name)
                           for name in sample_names if name.startswith('val/')]

        self.class_names_with_void = CLASS_NAMES_WITH_VOID
        self.class_names = CLASS_NAMES
        self.void_class_names = VOID_CLASS_NAMES
        self.num_classes = len(self.class_names)

        # Make a mapping array to map the class indices over, skipping void classes
        self.non_void_mapping = []
        out_cls_i = 0
        for cls_i, name in enumerate(self.class_names_with_void):
            if name in self.void_class_names:
                self.non_void_mapping.append(255)
            else:
                self.non_void_mapping.append(out_cls_i)
                out_cls_i += 1
        self.non_void_mapping = np.array(self.non_void_mapping)


    def read_label_image(self, file_list_entry):
        y = super(CityscapesDataSet, self).read_label_image(file_list_entry)
        return self.non_void_mapping[y]



    def train_xy(self, crop_size=(512, 1024), scale=True, mirror=True, range01=False, mean=(128, 128, 128), std=(1, 1, 1)):
        return abstract_dataset.AccessorXY(self, self._train_files, crop_size=crop_size, scale=scale, mirror=mirror, range01=range01, mean=mean, std=std)

    def train_y(self, crop_size=(512, 1024), scale=True, mirror=True, range01=False, mean=(128, 128, 128), std=(1, 1, 1)):
        return abstract_dataset.AccessorY(self, self._train_files, crop_size=crop_size, scale=scale, mirror=mirror, range01=range01, mean=mean, std=std)

    def val_xy(self, crop_size=(512, 1024), scale=False, mirror=False, range01=False, mean=(128, 128, 128), std=(1, 1, 1)):
        return abstract_dataset.AccessorXY(self, self._val_files, crop_size=crop_size, scale=scale, mirror=mirror, range01=range01, mean=mean, std=std)
