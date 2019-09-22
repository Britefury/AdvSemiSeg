import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
import settings
from dataset import abstract_dataset


class VOCDataSet(abstract_dataset.FileSystemDataset):
    def __init__(self, augmented_pascal=True, ignore_label=255):
        super(VOCDataSet, self).__init__(read_labels_with_pillow=not augmented_pascal)
        self.root = settings.get_config_dir('pascal_voc')
        self.augmented_pascal = augmented_pascal
        self.ignore_label = ignore_label
        if augmented_pascal:
            self._train_files = self.file_list(os.path.join(self.root, 'ImageSets', 'SegmentationAug', 'train_aug.txt'))
            self._val_files = self.file_list(os.path.join(self.root, 'ImageSets', 'SegmentationAug', 'val.txt'))
        else:
            self._train_files = self.file_list(os.path.join(self.root, 'ImageSets', 'Segmentation', 'train.txt'))
            self._val_files = self.file_list(os.path.join(self.root, 'ImageSets', 'Segmentation', 'val.txt'))

        self.num_classes = 21

        self.class_names = ['background',  # always index 0
                            'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog', 'horse',
                            'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor']

    def file_list(self, list_path):
        img_ids = [i_id.strip() for i_id in open(list_path)]
        img_ids.sort()
        files = []
        # for split in ["train", "trainval", "val"]:
        for name in img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            if self.augmented_pascal:
                label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            else:
                label_file = osp.join(self.root, "SegmentationClass/%s.png" % name)
            files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        return files

    def train_xy(self, crop_size=(321, 321), scale=True, mirror=True, range01=False, mean=(128, 128, 128), std=(1, 1, 1)):
        return abstract_dataset.AccessorXY(self, self._train_files, crop_size=crop_size, scale=scale, mirror=mirror, range01=range01, mean=mean, std=std)

    def train_y(self, crop_size=(321, 321), scale=True, mirror=True, range01=False, mean=(128, 128, 128), std=(1, 1, 1)):
        return abstract_dataset.AccessorY(self, self._train_files, crop_size=crop_size, scale=scale, mirror=mirror, range01=range01, mean=mean, std=std)

    def val_xy(self, crop_size=(321, 321), scale=False, mirror=False, range01=False, mean=(128, 128, 128), std=(1, 1, 1)):
        return abstract_dataset.AccessorXY(self, self._val_files, crop_size=crop_size, scale=scale, mirror=mirror, range01=range01, mean=mean, std=std)



if __name__ == '__main__':
    dst = VOCDataSet()
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
