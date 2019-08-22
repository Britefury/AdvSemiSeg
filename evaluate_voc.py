import click


@click.command()
@click.option("--arch", type=click.Choice(['deeplab2', 'unet_resnet50']), default='deeplab2', help="available options : deeplab2/unet_resnet50")
@click.option("--dataset", type=click.Choice(['pascal_aug']), default='pascal_aug', help="available options : pascal_aug")
@click.option("--ignore-label", type=int, default=255,
                    help="The index of the label to ignore during the training.")
@click.option("--restore-from", type=str, default='http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.125-8d75b3f1.pth',
                    help="Where restore model parameters from.")
@click.option("--pretrained-model", type=str, default=None,
                        help="Where restore model parameters from.")
@click.option("--save-dir", type=str, default='results',
                    help="Directory to store results")
@click.option("--device", type=str, default='cuda:0',
                    help="choose gpu device.")
def evaluate(arch, dataset, ignore_label, restore_from, pretrained_model, save_dir, device):
    import argparse
    import scipy
    from scipy import ndimage
    import cv2
    import numpy as np
    import sys
    from collections import OrderedDict
    import os

    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    import torchvision.models as models
    import torch.nn.functional as F
    from torch.utils import data, model_zoo

    from model.deeplab import Res_Deeplab
    from model.unet import unet_resnet50
    from model.deeplabv3 import resnet101_deeplabv3
    from dataset.voc_dataset import VOCDataSet

    from PIL import Image

    import matplotlib.pyplot as plt

    pretrianed_models_dict ={'semi0.125': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.125-03c6f81c.pth',
                             'semi0.25': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.25-473f8a14.pth',
                             'semi0.5': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.5-acf6a654.pth',
                             'advFull': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSegVOCFull-92fbc7ee.pth'}


    class VOCColorize(object):
        def __init__(self, n=22):
            self.cmap = color_map(22)
            self.cmap = torch.from_numpy(self.cmap[:n])

        def __call__(self, gray_image):
            size = gray_image.shape
            color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

            for label in range(0, len(self.cmap)):
                mask = (label == gray_image)
                color_image[0][mask] = self.cmap[label][0]
                color_image[1][mask] = self.cmap[label][1]
                color_image[2][mask] = self.cmap[label][2]

            # handle void
            mask = (255 == gray_image)
            color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

            return color_image

    def color_map(N=256, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap/255 if normalized else cmap
        return cmap


    def get_iou(data_list, class_num, ignore_label, class_names, save_path=None):
        from multiprocessing import Pool
        from utils.evaluation import EvaluatorIoU

        evaluator = EvaluatorIoU(class_num)
        for truth, prediction in data_list:
            evaluator.sample(truth, prediction, ignore_value=ignore_label)

        per_class_iou = evaluator.score()
        mean_iou = per_class_iou.mean()

        for i, (class_name, iou) in enumerate(zip(class_names, per_class_iou)):
            print('class {:2d} {:12} IU {:.2f}'.format(i, class_name, iou))


        print('meanIOU: ' + str(mean_iou) + '\n')
        if save_path:
            with open(save_path, 'w') as f:
                for i, (class_name, iou) in enumerate(zip(class_names, per_class_iou)):
                    f.write('class {:2d} {:12} IU {:.2f}'.format(i, class_name, iou) + '\n')
                f.write('meanIOU: ' + str(mean_iou) + '\n')

    def show_all(gt, pred):
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, axes = plt.subplots(1, 2)
        ax1, ax2 = axes

        colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5),
                        (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5),
                        (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0),
                        (0.5,0.75,0),(0,0.25,0.5)]
        cmap = colors.ListedColormap(colormap)
        bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        ax1.set_title('gt')
        ax1.imshow(gt, cmap=cmap, norm=norm)

        ax2.set_title('pred')
        ax2.imshow(pred, cmap=cmap, norm=norm)

        plt.show()



    torch_device = torch.device(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    if dataset == 'pascal_aug':
        ds = VOCDataSet()
    else:
        print('Dataset {} not yet supported'.format(dataset))
        return


    if arch == 'deeplab2':
        model = Res_Deeplab(num_classes=ds.num_classes)
    elif arch == 'unet_resnet50':
        model = unet_resnet50(num_classes=ds.num_classes)
    elif arch == 'resnet101_deeplabv3':
        model = resnet101_deeplabv3(num_classes=ds.num_classes)
    else:
        print('Architecture {} not supported'.format(arch))
        return

    if pretrained_model is not None:
        restore_from = pretrianed_models_dict[pretrained_model]

    if restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(restore_from)
    else:
        saved_state_dict = torch.load(restore_from)

    model.load_state_dict(saved_state_dict)

    model.eval()
    model = model.to(torch_device)

    ds_val_xy = ds.val_xy(crop_size=(505, 505), scale=False, mirror=False, mean=model.MEAN, std=model.STD)

    testloader = data.DataLoader(ds_val_xy, batch_size=1, shuffle=False, pin_memory=True)

    data_list = []

    colorize = VOCColorize()

    with torch.no_grad():
        for index, batch in enumerate(testloader):
            if index % 100 == 0:
                print('%d processd'%(index))
            image, label, size, name = batch
            size = size[0].numpy()
            image = torch.tensor(image, dtype=torch.float, device=torch_device)
            output = model(image)
            output = output.cpu().data[0].numpy()

            output = output[:,:size[0],:size[1]]
            gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)

            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

            filename = os.path.join(save_dir, '{}.png'.format(name[0]))
            color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
            color_file.save(filename)

            # show_all(gt, output)
            data_list.append([gt.flatten(), output.flatten()])

        filename = os.path.join(save_dir, 'result.txt')
        get_iou(data_list, ds.num_classes, ignore_label, ds.class_names, filename)


if __name__ == '__main__':
    evaluate()
