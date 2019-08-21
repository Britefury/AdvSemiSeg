import click

@click.command()
@click.option("--arch", type=click.Choice(['deeplab2', 'unet_resnet50']), default='deeplab2', help="available options : deeplab2/unet_resnet50")
@click.option("--dataset", type=click.Choice(['pascal_aug']), default='pascal_aug', help="available options : pascal_aug")
@click.option("--batch-size", type=int, default=10,
                    help="Number of images sent to the network in one step.")
@click.option("--iter-size", type=int, default=1,
                    help="Accumulate gradients for ITER_SIZE iterations.")
@click.option("--num-workers", type=int, default=4,
                    help="number of workers for multithread dataloading.")
@click.option("--partial-data", type=float, default=0.5,
                    help="The index of the label to ignore during the training.")
@click.option("--partial-id", type=str, default=None,
                    help="restore partial id list")
@click.option("--ignore-label", type=int, default=255,
                    help="The index of the label to ignore during the training.")
@click.option("--crop-size", type=str, default='321,321',
                    help="Comma-separated string with height and width of images.")
@click.option("--eval-crop-size", type=str, default='505,505',
                    help="Comma-separated string with height and width of images.")
@click.option("--is-training", is_flag=True, default=False,
                    help="Whether to updates the running means and variances during the training.")
@click.option("--learning-rate", type=float, default=2.5e-4,
                    help="Base learning rate for training with polynomial decay.")
@click.option("--learning-rate-d", type=float, default=1e-4,
                    help="Base learning rate for discriminator.")
@click.option("--lambda-adv-pred", type=float, default=0.1,
                    help="lambda_adv for adversarial training.")
@click.option("--lambda-semi", type=float, default=0.1,
                    help="lambda_semi for adversarial training.")
@click.option("--lambda-semi-adv", type=float, default=0.001,
                    help="lambda_semi for adversarial training.")
@click.option("--mask-t", type=float, default=0.2,
                    help="mask T for semi adversarial training.")
@click.option("--semi-start", type=int, default=5000,
                    help="start semi learning after # iterations")
@click.option("--semi-start-adv", type=int, default=0,
                    help="start semi learning after # iterations")
@click.option("--d-remain", type=bool, default=True,
                    help="Whether to train D with unlabeled data")
@click.option("--momentum", type=float, default=0.9,
                    help="Momentum component of the optimiser.")
@click.option("--not-restore-last", is_flag=True, default=False,
                    help="Whether to not restore last (FC) layers.")
@click.option("--num-steps", type=int, default=20000,
                    help="Number of training steps.")
@click.option("--power", type=float, default=0.9,
                    help="Decay parameter to compute the learning rate.")
@click.option("--random-mirror", is_flag=True, default=False,
                    help="Whether to randomly mirror the inputs during the training.")
@click.option("--random-scale", is_flag=True, default=False,
                    help="Whether to randomly scale the inputs during the training.")
@click.option("--random-seed", type=int, default=1234,
                    help="Random seed to have reproducible results.")
@click.option("--restore-from", type=str, default='http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth',
                    help="Where restore model parameters from.")
@click.option("--restore-from-d", type=str, default=None,
                    help="Where restore model parameters from.")
@click.option("--eval-every", type=int, default=1000,
                    help="Evaluate every n iters.")
@click.option("--save-snapshot-every", type=int, default=5000,
                    help="Save checkpoint every often.")
@click.option("--snapshot-dir", type=str, default=None,
                    help="Where to save snapshots of the model.")
@click.option("--weight-decay", type=float, default=0.0005,
                    help="Regularisation parameter for L2-loss.")
@click.option("--device", type=str, default='cuda:0',
                    help="choose gpu device.")
def train(arch, dataset, batch_size, iter_size, num_workers, partial_data, partial_id, ignore_label,
          crop_size, eval_crop_size, is_training, learning_rate, learning_rate_d, lambda_adv_pred, lambda_semi, lambda_semi_adv, mask_t, semi_start, semi_start_adv,
          d_remain, momentum, not_restore_last, num_steps, power, random_mirror, random_scale, random_seed, restore_from, restore_from_d,
          eval_every, save_snapshot_every, snapshot_dir, weight_decay, device):
    import cv2
    import torch
    import torch.nn as nn
    from torch.utils import data, model_zoo
    import numpy as np
    import pickle
    from torch.autograd import Variable
    import torch.optim as optim
    import torch.nn.functional as F
    import scipy.misc
    import torch.backends.cudnn as cudnn
    import sys
    import os
    import os.path as osp
    import pickle

    from model.deeplab import Res_Deeplab
    from model.unet import unet_resnet50
    from model.discriminator import FCDiscriminator
    from utils.loss import CrossEntropy2d, BCEWithLogitsLoss2d
    from utils.evaluation import EvaluatorIoU
    from dataset.voc_dataset import VOCDataSet
    
    
    torch_device = torch.device(device)
    
    
    
    import matplotlib.pyplot as plt
    import random
    import timeit
    start = timeit.default_timer()
    


    if dataset == 'pascal_aug':
        ds = VOCDataSet()
    else:
        print('Dataset {} not yet supported'.format(dataset))
        return




    def loss_calc(pred, label):
        """
        This function returns cross entropy loss for semantic segmentation
        """
        # out shape batch_size x channels x h x w -> batch_size x channels x h x w
        # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
        label = label.long().to(torch_device)
        criterion = CrossEntropy2d()
    
        return criterion(pred, label)
    
    
    def lr_poly(base_lr, iter, max_iter, power):
        return base_lr*((1-float(iter)/max_iter)**(power))
    
    
    def adjust_learning_rate(optimizer, i_iter):
        lr = lr_poly(learning_rate, i_iter, num_steps, power)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1 :
            optimizer.param_groups[1]['lr'] = lr * 10
    
    def adjust_learning_rate_D(optimizer, i_iter):
        lr = lr_poly(learning_rate_d, i_iter, num_steps, power)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1 :
            optimizer.param_groups[1]['lr'] = lr * 10
    
    def one_hot(label):
        label = label.numpy()
        one_hot = np.zeros((label.shape[0], ds.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
        for i in range(ds.num_classes):
            one_hot[:,i,...] = (label==i)
        #handle ignore labels
        return torch.tensor(one_hot, dtype=torch.float, device=torch_device)
    
    def make_D_label(label, ignore_mask):
        ignore_mask = np.expand_dims(ignore_mask, axis=1)
        D_label = np.ones(ignore_mask.shape)*label
        D_label[ignore_mask] = ignore_label
        D_label = torch.tensor(D_label, dtype=torch.float, device=torch_device)
    
        return D_label
    
    
    h, w = map(int, eval_crop_size.split(','))
    eval_crop_size = (h, w)

    h, w = map(int, crop_size.split(','))
    crop_size = (h, w)

    cudnn.enabled = True

    # create network
    if arch == 'deeplab2':
        model = Res_Deeplab(num_classes=ds.num_classes)
    elif arch == 'unet_resnet50':
        model = unet_resnet50(num_classes=ds.num_classes)
    else:
        print('Architecture {} not supported'.format(arch))
        return

    # load pretrained parameters
    if restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(restore_from)
    else:
        saved_state_dict = torch.load(restore_from)

    # only copy the params that exist in current model (caffe-like)
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        print(name)
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
            print('copy {}'.format(name))
    model.load_state_dict(new_params)


    model.train()
    model = model.to(torch_device)

    cudnn.benchmark = True

    # init D
    model_D = FCDiscriminator(num_classes=ds.num_classes)
    if restore_from_d is not None:
        model_D.load_state_dict(torch.load(restore_from_d))
    model_D.train()
    model_D = model_D.to(torch_device)


    if snapshot_dir is not None:
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)


    ds_train_xy = ds.train_xy(crop_size=crop_size, scale=random_scale, mirror=random_mirror, mean=model.MEAN, std=model.STD)
    ds_train_y = ds.train_y(crop_size=crop_size, scale=random_scale, mirror=random_mirror, mean=model.MEAN, std=model.STD)
    ds_val_xy = ds.val_xy(crop_size=eval_crop_size, scale=False, mirror=False, mean=model.MEAN, std=model.STD)

    train_dataset_size = len(ds_train_xy)

    if partial_data == 1.0:
        trainloader = data.DataLoader(ds_train_xy,
                        batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)

        trainloader_gt = data.DataLoader(ds_train_y,
                        batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
    else:
        #sample partial data
        partial_size = int(partial_data * train_dataset_size)

        if partial_id is not None:
            train_ids = pickle.load(open(partial_id))
            print('loading train ids from {}'.format(partial_id))
        else:
            rng = np.random.RandomState(random_seed)
            train_ids = list(range(train_dataset_size))
            rng.shuffle(train_ids)

        if snapshot_dir is not None:
            pickle.dump(train_ids, open(osp.join(snapshot_dir, 'train_id.pkl'), 'wb'))

        train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
        train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
        train_gt_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])

        trainloader = data.DataLoader(ds_train_xy,
                        batch_size=batch_size, sampler=train_sampler, num_workers=3, pin_memory=True)
        trainloader_remain = data.DataLoader(ds_train_xy,
                        batch_size=batch_size, sampler=train_remain_sampler, num_workers=3, pin_memory=True)
        trainloader_gt = data.DataLoader(ds_train_y,
                        batch_size=batch_size, sampler=train_gt_sampler, num_workers=3, pin_memory=True)

        trainloader_remain_iter = enumerate(trainloader_remain)

    testloader = data.DataLoader(ds_val_xy, batch_size=1, shuffle=False, pin_memory=True)

    trainloader_iter = enumerate(trainloader)
    trainloader_gt_iter = enumerate(trainloader_gt)


    # implement model.optim_parameters(args) to handle different models' lr setting

    # optimizer for segmentation network
    optimizer = optim.SGD(model.optim_parameters(learning_rate),
                lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=learning_rate_d, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    # loss/ bilinear upsampling
    bce_loss = BCEWithLogitsLoss2d()


    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    loss_seg_value = 0
    loss_adv_pred_value = 0
    loss_D_value = 0
    loss_semi_value = 0
    loss_semi_adv_value = 0

    for i_iter in range(num_steps):

        model.train()
        model.freeze_batchnorm()

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        for sub_i in range(iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # do semi first
            if (lambda_semi > 0 or lambda_semi_adv > 0 ) and i_iter >= semi_start_adv :
                try:
                    _, batch =  next(trainloader_remain_iter)
                except:
                    trainloader_remain_iter = enumerate(trainloader_remain)
                    _, batch = next(trainloader_remain_iter)

                # only access to img
                images, _, _, _ = batch
                images = images.float().to(torch_device)


                pred = model(images)
                pred_remain = pred.detach()

                D_out = model_D(F.softmax(pred, dim=1))
                D_out_sigmoid = F.sigmoid(D_out).data.cpu().numpy().squeeze(axis=1)

                ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool)

                loss_semi_adv = lambda_semi_adv * bce_loss(D_out, make_D_label(gt_label, ignore_mask_remain))
                loss_semi_adv = loss_semi_adv/iter_size

                #loss_semi_adv.backward()
                loss_semi_adv_value += float(loss_semi_adv)/lambda_semi_adv

                if lambda_semi <= 0 or i_iter < semi_start:
                    loss_semi_adv.backward()
                    loss_semi_value = 0
                else:
                    # produce ignore mask
                    semi_ignore_mask = (D_out_sigmoid < mask_t)

                    semi_gt = pred.data.cpu().numpy().argmax(axis=1)
                    semi_gt[semi_ignore_mask] = ignore_label

                    semi_ratio = 1.0 - float(semi_ignore_mask.sum())/semi_ignore_mask.size
                    print('semi ratio: {:.4f}'.format(semi_ratio))

                    if semi_ratio == 0.0:
                        loss_semi_value += 0
                    else:
                        semi_gt = torch.FloatTensor(semi_gt)

                        loss_semi = lambda_semi * loss_calc(pred, semi_gt)
                        loss_semi = loss_semi/iter_size
                        loss_semi_value += float(loss_semi)/lambda_semi
                        loss_semi += loss_semi_adv
                        loss_semi.backward()

            else:
                loss_semi = None
                loss_semi_adv = None

            # train with source

            try:
                _, batch = next(trainloader_iter)
            except:
                trainloader_iter = enumerate(trainloader)
                _, batch = next(trainloader_iter)

            images, labels, _, _ = batch
            images = images.float().to(torch_device)
            ignore_mask = (labels.numpy() == ignore_label)
            pred = model(images)

            loss_seg = loss_calc(pred, labels)

            D_out = model_D(F.softmax(pred, dim=1))

            loss_adv_pred = bce_loss(D_out, make_D_label(gt_label, ignore_mask))

            loss = loss_seg + lambda_adv_pred * loss_adv_pred

            # proper normalization
            loss = loss/iter_size
            loss.backward()
            loss_seg_value += float(loss_seg)/iter_size
            loss_adv_pred_value += float(loss_adv_pred)/iter_size


            # train D

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with pred
            pred = pred.detach()

            if d_remain:
                pred = torch.cat((pred, pred_remain), 0)
                ignore_mask = np.concatenate((ignore_mask,ignore_mask_remain), axis = 0)

            D_out = model_D(F.softmax(pred, dim=1))
            loss_D = bce_loss(D_out, make_D_label(pred_label, ignore_mask))
            loss_D = loss_D/iter_size/2
            loss_D.backward()
            loss_D_value += float(loss_D)


            # train with gt
            # get gt labels
            try:
                _, batch = next(trainloader_gt_iter)
            except:
                trainloader_gt_iter = enumerate(trainloader_gt)
                _, batch = next(trainloader_gt_iter)

            _, labels_gt, _, _ = batch
            D_gt_v = one_hot(labels_gt)
            ignore_mask_gt = (labels_gt.numpy() == ignore_label)

            D_out = model_D(D_gt_v)
            loss_D = bce_loss(D_out, make_D_label(gt_label, ignore_mask_gt))
            loss_D = loss_D/iter_size/2
            loss_D.backward()
            loss_D_value += float(loss_D)



        optimizer.step()
        optimizer_D.step()

        sys.stdout.write('.')
        sys.stdout.flush()

        if i_iter % eval_every == 0 and i_iter != 0:
            model.eval()
            with torch.no_grad():
                evaluator = EvaluatorIoU(ds.num_classes)
                for index, batch in enumerate(testloader):
                    if index % 100 == 0:
                        print('%d processd' % (index))
                    image, label, size, name = batch
                    size = size[0].numpy()
                    image = image.float().to(torch_device)
                    output = model(image)
                    output = output.cpu().data[0].numpy()

                    output = output[:, :size[0], :size[1]]
                    gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=np.int)

                    output = output.transpose(1, 2, 0)
                    output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

                    evaluator.sample(gt, output, ignore_value=ignore_label)

                    sys.stdout.write('+')
                    sys.stdout.flush()

            per_class_iou = evaluator.score()
            mean_iou = per_class_iou.mean()

            loss_seg_value /= eval_every
            loss_adv_pred_value /= eval_every
            loss_D_value /= eval_every
            loss_semi_value /= eval_every
            loss_semi_adv_value /= eval_every

            sys.stdout.write('\n')

            print(
                'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.3f}, loss_semi = {5:.3f}, loss_semi_adv = {6:.3f}'.format(
                    i_iter, num_steps, loss_seg_value, loss_adv_pred_value, loss_D_value, loss_semi_value,
                    loss_semi_adv_value))

            for i, (class_name, iou) in enumerate(zip(ds.class_names, per_class_iou)):
                print('class {:2d} {:12} IU {:.2f}'.format(i, class_name, iou))

            print('meanIOU: ' + str(mean_iou) + '\n')

            loss_seg_value = 0
            loss_adv_pred_value = 0
            loss_D_value = 0
            loss_semi_value = 0
            loss_semi_adv_value = 0

        if snapshot_dir is not None and i_iter % save_snapshot_every == 0 and i_iter!=0:
            print('taking snapshot ...')
            torch.save(model.state_dict(),osp.join(snapshot_dir, 'VOC_'+str(i_iter)+'.pth'))
            torch.save(model_D.state_dict(),osp.join(snapshot_dir, 'VOC_'+str(i_iter)+'_D.pth'))

    end = timeit.default_timer()
    print(end-start,'seconds')

    if snapshot_dir is not None:
        print('save model ...')
        torch.save(model.state_dict(), osp.join(snapshot_dir, 'VOC_' + str(num_steps) + '.pth'))
        torch.save(model_D.state_dict(), osp.join(snapshot_dir, 'VOC_' + str(num_steps) + '_D.pth'))


if __name__ == '__main__':
    train()
