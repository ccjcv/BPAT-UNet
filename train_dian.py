import argparse
import glob
import os
import random
import socket
import time
from datetime import datetime
import math
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
# PyTorch includes
import torch
import torch.optim as optim
# Tensorboard include
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
# Dataloaders includes
from dataloaders import tn3k, tg3k, tatn, tn3k_point
from dataloaders import custom_transforms_2 as trforms
from dataloaders import utils
from our_model.BPATUNet_all import BPATUNet
from utils import soft_dice
#zhi biao
from dataloaders.utils import get_dice
from dataloaders.utils import cal_HD_2
import torch.nn.functional as F
def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.6,  #0.8
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="mean")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    ## Model settings
    parser.add_argument('-model_name', type=str,
                        default='BPAT-UNet')  # unet, trfe, trfe1, trfe2, mtnet, segnet, deeplab-resnet50, fcn
    parser.add_argument('-criterion', type=str, default='Dice')
    parser.add_argument('-pretrain', type=str, default='None')  # THYROID

    parser.add_argument('-num_classes', type=int, default=1)
    parser.add_argument('-input_size', type=int, default=256)#segformer256
    parser.add_argument('-output_stride', type=int, default=16)

    ## Train settings
    parser.add_argument('-dataset', type=str, default='TN3K_point')  # TN3K, TG3K, TATN
    parser.add_argument('-fold', type=str, default='0')
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-nepochs', type=int, default=150)
    parser.add_argument('-resume_epoch', type=int, default=0)

    ## Optimizer settings
    parser.add_argument('-naver_grad', type=str, default=1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-update_lr_every', type=int, default=10)
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--warm-up-epochs", default=5, type=int)

    ## Visualization settings
    parser.add_argument('-save_every', type=int, default=10)
    parser.add_argument('-log_every', type=int, default=40)
    parser.add_argument('-load_path', type=str, default='')
    parser.add_argument('-run_id', type=int, default=-1)
    parser.add_argument('-use_eval', type=int, default=1)
    parser.add_argument('-use_test', type=int, default=1)
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1234)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if args.resume_epoch != 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    if args.run_id >= 0:
        run_id = args.run_id

    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
    log_dir = os.path.join(save_dir, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    batch_size = args.batch_size


    if 'BPAT-UNet' in args.model_name:
        net = BPATUNet(n_classes=1)
    else:
        raise NotImplementedError

    if args.resume_epoch == 0:
        print('Training ' + args.model_name + ' from scratch...')
    else:
        load_path = os.path.join(save_dir, args.model_name + '_epoch-' + str(args.resume_epoch) + '.pth')
        print('Initializing weights from: {}...'.format(load_path))
        net.load_state_dict(torch.load(load_path))

    if args.pretrain == 'THYROID':
        net.load_state_dict(torch.load('/home/jiang/ccj_dl/code/TRFE-Net_ori/quanzhong_2/unet_3_3_best.pth', map_location=lambda storage, loc: storage))
        print('loading pretrain model......')

    torch.cuda.set_device(device=0)
    net.cuda()

    # optimizer = optim.SGD(
    #     net.parameters(),
    #     lr=args.lr,
    #     momentum=args.momentum
    # )
    params_to_optimize = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr,
        weight_decay=0.0001)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    warm_up_cosine_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.5 * (
            math.cos((epoch - args.warm_up_epochs) / (args.nepochs - args.warm_up_epochs) * math.pi) + 1)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_cosine_lr)

    if args.criterion == 'Dice':
        criterion = soft_dice
    else:
        raise NotImplementedError

    composed_transforms_tr = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.RandomHorizontalFlip(),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    if args.dataset == 'TN3K':
        train_data = tn3k.TN3K(mode='train', transform=composed_transforms_tr, fold=args.fold)
        val_data = tn3k.TN3K(mode='val', transform=composed_transforms_ts, fold=args.fold)
    elif args.dataset == 'TN3K_point':
        train_data = tn3k_point.TN3K(mode='train', transform=composed_transforms_tr, fold=args.fold)
        val_data = tn3k_point.TN3K(mode='val', transform=composed_transforms_ts, fold=args.fold)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

    num_iter_tr = len(trainloader)
    num_iter_ts = len(testloader)
    nitrs = args.resume_epoch * num_iter_tr
    nsamples = args.resume_epoch * len(train_data)
    print('nitrs: %d num_iter_tr: %d' % (nitrs, num_iter_tr))
    print('nsamples: %d tot_num_samples: %d' % (nsamples, len(train_data)))

    aveGrad = 0
    global_step = 0
    recent_losses = []
    start_t = time.time()

    best_f, cur_f = 0.0, 0.0
    for epoch in range(args.resume_epoch, args.nepochs):
        net.train()
        epoch_losses = []
        for ii, sample_batched in enumerate(trainloader):
            if 'trfe' in args.model_name or args.model_name == 'mtnet':
                nodules, glands = sample_batched
                inputs_n, labels_n = nodules['image'].cuda(), nodules['label'].cuda()
                inputs_g, labels_g = glands['image'].cuda(), glands['label'].cuda()
                inputs = torch.cat([inputs_n[0].unsqueeze(0), inputs_g[0].unsqueeze(0)], dim=0)

                for i in range(1, inputs_n.size()[0]):
                    inputs = torch.cat([inputs, inputs_n[i].unsqueeze(0)], dim=0)
                    inputs = torch.cat([inputs, inputs_g[i].unsqueeze(0)], dim=0)

                global_step += inputs.data.shape[0]
                nodule, thyroid = net.forward(inputs)
                loss = 0
                for i in range(inputs.size()[0]):
                    if i % 2 == 0:
                        loss += criterion(nodule[i], labels_n[int(i / 2)], size_average=False, batch_average=True)
                    else:
                        loss += 0.5 * criterion(thyroid[i], labels_g[int((i-1) / 2)], size_average=False, batch_average=True)

            else:
                inputs, labels = sample_batched['image'].cuda(), sample_batched['label'].cuda()
                point = sample_batched['point'].cuda()
                # point = point.cpu().numpy()
                global_step += inputs.data.shape[0]

                outputs, bian = net.forward(inputs)
                # outputs_train = torch.sigmoid(outputs)

                loss_1 = criterion(outputs, labels, size_average=False, batch_average=True)
                # loss_2 = focal_loss(bian, point)
                loss_2 = criterion(bian, point, size_average=False, batch_average=True)
                loss = loss_1 + loss_2
                outputs_train = torch.sigmoid(outputs)
                bian_train = torch.sigmoid(bian)
                bian_show = bian_train[0]
                outputs_show = outputs_train[0]
                writer.add_image('point', bian_show, ii)
                writer.add_image('pre', outputs_show, ii)

            trainloss = loss.item()
            epoch_losses.append(trainloss)
            if len(recent_losses) < args.log_every:
                recent_losses.append(trainloss)
            else:
                recent_losses[nitrs % len(recent_losses)] = trainloss

            # Backward the averaged gradient
            #loss.backward()
            aveGrad += 1
            nitrs += 1
            nsamples += args.batch_size

            # Update the weights once in p['nAveGrad'] forward passes
            # if aveGrad % args.naver_grad == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            #     aveGrad = 0

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()

            if nitrs % args.log_every == 0:
                meanloss = sum(recent_losses) / len(recent_losses)
                print('epoch: %d ii: %d trainloss: %.2f timecost:%.2f secs' % (
                    epoch, ii, meanloss, time.time() - start_t))
                writer.add_scalar('data/trainloss', meanloss, nsamples)


        meanloss = sum(epoch_losses) / len(epoch_losses)
        print('epoch: %d meanloss: %.2f' % (epoch, meanloss))
        writer.add_scalar('data/epochloss', meanloss, nsamples)


        if args.use_test == 1:
            prec_lists = []
            recall_lists = []
            sum_testloss = 0.0
            total_mae = 0.0
            cnt = 0
            count = 0
            jac = 0

            dsc = 0
            HD = 0
            HD_2 = 0


            if args.use_eval == 1:
                net.eval()
            for ii, sample_batched in enumerate(testloader):
                inputs, labels = sample_batched['image'].cuda(), sample_batched['label'].cuda()
                point = sample_batched['point'].cuda()
                with torch.no_grad():
                    if 'trfe' in args.model_name or args.model_name == 'mtnet':
                        outputs, _ = net.forward(inputs)
                    else:
                        outputs, bian = net.forward(inputs)
                        # outputs_val = torch.sigmoid(outputs)
                        bian_val = torch.sigmoid(bian)

                loss_1 = criterion(outputs, labels, size_average=False, batch_average=True)
                # loss_2 = focal_loss(bian, point)
                loss_2 = criterion(bian_val, point, size_average=False, batch_average=True)
                loss = loss_1 + loss_2
                sum_testloss += loss.item()

                predictions = torch.sigmoid(outputs)
                # predictions = outputs_val

                jac += utils.get_iou(predictions, labels)
                count += 1

                total_mae += utils.get_mae(predictions, labels) * predictions.size(0)
                prec_list, recall_list = utils.get_prec_recall(predictions, labels)
                prec_lists.extend(prec_list)
                recall_lists.extend(recall_list)
                cnt += predictions.size(0)

                #dsc
                dsc += get_dice(predictions, labels)
                # HD += cal_HD(predictions, labels)
                HD_2 += cal_HD_2(predictions, labels)



                if ii % num_iter_ts == num_iter_ts - 1:
                    mmae = total_mae / cnt
                    mean_testloss = sum_testloss / num_iter_ts
                    mean_prec = sum(prec_lists) / len(prec_lists)
                    mean_recall = sum(recall_lists) / len(recall_lists)
                    fbeta = 1.3 * mean_prec * mean_recall / (0.3 * mean_prec + mean_recall)
                    jac = jac / count
                    dsc = dsc / count
                    HD = HD / count
                    HD_2 = HD_2 / count


                    print('Validation:')
                    print('epoch: %d, numImages: %d testloss: %.2f mmae: %.4f fbeta: %.4f jac: %.4f' % (
                        epoch, cnt, mean_testloss, mmae, fbeta, jac))
                    print('dsc: %.4f, HD_2: %.4f, prec: %.4f, recall: %.4f' % (dsc, HD_2, mean_prec, mean_recall))
                    writer.add_scalar('data/validloss', mean_testloss, nsamples)
                    writer.add_scalar('data/validmae', mmae, nsamples)
                    writer.add_scalar('data/validfbeta', fbeta, nsamples)
                    writer.add_scalar('data/validjac', jac, epoch)
                    writer.add_scalar('data/validdsc', dsc, epoch)
                    writer.add_scalar('data/validHD', HD, epoch)
                    writer.add_scalar('data/validHD_2', HD_2, epoch)
                    writer.add_scalar('data/validprec', mean_prec, epoch)
                    writer.add_scalar('data/validrecall', mean_recall, epoch)
                    # bian = bian[0]
                    # bian = bian.cpu().numpy()
                    # writer.add_image('point', bian[0], ii)
                    # ii += 1


                    cur_f = jac
                    if cur_f > best_f:
                        save_path = os.path.join(save_dir, args.model_name + '_best' + '.pth')
                        torch.save(net.state_dict(), save_path)
                        print("Save model at {}\n".format(save_path))
                        best_f = cur_f


        if epoch % args.save_every == args.save_every - 1:
            save_path = os.path.join(save_dir, args.model_name + '_epoch-' + str(epoch) + '.pth')
            torch.save(net.state_dict(), save_path)
            print("Save model at {}\n".format(save_path))


if __name__ == "__main__":
    args = get_arguments()
    main(args)
