import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders import custom_transforms as trforms
# Dataloaders includes
from dataloaders import tn3k
from dataloaders import utils
# Custom includes
from visualization.metrics import Metrics, evaluate
from our_model.BPATUNet_all import BPATUNet

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-model_name', type=str,
                        default='BPAT-UNet')  # unet, mtnet, segnet, deeplab-resnet50, fcn, trfe, trfe1, trfe2
    parser.add_argument('-num_classes', type=int, default=1)
    parser.add_argument('-input_size', type=int, default=256)
    parser.add_argument('-output_stride', type=int, default=16)
    parser.add_argument('-load_path', type=str, default='/home/jiang/ccj_dl/code/Our_BPAT-UNet/quanzhong/BPAT-UNet_best.pth')
    parser.add_argument('-save_dir', type=str, default='./results')
    parser.add_argument('-test_dataset', type=str, default='TN3K')
    parser.add_argument('-test_fold', type=str, default='test')
    parser.add_argument('-fold', type=int, default=0)
    ## for transunet
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is 3')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # FIXME add other models
    if 'BPAT-UNet' == args.model_name:
        net = BPATUNet(n_classes=1)
    else:
        raise NotImplementedError
    sta = torch.load(args.load_path)
    missing_keys, unexpected_keys = net.load_state_dict(sta, strict=True)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("mit missing_keys: ", missing_keys)
        print("mit unexpected_keys: ", unexpected_keys)
    net.cuda()

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    if args.test_dataset == 'TN3K':
        test_data = tn3k.TN3K(mode='test', transform=composed_transforms_ts, return_size=True)

    save_dir = args.save_dir + os.sep + args.test_fold + '-' + args.test_dataset + os.sep + args.model_name + os.sep + 'fold' + str(
        args.fold) + os.sep
    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    num_iter_ts = len(testloader)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    net.cuda()
    net.eval()
    with torch.no_grad():
        all_start = time.time()
        metrics = Metrics(['precision', 'recall', 'specificity', 'F1_score'
                              , 'auc', 'acc', 'iou', 'dice', 'mae', 'hd', 'DSC'])
        total_iou = 0
        total_cost_time = 0
        for sample_batched in tqdm(testloader):
            inputs, labels, label_name, size = sample_batched['image'], sample_batched['label'], sample_batched.get(
                'label_name'), sample_batched['size']

            labels = labels.cuda()
            inputs = inputs.cuda()
            if 'trfe' in args.model_name or 'mtnet' in args.model_name:
                if 'trfeplus' in args.model_name:
                    start = time.time()
                    nodule_pred, gland_pred, _ = net.forward(inputs)
                    cost_time = time.time() - start
                else:
                    start = time.time()
                    nodule_pred, gland_pred = net.forward(inputs)
                    cost_time = time.time() - start
                gland_pred = torch.sigmoid(gland_pred)
            elif 'cpfnet' in args.model_name:
                start = time.time()
                nodule_pred = net(inputs)
                cost_time = time.time() - start
            else:
                start = time.time()
                #nodule_pred = net.forward(inputs)
                nodule_pred, bian = net.forward(inputs)
                cost_time = time.time() - start
            prob_pred = torch.sigmoid(nodule_pred)
            iou = utils.get_iou(prob_pred, labels)
            _precision, _recall, _specificity, _f1, _auc, _acc, _iou, _dice, _mae, _hd, _DSC = evaluate(prob_pred, labels)
            metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                           F1_score=_f1, acc=_acc, iou=_iou, mae=_mae, dice=_dice, hd=_hd, auc=_auc,
                           DSC=_DSC)

            total_iou += iou
            total_cost_time += cost_time

            shape = (size[0, 0], size[0, 1])
            prob_pred = F.interpolate(prob_pred, size=shape, mode='bilinear', align_corners=True).cpu().data
            save_data = prob_pred[0]
            save_png = save_data[0].numpy()
            save_saliency = save_png * 255
            save_saliency = save_saliency.astype(np.uint8)

            save_png = np.round(save_png)
            # print(save_png.shape)

            save_png = save_png * 255
            save_png = save_png.astype(np.uint8)
            save_path = save_dir + label_name[0]
            if not os.path.exists(save_path[:save_path.rfind('/')]):
                os.makedirs(save_path[:save_path.rfind('/')])
            save_path_s = save_dir + 's' + label_name[0]
            # cv2.imwrite(save_path_s, save_saliency)
            cv2.imwrite(save_dir + label_name[0], save_png)

    print(args.model_name)
    metrics_result = metrics.mean(len(testloader))
    DSC_new = (2 * metrics_result['iou']) / (1 + metrics_result['iou'])
    print("Test Result:")
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1_score:%.4f, acc: %.4f, '
          'iou: %.4f, mae: %.4f, hd: %.4f, auc: %.4f, DSC: %.4f'
        % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
           metrics_result['F1_score'],
           metrics_result['acc'], metrics_result['iou'], metrics_result['mae'],
           metrics_result['hd'], metrics_result['auc'], DSC_new))
    print("total_cost_time:", total_cost_time)
    print("loop_cost_time:", time.time() - all_start)
    evaluation_dir = os.path.sep.join([args.save_dir, 'metrics', args.test_fold + '-' + args.test_dataset + '/'])
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)

    # keys_txt = ''
    metrics_result['inference_time'] = total_cost_time / len(testloader)
    values_txt = str(args.fold) + '\t'
    for k, v in metrics_result.items():
        if k != 'mae' or k != 'hd':
            v = 100 * v
        # keys_txt += k + '\t'
        values_txt += '%.2f' % v + '\t'
    DSC_new = DSC_new * 100
    values_txt += '%.2f' % DSC_new + '\t'
    text = values_txt + '\n'
    save_path = evaluation_dir + args.model_name + '.txt'
    with open(save_path, 'a+') as f:
        f.write(text)
    print(f'metrics saved in {save_path}')
    print("------------------------------------------------------------------")


if __name__ == '__main__':
    args = get_arguments()
    main(args)
