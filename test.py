import os
import argparse
import time

import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix, accuracy_score

from dataset import TBNDataSet
from models import TBN
from transforms import *
import pickle


def average_crops(results, num_crop, num_class):

    return results.cpu().numpy()\
              .reshape((num_crop, args.test_segments, num_class))\
              .mean(axis=0)\
              .reshape((args.test_segments, 1, num_class))


def eval_video(data, net, num_class, device):
    num_crop = args.test_crops

    for m in args.modality:
        data[m] = data[m].to(device)

    rst = net(data)

    if args.dataset != 'epic':
        return average_crops(rst, num_crop, num_class)
    else:
        return {'verb': average_crops(rst[0], num_crop, num_class[0]),
                'noun': average_crops(rst[1], num_crop, num_class[1])}


def evaluate_model(num_class):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = TBN(num_class, 1, args.modality,
              base_model=args.arch,
              consensus_type=args.crop_fusion_type,
              dropout=args.dropout,
              midfusion=args.midfusion)

    weights = '{weights_dir}/model_best.pth.tar'.format(
        weights_dir=args.weights_dir)
    checkpoint = torch.load(weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)

    test_transform = {}
    image_tmpl = {}
    for m in args.modality:
        if m != 'Spec':
            if args.test_crops == 1:
                cropping = torchvision.transforms.Compose([
                    GroupScale(net.scale_size[m]),
                    GroupCenterCrop(net.input_size[m]),
                ])
            elif args.test_crops == 10:
                cropping = torchvision.transforms.Compose([
                    GroupOverSample(net.input_size[m], net.scale_size[m])
                ])
            else:
                raise ValueError("Only 1 and 10 crops are supported" +
                                 " while we got {}".format(args.test_crops))


            test_transform[m] = torchvision.transforms.Compose([
                cropping, Stack(roll=args.arch == 'BNInception'),
                ToTorchFormatTensor(div=args.arch != 'BNInception'),
                GroupNormalize(net.input_mean[m], net.input_std[m]), ])

            # Prepare dictionaries containing image name templates
            # for each modality
            if m in ['RGB', 'RGBDiff']:
                image_tmpl[m] = "img_{:010d}.jpg"
            elif m == 'Flow':
                image_tmpl[m] = args.flow_prefix + "{}_{:010d}.jpg"
        else:

            test_transform[m] = torchvision.transforms.Compose([
                Stack(roll=args.arch == 'BNInception'),
                ToTorchFormatTensor(div=False), ])


    data_length = net.new_length

    if args.dataset != 'epic' or len(args.test_list) == 1:
        test_loader = torch.utils.data.DataLoader(
            TBNDataSet(args.dataset,
                       args.test_list[0],
                       data_length,
                       args.modality,
                       image_tmpl,
                       visual_path=args.visual_path,
                       audio_path=args.audio_path,
                       num_segments=args.test_segments,
                       mode='test',
                       transform=test_transform,
                       fps=args.fps,
                       resampling_rate=args.resampling_rate),
            batch_size=1, shuffle=False,
            num_workers=args.workers * 2)
    else:
        test_seen_loader = torch.utils.data.DataLoader(
            TBNDataSet(args.dataset,
                       args.test_list[0],
                       data_length,
                       args.modality,
                       image_tmpl,
                       visual_path=args.visual_path,
                       audio_path=args.audio_path,
                       num_segments=args.test_segments,
                       mode='test',
                       transform=test_transform,
                       fps=args.fps,
                       resampling_rate=args.resampling_rate),
            batch_size=1, shuffle=False,
            num_workers=args.workers * 2)

        test_unseen_loader = torch.utils.data.DataLoader(
            TBNDataSet(args.dataset,
                       args.test_list[1],
                       data_length,
                       args.modality,
                       image_tmpl,
                       visual_path=args.visual_path,
                       audio_path=args.audio_path,
                       num_segments=args.test_segments,
                       mode='test',
                       transform=test_transform,
                       fps=args.fps,
                       resampling_rate=args.resampling_rate),
            batch_size=1, shuffle=False,
            num_workers=args.workers * 2)

    net = torch.nn.DataParallel(net, device_ids=args.gpus).to(device)
    with torch.no_grad():
        net.eval()
        data_gen_dict = {}
        results_dict = {}
        if args.dataset != 'epic' or len(args.test_list) == 1:
            data_gen_dict['test'] = test_loader
        else:
            data_gen_dict['test_seen'] = test_seen_loader
            data_gen_dict['test_unseen'] = test_unseen_loader

        for split, data_gen in data_gen_dict.items():
            results = []
            total_num = len(data_gen.dataset)

            proc_start_time = time.time()
            max_num = args.max_num if args.max_num > 0 else total_num
            for i, (data, label) in enumerate(data_gen):
                if i >= max_num:
                    break
                rst = eval_video(data, net, num_class, device)
                if label is not None:
                    if args.dataset != 'epic':
                        label_ = label.item()
                    else:
                        label_ = {k: v.item() for k, v in label.items()}
                    results.append((rst, label_))
                else:
                    results.append((rst,))
                cnt_time = time.time() - proc_start_time
                print('video {} done, total {}/{}, average {} sec/video'.format(
                    i, i + 1, total_num, float(cnt_time) / (i + 1)))

            results_dict[split] = results
        return results_dict


def print_accuracy(scores, labels):

    video_pred = [np.argmax(np.mean(score, axis=0)) for score in scores]
    cf = confusion_matrix(labels, video_pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_cnt[cls_hit == 0] = 1  # to avoid divisions by zero
    cls_acc = cls_hit / cls_cnt

    acc = accuracy_score(labels, video_pred)

    print('Accuracy {:.02f}%'.format(acc * 100))
    print('Average Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))


def save_scores(results_dict, scores_dir):

    for split, results in results_dict.items():
        save_dict = {}
        if args.dataset != 'epic':
            scores = np.array([result[0] for result in results])
            labels = np.array([result[1] for result in results])
        else:
            if len(results[0]) == 2:
                keys = results[0][0].keys()
                scores = {k: np.array([result[0][k] for result in results]) for k in keys}
                labels = {k: np.array([result[1][k] for result in results]) for k in keys}
            else:
                keys = results[0][0].keys()
                scores = {k: np.array([result[0][k] for result in results]) for k in keys}
                labels = None

        save_dict[split + '_scores'] = scores
        if labels is not None:
            save_dict[split + '_labels'] = labels

        scores_file = os.path.join(scores_dir, split + '.pkl')
        with open(scores_file, 'wb') as f:
            pickle.dump(save_dict, f)


def main():

    parser = argparse.ArgumentParser(description="Standard video-level" +
                                     " testing")
    parser.add_argument('dataset', type=str,
                        choices=['ucf101', 'hmdb51', 'kinetics', 'epic'])
    parser.add_argument('modality', type=str,
                        choices=['RGB', 'Flow', 'RGBDiff', 'Spec'],
                        nargs='+', default=['RGB', 'Flow', 'Spec'])
    parser.add_argument('weights_dir', type=str)
    parser.add_argument('--test_list', nargs='+')
    parser.add_argument('--visual_path')
    parser.add_argument('--audio_path')
    parser.add_argument('--arch', type=str, default="resnet101")
    parser.add_argument('--scores_root', type=str, default='scores')
    parser.add_argument('--test_segments', type=int, default=25)
    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--test_crops', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='avg',
                        choices=['avg', 'max', 'topk'])
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--flow_prefix', type=str, default='')
    parser.add_argument('--fps', type=float, default=60)
    parser.add_argument('--resampling_rate', type=int, default=24000)
    parser.add_argument('--midfusion', choices=['concat', 'gating_concat', 'multimodal_gating'],
                    default='concat')
    parser.add_argument('--exp_suffix', default='')

    global args
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'beoid':
        num_class = 34
    elif args.dataset == 'epic':
        num_class = (125, 352)
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    results_dict = evaluate_model(num_class)
    experiment_name = os.path.normpath(args.weights_dir).split('/')[-2]
    for split, results in results_dict.items():
        if args.dataset == 'epic':
            if len(results[0]) == 2:
                keys = results[0][0].keys()
                for task in keys:
                    print('Evaluation of {} in {}'.format(task.upper(), split.upper()))
                    print_accuracy([result[0][task] for result in results],
                                   [result[1][task] for result in results])
        else:
            print_accuracy([result[0] for result in results],
                           [result[1] for result in results])

    scores_dir = os.path.join(args.scores_root, experiment_name + args.exp_suffix)
    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)
    save_scores(results_dict, scores_dir)


if __name__ == '__main__':
    main()
