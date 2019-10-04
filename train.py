import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_norm_

from dataset import TBNDataSet
from models import TBN
from transforms import *
from opts import parser
from tensorboardX import SummaryWriter
from datetime import datetime
from collections import OrderedDict
from epic_kitchens.meta import training_labels

best_prec1 = 0
training_iterations = 0
best_loss = 10000000

args = parser.parse_args()
lr_steps_str = list(map(lambda k: str(int(k)), args.lr_steps))
experiment_name = '_'.join((args.dataset, args.arch,
                            ''.join(args.modality).lower(),
                            'lr' + str(args.lr),
                            'lr_st' + '_'.join(lr_steps_str),
                            'dr' + str(args.dropout),
                            'ep' + str(args.epochs),
                            'segs' + str(args.num_segments),
                            args.experiment_suffix))
experiment_dir = os.path.join(experiment_name, datetime.now().strftime('%b%d_%H-%M-%S'))
log_dir = os.path.join('runs', experiment_dir)
summaryWriter = SummaryWriter(logdir=log_dir)

def main():
    global args, best_prec1, train_list, experiment_dir, best_loss
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'epic':
        num_class = (125, 352)
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TBN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                midfusion=args.midfusion)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    data_length = model.new_length
    # policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    # Resume training from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            state_dict_new = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                state_dict_new[k.split('.', 1)[1]] = v
            model.load_state_dict(state_dict_new)
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    # Load pretrained weights for each stream
    if args.pretrained_flow_weights:
        print('Initialize Flow stream from Kinetics')
        pretrained = os.path.join('pretrained/kinetics_tsn_flow.pth.tar')
        state_dict = torch.load(pretrained)
        for k, v in state_dict.items():
            state_dict[k] = torch.squeeze(v, dim=0)
        base_model = getattr(model, 'flow')
        base_model.load_state_dict(state_dict, strict=False)

    # Freeze stream weights (leaves only fusion and classification trainable)
    if args.freeze:
        model.freeze_fn('modalities')

    # Freeze batch normalisation layers except the first
    if args.partialbn:
        model.freeze_fn('partialbn_parameters')

    model = torch.nn.DataParallel(model, device_ids=args.gpus).to(device)

    cudnn.benchmark = True

    # Data loading code
    normalize = {}
    for m in args.modality:
        if (m != 'Spec'):
            if (m != 'RGBDiff'):
                normalize[m] = GroupNormalize(input_mean[m], input_std[m])
            else:
                normalize[m] = IdentityTransform()


    image_tmpl = {}
    train_transform = {}
    val_transform = {}
    for m in args.modality:
        if (m != 'Spec'):
            # Prepare dictionaries containing image name templates for each modality
            if m in ['RGB', 'RGBDiff']:
                image_tmpl[m] = "img_{:010d}.jpg"
            elif m == 'Flow':
                image_tmpl[m] = args.flow_prefix + "{}_{:010d}.jpg"
            # Prepare train/val dictionaries containing the transformations
            # (augmentation+normalization)
            # for each modality
            train_transform[m] = torchvision.transforms.Compose([
                train_augmentation[m],
                Stack(roll=args.arch == 'BNInception'),
                ToTorchFormatTensor(div=args.arch != 'BNInception'),
                normalize[m],
            ])

            val_transform[m] = torchvision.transforms.Compose([
                GroupScale(int(scale_size[m])),
                GroupCenterCrop(crop_size[m]),
                Stack(roll=args.arch == 'BNInception'),
                ToTorchFormatTensor(div=args.arch != 'BNInception'),
                normalize[m],
            ])
        else:
            # Prepare train/val dictionaries containing the transformations
            # (augmentation+normalization)
            # for each modality
            train_transform[m] = torchvision.transforms.Compose([
                Stack(roll=args.arch == 'BNInception'),
                ToTorchFormatTensor(div=False),
            ])

            val_transform[m] = torchvision.transforms.Compose([
                Stack(roll=args.arch == 'BNInception'),
                ToTorchFormatTensor(div=False),
            ])

    if args.train_list is None:
        # If train_list is not provided, we train on the default
        # dataset which is all the training set
        train_loader = torch.utils.data.DataLoader(
            TBNDataSet(args.dataset,
                       training_labels(),
                       data_length,
                       args.modality,
                       image_tmpl,
                       visual_path=args.visual_path,
                       audio_path=args.audio_path,
                       num_segments=args.num_segments,
                       transform=train_transform,
                       fps=args.fps,
                       resampling_rate=args.resampling_rate),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            TBNDataSet(args.dataset,
                       args.train_list,
                       data_length,
                       args.modality,
                       image_tmpl,
                       visual_path=args.visual_path,
                       audio_path=args.audio_path,
                       num_segments=args.num_segments,
                       transform=train_transform,
                       fps=args.fps,
                       resampling_rate=args.resampling_rate),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    if args.train_list is not None:
        # we cannot validate on part of the training set
        # if we use all the training set for training
        val_loader = torch.utils.data.DataLoader(
            TBNDataSet(args.dataset,
                       args.val_list,
                       data_length,
                       args.modality,
                       image_tmpl,
                       visual_path=args.visual_path,
                       audio_path=args.audio_path,
                       num_segments=args.num_segments,
                       mode='val',
                       transform=val_transform,
                       fps=args.fps,
                       resampling_rate=args.resampling_rate),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    if len(args.modality) > 1:
        param_groups = [
                        {'params': filter(lambda p: p.requires_grad, model.module.rgb.parameters())},
                        {'params': filter(lambda p: p.requires_grad, model.module.flow.parameters()), 'lr': 0.001},
                        {'params': filter(lambda p: p.requires_grad, model.module.spec.parameters())},
                        {'params': filter(lambda p: p.requires_grad, model.module.fusion_classification_net.parameters())},
                       ]
    else:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.SGD(param_groups,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optimizer, args.lr_steps, gamma=0.1)
    if args.evaluate:
        validate(val_loader, model, criterion, device)
        return
    if args.save_stats:
        if args.dataset != 'epic':
            stats_dict = {'train_loss': np.zeros((args.epochs,)),
                          'val_loss': np.zeros((args.epochs,)),
                          'train_acc': np.zeros((args.epochs,)),
                          'val_acc': np.zeros((args.epochs,))}
        elif args.dataset == 'epic':
            if args.train_list is not None:
                stats_dict = {'train_loss': np.zeros((args.epochs,)),
                              'train_verb_loss': np.zeros((args.epochs,)),
                              'train_noun_loss': np.zeros((args.epochs,)),
                              'train_acc': np.zeros((args.epochs,)),
                              'train_verb_acc': np.zeros((args.epochs,)),
                              'train_noun_acc': np.zeros((args.epochs,)),
                              'val_loss': np.zeros((args.epochs,)),
                              'val_verb_loss': np.zeros((args.epochs,)),
                              'val_noun_loss': np.zeros((args.epochs,)),
                              'val_acc': np.zeros((args.epochs,)),
                              'val_verb_acc': np.zeros((args.epochs,)),
                              'val_noun_acc': np.zeros((args.epochs,))}
            else:
                stats_dict = {'train_loss': np.zeros((args.epochs,)),
                              'train_verb_loss': np.zeros((args.epochs,)),
                              'train_noun_loss': np.zeros((args.epochs,)),
                              'train_acc': np.zeros((args.epochs,)),
                              'train_verb_acc': np.zeros((args.epochs,)),
                              'train_noun_acc': np.zeros((args.epochs,))}

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        # train for one epoch
        training_metrics = train(train_loader, model, criterion, optimizer, epoch, device)
        if args.save_stats:
            for k, v in training_metrics.items():
                stats_dict[k][epoch] = v
        # evaluate on validation set
        if args.train_list is not None:
            if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
                test_metrics = validate(val_loader, model, criterion, device)
                if args.save_stats:
                    for k, v in test_metrics.items():
                        stats_dict[k][epoch] = v
                prec1 = test_metrics['val_acc']
                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                }, is_best)
        else:  #  No validation set
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': training_metrics['train_acc'],
            }, False)

    summaryWriter.close()

    if args.save_stats:
        save_stats_dir = os.path.join('stats', experiment_dir)
        if not os.path.exists(save_stats_dir):
            os.makedirs(save_stats_dir)
        with open(os.path.join(save_stats_dir, 'training_stats.npz'), 'wb') as f:
            np.savez(f, **stats_dict)


def train(train_loader, model, criterion, optimizer, epoch, device):
    global training_iterations

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if args.dataset == 'epic':
        verb_losses = AverageMeter()
        noun_losses = AverageMeter()
        verb_top1 = AverageMeter()
        verb_top5 = AverageMeter()
        noun_top1 = AverageMeter()
        noun_top5 = AverageMeter()

    # switch to train mode
    model.train()

    if args.partialbn:
        model.module.freeze_fn('partialbn_statistics')
    if args.freeze:
        model.module.freeze_fn('bn_statistics')

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)

        for m in args.modality:
            input[m] = input[m].to(device)

        # compute output
        output = model(input)
        batch_size = input[args.modality[0]].size(0)
        if args.dataset != 'epic':
            target = target.to(device)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1,5))
        else:
            target = {k: v.to(device) for k, v in target.items()}
            loss_verb = criterion(output[0], target['verb'])
            loss_noun = criterion(output[1], target['noun'])
            loss = 0.5 * (loss_verb + loss_noun)
            verb_losses.update(loss_verb.item(), batch_size)
            noun_losses.update(loss_noun.item(), batch_size)

            verb_output = output[0]
            noun_output = output[1]
            verb_prec1, verb_prec5 = accuracy(verb_output, target['verb'], topk=(1, 5))
            verb_top1.update(verb_prec1, batch_size)
            verb_top5.update(verb_prec5, batch_size)

            noun_prec1, noun_prec5 = accuracy(noun_output, target['noun'], topk=(1, 5))
            noun_top1.update(noun_prec1, batch_size)
            noun_top5.update(noun_prec5, batch_size)

            prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                                              (target['verb'], target['noun']),
                                              topk=(1, 5))

        losses.update(loss.item(), batch_size)
        top1.update(prec1, batch_size)
        top5.update(prec5, batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        training_iterations += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            summaryWriter.add_scalars('data/loss', {
                'training': losses.avg,
            }, training_iterations)
            summaryWriter.add_scalar('data/epochs', epoch, training_iterations)
            summaryWriter.add_scalar('data/learning_rate', optimizer.param_groups[-1]['lr'], training_iterations)
            summaryWriter.add_scalars('data/precision/top1', {
                'training': top1.avg,
            }, training_iterations)
            summaryWriter.add_scalars('data/precision/top5', {
                'training': top5.avg
            }, training_iterations)

            if args.dataset != 'epic':

                message = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                           'Time {batch_time.avg:.3f} ({batch_time.avg:.3f})\t'
                           'Data {data_time.avg:.3f} ({data_time.avg:.3f})\t'
                           'Loss {loss.avg:.4f} ({loss.avg:.4f})\t'
                           'Prec@1 {top1.avg:.3f} ({top1.avg:.3f})\t'
                           'Prec@5 {top5.avg:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5,
                        lr=optimizer.param_groups[-1]['lr']))
            else:
                summaryWriter.add_scalars('data/verb/loss', {
                    'training': verb_losses.avg,
                }, training_iterations)
                summaryWriter.add_scalars('data/noun/loss', {
                    'training': noun_losses.avg,
                }, training_iterations)
                summaryWriter.add_scalars('data/verb/precision/top1', {
                    'training': verb_top1.avg,
                }, training_iterations)
                summaryWriter.add_scalars('data/verb/precision/top5', {
                    'training': verb_top5.avg
                }, training_iterations)
                summaryWriter.add_scalars('data/noun/precision/top1', {
                    'training': noun_top1.avg,
                }, training_iterations)
                summaryWriter.add_scalars('data/noun/precision/top5', {
                    'training': noun_top5.avg
                }, training_iterations)

                message = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' +
                           'Time {batch_time.avg:.3f} ({batch_time.avg:.3f})\t' +
                           'Data {data_time.avg:.3f} ({data_time.avg:.3f})\t' +
                           'Loss {loss.avg:.4f} ({loss.avg:.4f})\t' +
                           'Verb Loss {verb_loss.avg:.4f} ({verb_loss.avg:.4f})\t' +
                           'Noun Loss {noun_loss.avg:.4f} ({noun_loss.avg:.4f})\t' +
                           'Prec@1 {top1.avg:.3f} ({top1.avg:.3f})\t' +
                           'Prec@5 {top5.avg:.3f} ({top5.avg:.3f})\t' +
                           'Verb Prec@1 {verb_top1.avg:.3f} ({verb_top1.avg:.3f})\t' +
                           'Verb Prec@5 {verb_top5.avg:.3f} ({verb_top5.avg:.3f})\t' +
                           'Noun Prec@1 {noun_top1.avg:.3f} ({noun_top1.avg:.3f})\t' +
                           'Noun Prec@5 {noun_top5.avg:.3f} ({noun_top5.avg:.3f})'
                           ).format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, verb_loss=verb_losses,
                    noun_loss=noun_losses, top1=top1, top5=top5,
                    verb_top1=verb_top1, verb_top5=verb_top5,
                    noun_top1=noun_top1, noun_top5=noun_top5, lr=optimizer.param_groups[-1]['lr'])

            print(message)
    if args.dataset != 'epic':
        training_metrics = {'train_loss': losses.avg, 'train_acc': top1.avg}
    else:
        training_metrics = {'train_loss': losses.avg,
                            'train_noun_loss': noun_losses.avg,
                            'train_verb_loss': verb_losses.avg,
                            'train_acc': top1.avg,
                            'train_verb_acc': verb_top1.avg,
                            'train_noun_acc': noun_top1.avg}
    return training_metrics


def validate(val_loader, model, criterion, device):
    global training_iterations

    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        if args.dataset == 'epic':
            verb_losses = AverageMeter()
            noun_losses = AverageMeter()
            verb_top1 = AverageMeter()
            verb_top5 = AverageMeter()
            noun_top1 = AverageMeter()
            noun_top5 = AverageMeter()
        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            for m in args.modality:
                input[m] = input[m].to(device)

            # compute output
            output = model(input)
            batch_size = input[args.modality[0]].size(0)
            if args.dataset != 'epic':
                target = target.to(device)
                loss = criterion(output, target)
                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, target, topk=(1,5))
            else:
                target = {k: v.to(device) for k, v in target.items()}
                loss_verb = criterion(output[0], target['verb'])
                loss_noun = criterion(output[1], target['noun'])
                loss = 0.5 * (loss_verb + loss_noun)
                verb_losses.update(loss_verb.item(), batch_size)
                noun_losses.update(loss_noun.item(), batch_size)

                verb_output = output[0]
                noun_output = output[1]
                verb_prec1, verb_prec5 = accuracy(verb_output, target['verb'], topk=(1, 5))
                verb_top1.update(verb_prec1, batch_size)
                verb_top5.update(verb_prec5, batch_size)

                noun_prec1, noun_prec5 = accuracy(noun_output, target['noun'], topk=(1, 5))
                noun_top1.update(noun_prec1, batch_size)
                noun_top5.update(noun_prec5, batch_size)

                prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                                                  (target['verb'], target['noun']),
                                                  topk=(1, 5))

            losses.update(loss.item(), batch_size)
            top1.update(prec1, batch_size)
            top5.update(prec5, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if args.dataset != 'epic':
            summaryWriter.add_scalars('data/loss', {
                'validation': losses.avg,
            }, training_iterations)
            summaryWriter.add_scalars('data/precision/top1', {
                'validation': top1.avg,
            }, training_iterations)
            summaryWriter.add_scalars('data/precision/top5', {
                'validation': top5.avg
            }, training_iterations)

            message = ('Testing Results: '
                       'Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} '
                       'Loss {loss.avg:.5f}').format(top1=top1,
                                                     top5=top5,
                                                     loss=losses)
        else:
            summaryWriter.add_scalars('data/loss', {
                'validation': losses.avg,
            }, training_iterations)
            summaryWriter.add_scalars('data/precision/top1', {
                'validation': top1.avg,
            }, training_iterations)
            summaryWriter.add_scalars('data/precision/top5', {
                'validation': top5.avg
            }, training_iterations)
            summaryWriter.add_scalars('data/verb/loss', {
                'validation': verb_losses.avg,
            }, training_iterations)
            summaryWriter.add_scalars('data/noun/loss', {
                'validation': noun_losses.avg,
            }, training_iterations)
            summaryWriter.add_scalars('data/verb/precision/top1', {
                'validation': verb_top1.avg,
            }, training_iterations)
            summaryWriter.add_scalars('data/verb/precision/top5', {
                'validation': verb_top5.avg
            }, training_iterations)
            summaryWriter.add_scalars('data/noun/precision/top1', {
                'validation': noun_top1.avg,
            }, training_iterations)
            summaryWriter.add_scalars('data/noun/precision/top5', {
                'validation': noun_top5.avg
            }, training_iterations)

            message = ("Testing Results: "
                       "Verb Prec@1 {verb_top1.avg:.3f} Verb Prec@5 {verb_top5.avg:.3f} "
                       "Noun Prec@1 {noun_top1.avg:.3f} Noun Prec@5 {noun_top5.avg:.3f} "
                       "Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} "
                       "Verb Loss {verb_loss.avg:.5f} "
                       "Noun Loss {noun_loss.avg:.5f} "
                       "Loss {loss.avg:.5f}").format(verb_top1=verb_top1, verb_top5=verb_top5,
                                                     noun_top1=noun_top1, noun_top5=noun_top5,
                                                     top1=top1, top5=top5,
                                                     verb_loss=verb_losses,
                                                     noun_loss=noun_losses,
                                                     loss=losses)

        print(message)
        if args.dataset != 'epic':
            test_metrics = {'val_loss': losses.avg, 'val_acc': top1.avg}
        else:
            test_metrics = {'val_loss': losses.avg,
                            'val_noun_loss': noun_losses.avg,
                            'val_verb_loss': verb_losses.avg,
                            'val_acc': top1.avg,
                            'val_verb_acc': verb_top1.avg,
                            'val_noun_acc': noun_top1.avg}
        return test_metrics


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    global experiment_dir
    weights_dir = os.path.join('models', experiment_dir)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    torch.save(state, os.path.join(weights_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(weights_dir, filename),
                        os.path.join(weights_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).to(torch.float32).sum(0)
        res.append(float(correct_k.mul_(100.0 / batch_size)))
    return tuple(res)


def multitask_accuracy(outputs, labels, topk=(1,)):
    """
    Args:
        outputs: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
        topk: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = int(np.max(topk))
    task_count = len(outputs)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    if torch.cuda.is_available():
        all_correct = all_correct.cuda()
    for output, label in zip(outputs, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    accuracies = []
    for k in topk:
        all_tasks_correct = torch.ge(all_correct[:k].float().sum(0), task_count)
        accuracy_at_k = float(all_tasks_correct.float().sum(0) * 100.0 / batch_size)
        accuracies.append(accuracy_at_k)
    return tuple(accuracies)


if __name__ == '__main__':
    main()
