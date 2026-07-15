# training code for Reloc3r
# references: DUSt3R: https://github.com/naver/dust3r


import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from reloc3r.reloc3r_relpose import Reloc3rRelpose
from reloc3r.datasets import get_data_loader  # noqa
from reloc3r.loss import * # noqa: F401

import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

from pdb import set_trace as bb


def get_args_parser():
    parser = argparse.ArgumentParser('Reloc3r training', add_help=False)
    # model and criterion
    parser.add_argument('--model', default="Reloc3rRelpose(img_size=512)",
                        type=str, help="string containing the model to build")
    parser.add_argument('--pretrained', default=None, 
                        type=str, help='path of a starting checkpoint')
    parser.add_argument('--train_criterion', default="RelativeCameraPoseRegression(L21)",
                        type=str, help="train criterion")
    parser.add_argument('--test_criterion', default="RelativeCameraPoseRegression(L21)", 
                        type=str, help="test criterion")

    # dataset
    parser.add_argument('--train_dataset', required=True, 
                        type=str, help="training set")
    parser.add_argument('--test_dataset', default='[None]', 
                        type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=0, 
                        type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, 
                        type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, 
                        type=int, help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=100, 
                        type=int, help="Maximum number of epochs for the scheduler")
    parser.add_argument('--max_train_steps', default=0,
                        type=int, help="Optional max train-loader steps inside each epoch. 0 keeps full epoch training.")
    parser.add_argument('--step_checkpoint_freq', default=0,
                        type=int, help="Optional step checkpoint frequency. 0 disables intra-epoch checkpointing.")
    parser.add_argument('--step_checkpoint_keep_named', default=1,
                        type=int, choices=[0, 1], help="Keep checkpoint-stepXXXXXX.pth when step_checkpoint_freq fires.")
    parser.add_argument('--step_checkpoint_model_only', default=0,
                        type=int, choices=[0, 1], help="Save named step checkpoints without optimizer/scaler state.")
    parser.add_argument('--milestone_steps', default="",
                        type=str, help="Comma-separated completed train steps to keep as checkpoint-stepXXXXXX.pth.")
    parser.add_argument('--milestone_model_only', default=0,
                        type=int, choices=[0, 1], help="Save milestone checkpoints without optimizer/scaler state.")
    parser.add_argument('--resume_fast_forward_batches', default=0,
                        type=int, choices=[0, 1], help="When resuming mid-epoch, skip prior batch indices without loading samples.")

    parser.add_argument('--weight_decay', default=0.05, 
                        type=float, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', default=1e-6, 
                        type=float, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', default=1.5e-4, 
                        type=float, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', default=0., 
                        type=float, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', default=40, 
                        type=int, metavar='N', help='epochs to warmup LR')

    parser.add_argument('--amp', default=0,
                        type=int, choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")

    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--eval_freq', default=1, 
                        type=int, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, 
                        type=int, help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=20, 
                        type=int, help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, 
                        type=int, help='frequence (number of iterations) to print infos while training')

    parser.add_argument('--freeze_encoder', action="store_true", help='freeze encoder')
    parser.add_argument(
        '--freeze_except_angle_specialist',
        action="store_true",
        help='freeze all parameters except angle_specialist modules',
    )
    parser.add_argument(
        '--trainable_policy',
        default="",
        type=str,
        help='Optional selective trainable policy: angle_specialist, pose_head, pose_head_last_decoder1, pose_head_last_decoder2',
    )
    
    # output dir
    parser.add_argument('--output_dir', default='./checkpoints/tmp', 
                        type=str, help="path where to save the output")
    return parser


def main(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()

    print("output_dir: "+args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)


    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # training dataset and loader
    print('Building train dataset {:s}'.format(args.train_dataset))
    #  dataset and loader
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    # model
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    print(f'>> Creating train criterion = {args.train_criterion}')
    train_criterion = eval(args.train_criterion).to(device)
    print(f'>> Creating test criterion = {args.test_criterion or args.train_criterion}')
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        if 'model' in ckpt: 
            ckpt = ckpt['model']
        new_ckpt = dict(ckpt)
        if any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks2'):
                    new_ckpt[key.replace('dec_blocks2', 'dec_blocks')] = value
        ckpt = new_ckpt
        print(model.load_state_dict(ckpt, strict=False))
        del ckpt        # in case it occupies memory
        del new_ckpt    # in case it occupies memory
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=False)
        model_without_ddp = model.module

    if args.freeze_encoder:
        model_without_ddp.freeze_encoder() 
    if args.freeze_except_angle_specialist and args.trainable_policy:
        raise ValueError("--freeze_except_angle_specialist and --trainable_policy are mutually exclusive")
    if args.freeze_except_angle_specialist:
        if not hasattr(model_without_ddp, 'freeze_except_angle_specialist'):
            raise AttributeError('Model does not implement freeze_except_angle_specialist()')
        freeze_summary = model_without_ddp.freeze_except_angle_specialist()
        print(
            "freeze_except_angle_specialist: "
            f"trainable_params={freeze_summary['trainable_params']} "
            f"total_params={freeze_summary['total_params']}"
        )
        print("freeze_except_angle_specialist trainable names:")
        for name in freeze_summary["trainable_names"]:
            print(f"  {name}")
    if args.trainable_policy:
        if not hasattr(model_without_ddp, 'freeze_for_trainable_policy'):
            raise AttributeError('Model does not implement freeze_for_trainable_policy()')
        freeze_summary = model_without_ddp.freeze_for_trainable_policy(args.trainable_policy)
        print(
            "trainable_policy: "
            f"policy={freeze_summary['policy']} "
            f"trainable_params={freeze_summary['trainable_params']} "
            f"total_params={freeze_summary['total_params']}"
        )
        print("trainable_policy trainable names:")
        for name in freeze_summary["trainable_names"][:100]:
            print(f"  {name}")
        if len(freeze_summary["trainable_names"]) > 100:
            print(f"  ... {len(freeze_summary['trainable_names']) - 100} more")
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update({test_name+'_'+k: v for k, v in test_stats[test_name].items()})

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, fname=fname, best_so_far=best_so_far)

    best_so_far = misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)

    best_so_far = float('inf')
    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.start_epoch, args.epochs+1):

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                save_model(epoch-1, 'last', best_so_far)

        # Test on multiple datasets
        new_best = False
        if (epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0):
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(model, test_criterion, testset,
                                       device, epoch, log_writer=log_writer, args=args, prefix=test_name)
                test_stats[test_name] = stats

                # save best checkpoint measured by loss
                if stats['loss_med'] < best_so_far:
                    best_so_far = stats['loss_med']
                    new_best = True

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch-1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch-1, 'best', best_so_far)
        if epoch >= args.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)
        if train_stats.get('_bounded_stop'):
            print("Bounded training stop requested; skipping epoch-level save/eval.")
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'
    to_save = {
        'args': args,
        'model': model_without_ddp if isinstance(model_without_ddp, dict) else model_without_ddp.cpu().state_dict(),
        'epoch': epoch
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader


class _FastForwardBatchSampler:
    def __init__(self, batch_sampler, start_batch):
        self.batch_sampler = batch_sampler
        self.start_batch = max(int(start_batch), 0)

    def __iter__(self):
        import itertools
        return itertools.islice(iter(self.batch_sampler), self.start_batch, None)

    def __len__(self):
        return max(len(self.batch_sampler) - self.start_batch, 0)


def _make_fast_forward_loader(data_loader, start_batch):
    if start_batch <= 0:
        return data_loader
    return torch.utils.data.DataLoader(
        data_loader.dataset,
        batch_sampler=_FastForwardBatchSampler(data_loader.batch_sampler, start_batch),
        num_workers=data_loader.num_workers,
        collate_fn=data_loader.collate_fn,
        pin_memory=data_loader.pin_memory,
        worker_init_fn=data_loader.worker_init_fn,
        persistent_workers=getattr(data_loader, 'persistent_workers', False),
        prefetch_factor=getattr(data_loader, 'prefetch_factor', None),
    )


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args,
                    log_writer=None):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.7f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()
    start_step = getattr(args, 'start_step', 0) if epoch == getattr(args, 'start_epoch', 0) else 0
    effective_data_loader = data_loader
    data_iter_offset = 0
    if start_step:
        print(f"Resume epoch {epoch} from train step {start_step}")
        metric_logger.update(epoch=float(epoch))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if getattr(args, 'resume_fast_forward_batches', 0):
            effective_data_loader = _make_fast_forward_loader(data_loader, start_step)
            data_iter_offset = start_step
            print(f"Fast-forwarding train loader by {start_step} batch indices without loading skipped samples.")
    if args.max_train_steps and start_step >= args.max_train_steps:
        print(f"max_train_steps={args.max_train_steps} already reached at start_step={start_step}; skipping epoch {epoch}.")
        return {'_bounded_stop': 1}

    milestone_steps = set()
    if getattr(args, 'milestone_steps', ''):
        milestone_steps = {int(item) for item in args.milestone_steps.split(',') if item.strip()}

    def save_step_checkpoint(data_iter_step, checkpoint_path=None, model_only=False):
        if not args.output_dir:
            return
        model_to_save = model.module if hasattr(model, 'module') else model
        completed_train_steps = max(data_iter_step + 1, 0)
        if checkpoint_path is None:
            checkpoint_path = Path(args.output_dir) / 'checkpoint-last.pth'
        else:
            checkpoint_path = Path(checkpoint_path)
        to_save = {
            'model': model_to_save.state_dict(),
            'args': args,
            'epoch': epoch,
            'step_in_epoch': data_iter_step,
            'global_train_step': epoch * len(data_loader) + data_iter_step,
            'completed_train_steps': completed_train_steps,
        }
        if not model_only:
            to_save['optimizer'] = optimizer.state_dict()
            to_save['scaler'] = loss_scaler.state_dict()
        print(
            f'>> Saving step checkpoint to {checkpoint_path} '
            f'at epoch={epoch}, step={data_iter_step}, completed_train_steps={completed_train_steps}, '
            f'model_only={int(model_only)} ...'
        )
        misc.save_on_master(to_save, checkpoint_path)

    named_step_checkpoints_saved = set()

    def save_named_step_checkpoint(completed_step, data_iter_step, model_only=False):
        if completed_step in named_step_checkpoints_saved:
            return
        save_step_checkpoint(
            data_iter_step,
            Path(args.output_dir) / f'checkpoint-step{completed_step:06d}.pth',
            model_only=model_only,
        )
        named_step_checkpoints_saved.add(completed_step)

    bounded_stop = False
    if 0 in milestone_steps and start_step == 0 and epoch == getattr(args, 'start_epoch', 0):
        save_named_step_checkpoint(
            0,
            -1,
            model_only=bool(getattr(args, 'milestone_model_only', 0)),
        )
    for local_data_iter_step, batch in enumerate(metric_logger.log_every(effective_data_loader, args.print_freq, header)):
        data_iter_step = local_data_iter_step + data_iter_offset
        if data_iter_step < start_step:
            del batch
            continue
        epoch_f = epoch + data_iter_step / len(data_loader)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        loss_tuple = loss_of_one_batch(batch, model, criterion, device,
                                       use_amp=bool(args.amp), ret='loss')
        loss, loss_details = loss_tuple  # criterion returns two values
        loss_value = float(loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar('train_'+name, val, epoch_1000x)

        completed_step = data_iter_step + 1
        if args.step_checkpoint_freq and completed_step % args.step_checkpoint_freq == 0:
            if getattr(args, 'step_checkpoint_keep_named', 1):
                save_named_step_checkpoint(
                    completed_step,
                    data_iter_step,
                    model_only=bool(getattr(args, 'step_checkpoint_model_only', 0)),
                )
            save_step_checkpoint(data_iter_step)
        if completed_step in milestone_steps:
            save_named_step_checkpoint(
                completed_step,
                data_iter_step,
                model_only=bool(getattr(args, 'milestone_model_only', 0)),
            )

        if args.max_train_steps and completed_step >= args.max_train_steps:
            if getattr(args, 'step_checkpoint_keep_named', 1) and completed_step not in named_step_checkpoints_saved:
                save_named_step_checkpoint(
                    completed_step,
                    data_iter_step,
                    model_only=bool(getattr(args, 'step_checkpoint_model_only', 0)),
                )
            save_step_checkpoint(data_iter_step)
            print(f"Reached max_train_steps={args.max_train_steps}; ending epoch {epoch} early.")
            bounded_stop = True
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if bounded_stop:
        stats['_bounded_stop'] = 1
    return stats


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    for _, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        loss_tuple = loss_of_one_batch(batch, model, criterion, device,
                                       use_amp=bool(args.amp), ret='loss')
        loss_value, loss_details = loss_tuple  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix+'_'+name, val, 1000*epoch)

    return results


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
