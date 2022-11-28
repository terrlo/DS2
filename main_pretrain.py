import json
import os
import time
import torch.cuda.amp as amp
import torch
import torch.distributed as dist
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from contrast import models
from contrast import resnet
from contrast.data import get_loader
from contrast.logger import setup_logger
from contrast.lr_scheduler import get_scheduler
from contrast.pretrain_options import parse_option
from contrast.util import AverageMeter
from contrast.lars import add_weight_decay, LARS

use_gpu = torch.cuda.is_available()
use_amp = use_gpu and False
scaler = amp.GradScaler(enabled=use_amp)


def build_model(args):
    encoder = resnet.__dict__[args.arch]
    if use_gpu:
        model = models.__dict__[args.model](encoder, args).cuda()
    else:
        model = models.__dict__[args.model](encoder, args)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.batch_size * dist.get_world_size() / 256 * args.base_learning_rate
            if args.dataset in ["MVTecAD"]
            else args.batch_size * dist.get_world_size() / 32 * args.base_learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "lars":
        params = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.SGD(
            params,
            lr=args.batch_size * dist.get_world_size() / 256 * args.base_learning_rate
            if args.dataset in ["MVTecAD"]
            else args.base_learning_rate,
            momentum=args.momentum,
        )
        optimizer = LARS(optimizer)
    else:
        raise NotImplementedError

    if use_gpu:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            broadcast_buffers=False,
        )

    return model, optimizer


def save_checkpoint(args, epoch, model):
    logger.info("==> Saving...")
    state = {"model": model.state_dict()}
    if use_amp:
        state["scaler"] = scaler.state_dict()
    file_name = os.path.join(
        args.output_dir,
        f"ckpt_epoch_{epoch}.pth",
    )
    torch.save(state, file_name)


def main(args):
    train_loader = get_loader(args)
    args.num_instances = len(train_loader.dataset)
    logger.info(f"length of training dataset: {args.num_instances}")
    model, optimizer = build_model(args)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    for epoch in range(args.start_epoch, args.epochs + 1):
        logger.info(f">>> epoch: {epoch}")

        if isinstance(train_loader.sampler, DistributedSampler):
            # calling the set_epoch() method at the beginning of each epoch before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.
            train_loader.sampler.set_epoch(epoch)

        train(epoch, train_loader, model, optimizer, scheduler, args)

        if dist.get_rank() == 0 and (
            epoch % args.save_freq == 0 or epoch == args.epochs
        ):
            save_checkpoint(
                args,
                epoch,
                model,
            )


def train(epoch, train_loader, model, optimizer, scheduler, args):
    """
    one epoch training
    """
    model.train()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()

    for idx, data in enumerate(train_loader):

        if use_gpu:
            data = [item.cuda(non_blocking=True) for item in data]

        if use_amp:
            with amp.autocast(enabled=use_amp):
                loss = model(*data)
        else:
            loss = model(*data)

        # backward
        optimizer.zero_grad()

        if use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        # update meters and print info
        loss_meter.update(loss.item(), data[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        train_len = len(train_loader)
        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Train: [{epoch}/{args.epochs}][{idx}/{train_len}]  "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                f"lr {lr:.3f}  "
                f"loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})"
            )


if __name__ == "__main__":
    opt = parse_option()

    if use_gpu:
        torch.cuda.set_device(opt.local_rank)
        cudnn.benchmark = True

    torch.distributed.init_process_group(
        backend="nccl" if use_gpu else "gloo", init_method="env://"
    )
    # modify options
    opt.output_dir = opt.output_dir + "_" + opt.timestamp
    opt.instance_branch_class_aware = (
        opt.instance_loss_weight > 0.0
        and opt.instance_loss_func in ["DistAug_MOCOv2", "MOCOv2"]
        and opt.dataset == "MVTecAD"
    )
    opt.dense_branch_class_aware = (
        opt.dense_loss_weight > 0.0
        and opt.dense_branch_class_aware
        and opt.dense_loss_func in ["PixPro_DenseCL"]
    )

    # setup logger
    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(
        output=opt.output_dir,
        distributed_rank=dist.get_rank(),
        name="contrast",
        timestamp=opt.timestamp,
    )

    logger.info(
        f">>> device names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
    )

    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, f"config_{opt.timestamp}.json")
        with open(path, "w") as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    # print args
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(opt)).items()))
    )

    opt.logger = logger

    main(opt)
