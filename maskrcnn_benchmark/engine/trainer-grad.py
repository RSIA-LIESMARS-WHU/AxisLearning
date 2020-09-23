# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

# 分布式
def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = [] 
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def get_normal(model, lossname=""):
    total_norm = 0
    
    for k,w in model.named_parameters():
        #  一个参数的所有数据的范数值
        # print(k)
        if w.grad is not None:
            param_norm = w.grad.data.norm(2)                              
            #tensor.norm(p)   计算p范数
            total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2) 
    # print(lossname,total_norm)
    return total_norm.cpu().numpy()


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    with open("./grad.txt", "w") as grad_file:
        for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            scheduler.step()

            images = images.to(device)
            targets = [target.to(device) for target in targets]

            loss_dict = model(images, targets)
            cls_losses = loss_dict["loss_cls"]
            loss_reg = loss_dict["loss_reg"]
            loss_centerness = loss_dict["loss_centerness"]
            # losses = sum(loss for loss in loss_dict.values())
            
            # # reduce losses over all GPUs for logging purposes
            # loss_dict_reduced = reduce_loss_dict(loss_dict)

            # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            # meters.update(loss=losses_reduced, **loss_dict_reduced)
            
            optimizer.zero_grad()
            cls_losses.backward(retain_graph=True)
            cls_normal = get_normal(model, "cls_losses")

            optimizer.zero_grad()
            loss_reg.backward(retain_graph=True)
            reg_normal = get_normal(model, "reg_losses")
            
            optimizer.zero_grad()
            loss_centerness.backward()
            center_normal = get_normal(model, "loss_centerness")
            print("reg:{} cls:{} center:{}\n".format(reg_normal, cls_normal, center_normal))
            grad_file.write("{} {} {}\n".format(reg_normal, cls_normal, center_normal))
            # optimizer.step()

            # batch_time = time.time() - end
            # end = time.time()
            # meters.update(time=batch_time, data=data_time)

            # eta_seconds = meters.time.global_avg * (max_iter - iteration)
            # eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            # if iteration % 20 == 0 or iteration == max_iter:
            #     logger.info(
            #         meters.delimiter.join(
            #             [
            #                 "eta: {eta}",
            #                 "iter: {iter}",
            #                 "{meters}",
            #                 "lr: {lr:.6f}",
            #                 "max mem: {memory:.0f}",
            #             ]
            #         ).format(
            #             eta=eta_string,
            #             iter=iteration,
            #             meters=str(meters),
            #             lr=optimizer.param_groups[0]["lr"],
            #             memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
            #         )
            #     )
            # if iteration % checkpoint_period == 0:
            #     checkpointer.save("model_{:07d}".format(iteration), **arguments)
            # if iteration == max_iter:
            #     checkpointer.save("model_final", **arguments)

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iter)
            )
        )
