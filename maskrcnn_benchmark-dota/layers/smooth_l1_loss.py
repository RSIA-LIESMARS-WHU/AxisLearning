# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


# # TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, weight=None, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    # len(pos_locations 5)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if weight is not None and weight.sum() > 0:
        # print("input ", input)
        # print("weight", weight)
        loss = (loss * weight).sum() / weight.sum()
        # # loss = (loss * weight)
        # print("loss", loss)
        return loss
    elif size_average:
        return loss.mean()
    # return loss.sum()


# TODO maybe push this to nn?
# def smooth_l1_loss(input, target, beta=1. / 9, weight=None, size_average=True):
#     """
#     very similar to the smooth_l1_loss from pytorch, but with
#     the extra beta parameter
#     """
#     n = torch.abs(input - target)
#     cond = n < beta
#     # len(pos_locations 5)
#     loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

#     if weight is not None and weight.sum() > 0:
#         # print("input ", input)
#         # print("weight", weight)
#         loss1, loss2 = loss.chunk(2, 1)
#         loss = torch.where(loss1 < loss2, loss1, loss2)
#         loss = (loss * weight).sum() / weight.sum()
#         # # loss = (loss * weight)
#         # print("loss", loss)
#         return loss
#     elif size_average:
#         return loss.mean()
#     return loss.sum()
