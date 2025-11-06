import torch

from util import accuracy, get_kappa, class_wise_accuracies


@torch.no_grad()
def val_step(net, x, val_mask, adj, y, loss_func):
    net.eval()
    output = net(x, adj)
    loss_val = loss_func(output[val_mask], y[val_mask])
    acc_val = accuracy(output[val_mask], y[val_mask])
    kappa = get_kappa(y[val_mask], output[val_mask])
    cc = class_wise_accuracies(output[val_mask], y[val_mask], 16)
    return loss_val, acc_val, kappa, cc, output[val_mask]
