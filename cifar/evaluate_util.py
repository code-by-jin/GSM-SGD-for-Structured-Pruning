# adapted from https://github.com/yanghr/DeepHoyer/issues/2

from torchprofile import profile_macs
import torch
import numpy as np
from final_pruning import final_unstruct_pruning, final_struct_pruning
import torch.nn as nn
import matplotlib.pyplot as plt

def compute_conv_flops(model, cuda=False, prune=False):
    """
    compute the FLOPs for sparse model
    @param cuda: if use gpu to do the forward pass
    @param prune: if compute the FLOPs of the pruned model. If prune == False, the method compute the baseline FLOPs.
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    list_conv = []
    list_linear = []

    def conv_hook(self, input, output):

        batch_size, input_channels, input_height, input_width = input[0].size()

        # prune the conv layers
        tensor = self.weight.data.cpu().numpy()
        tensor = np.abs(tensor)
        if len(tensor.shape) == 4:
            dim0 = np.sum(np.sum(tensor, axis=0), axis=(1, 2))
            dim1 = np.sum(np.sum(tensor, axis=1), axis=(1, 2))
        if len(tensor.shape) == 2:
            dim0 = np.sum(tensor, axis=0)
            dim1 = np.sum(tensor, axis=1)
        nz_count0 = np.count_nonzero(dim0)  # input channel
        nz_count1 = np.count_nonzero(dim1)  # output channel

        output_channels, output_height, output_width = output[0].size()

        if prune:
            input_channels = nz_count0
            output_channels = nz_count1

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (input_channels / self.groups)

        flops = kernel_ops * output_channels * output_height * output_width

        list_conv.append(flops)

    def linear_hook(self, input, output):
        batch_size, input_channels = input[0].size()
        tensor = self.weight.data.cpu().numpy()
        tensor = np.abs(tensor)
        dim0 = np.sum(tensor, axis=0)
        dim1 = np.sum(tensor, axis=1)
        nz_count0 = np.count_nonzero(dim0)  # input channel
        nz_count1 = np.count_nonzero(dim1)  # output channel
        output_channels = output[0].size()
        
#         weight_ops = self.weight.nelement()
        if prune:
            input_channels = nz_count0
            output_channels = nz_count1
            
        flops = input_channels*output_channels
        list_linear.append(flops)

    def add_hooks(net, hook_handles: list):
        """
        apply FLOPs handles to conv layers recursively
        """
        children = list(net.children())
        if not children:
            if isinstance(net, torch.nn.Conv2d):
                hook_handles.append(net.register_forward_hook(conv_hook))
            if isinstance(net, torch.nn.Linear):
                hook_handles.append(net.register_forward_hook(linear_hook))
            return
        for c in children:
            add_hooks(c, hook_handles)

    handles = []
    add_hooks(model, handles)
    demo_input = torch.rand(8, 3, 32, 32)
    if cuda:
        demo_input = demo_input.cuda()
        model = model.cuda()
    model(demo_input)

    total_flops = sum(list_conv) + sum(list_linear)

    # clear handles
    for h in handles:
        h.remove()
    return total_flops
