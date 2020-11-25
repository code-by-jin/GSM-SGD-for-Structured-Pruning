import time

import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim


def final_unstruct_pruning (net, nonzero_ratio, net_name):

    assert isinstance(net, nn.Module)
    to_concat = []
    for n, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            weight = m.weight.data
            to_concat.append(torch.abs(weight.view(-1)))

    all_abs_weights = torch.cat(to_concat)
    num_zero = int(all_abs_weights.size(0) * nonzero_ratio)


    top_values, _ = torch.topk(torch.abs(all_abs_weights), num_zero)
    abs_thresh = top_values[-1]

    for n, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            weight = m.weight.data
            mask = torch.abs(weight) >= abs_thresh
            weight = weight*mask
            m.weight.data = torch.cuda.FloatTensor(weight)
    torch.save(net.state_dict(), 'saved_models/'+net_name)
        
# def final_struct_pruning (net, nonzero_ratio, net_name):

#     assert isinstance(net, nn.Module)
#     to_concat = []
#     for n, m in net.named_modules():
#         if isinstance(m, nn.Conv2d):
#             weight = m.weight.data
#             metric = torch.mean(torch.abs(weight), dim =(1, 2, 3))
#             to_concat.append(metric.view(-1))
#         if isinstance(m, nn.Linear):
#             weight = m.weight.data
#             metric = torch.mean(torch.abs(weight), dim =1)
#             to_concat.append(metric.view(-1))
            
#     all_abs_weights = torch.cat(to_concat)
    
#     num_params = all_abs_weights.size(0)
#     num_zero = int(num_params*nonzero_ratio)
#     top_values, _ = torch.topk(torch.abs(all_abs_weights), num_zero)
#     abs_thresh = top_values[-1]

#     for n, m in net.named_modules():
#         if isinstance(m, nn.Conv2d):
#             weight = m.weight.data
#             mask = torch.mean(torch.abs(weight), dim =(1, 2, 3)) >= abs_thresh
#             mask = torch.reshape(mask, (mask.size(0), 1, 1, 1))
#             m.weight.data = weight*mask
#         if isinstance(m, nn.Linear):
#             weight = m.weight.data
#             mask = torch.mean(torch.abs(weight), dim =(1)) >= abs_thresh
#             mask = torch.reshape(mask, (mask.size(0), 1))
#             m.weight.data = weight*mask
#     torch.save(net.state_dict(), 'saved_models/'+net_name)
            
def final_struct_pruning (net, nonzero_ratio, net_name):

    assert isinstance(net, nn.Module)
    to_concat = []
    for n, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            weight = m.weight.data
            metric = torch.mean(torch.abs(weight), dim =(1, 2, 3))
            to_concat.append(metric.view(-1))
        if isinstance(m, nn.Linear):
            weight = m.weight.data
            metric = torch.mean(torch.abs(weight), dim =0)
            to_concat.append(metric.view(-1))
    all_abs_weights = torch.cat(to_concat)

    num_params = all_abs_weights.size(0)
    num_zero = int(num_params*nonzero_ratio)

    top_values, _ = torch.topk(torch.abs(all_abs_weights), num_zero)
    abs_thresh = top_values[-1]
    for n, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            weight = m.weight.data
            mask = torch.mean(torch.abs(weight), dim =(1, 2, 3)) >= abs_thresh
            mask = torch.reshape(mask, (mask.size(0), 1, 1, 1))
            m.weight.data = weight*mask
        if isinstance(m, nn.Linear):
            weight = m.weight.data
            mask = torch.mean(torch.abs(weight), dim =(0)) >= abs_thresh
            mask = torch.reshape(mask, (1, mask.size(0)))
            m.weight.data = weight*mask

    torch.save(net.state_dict(), 'saved_models/'+net_name)