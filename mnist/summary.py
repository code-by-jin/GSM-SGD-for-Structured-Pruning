import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def summary(net):
    assert isinstance(net, nn.Module)
    print("Layer id\tType\t\tParameter\tNon-zero parameter\tSparsity(\%)")
    layer_id = 0
    num_total_filters = 0
    num_total_params = 0
    num_total_nonzero_params = 0
    num_total_nonzero_filters = 0
    for n, m in net.named_modules():
        if isinstance(m, nn.Linear):
            weight = m.weight.data.cpu().numpy()
            weight_sum = np.sum(np.abs(weight), axis = (0)).flatten()
            num_filters = weight_sum.shape[0]
            num_nonzero_filters = (weight_sum != 0).sum()
            sparisty_filters = 1 - num_nonzero_filters / num_filters
            
            weight = weight.flatten()
            num_parameters = weight.shape[0]
            num_nonzero_parameters = (weight != 0).sum()
            sparisty = 1 - num_nonzero_parameters / num_parameters
            layer_id += 1
            print("%d\t\tLinear\t\t%d\t\t%d\t\t\t%f" %(layer_id, num_parameters, num_nonzero_parameters, sparisty))
            print("%d\t\ttLinear_Filter\t%d\t\t%d\t\t\t%f" % (layer_id, num_filters, num_nonzero_filters, sparisty_filters))
            num_total_params += num_parameters
            num_total_nonzero_params += num_nonzero_parameters
            num_total_filters += num_filters
            num_total_nonzero_filters += num_nonzero_filters            
        elif isinstance(m, nn.Conv2d):
            weight = m.weight.data.cpu().numpy()
            weight_sum = np.sum(np.abs(weight), axis = (1, 2, 3)).flatten()
            num_filters = weight_sum.shape[0]
            num_nonzero_filters = (weight_sum != 0).sum()
            sparisty_filters = 1 - num_nonzero_filters / num_filters
            
            weight = weight.flatten()
            num_parameters = weight.shape[0]
            num_nonzero_parameters = (weight != 0).sum()
            sparisty = 1 - num_nonzero_parameters / num_parameters
            layer_id += 1
            print("%d\t\tConvolutional_Param\t%d\t\t%d\t\t\t%f" % (layer_id, num_parameters, num_nonzero_parameters, sparisty))
            print("%d\t\tConvolutional_Filter\t%d\t\t%d\t\t\t%f" % (layer_id, num_filters, num_nonzero_filters, sparisty_filters))
            num_total_params += num_parameters
            num_total_nonzero_params += num_nonzero_parameters
            num_total_filters += num_filters
            num_total_nonzero_filters += num_nonzero_filters
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            layer_id += 1
            print("%d\t\tBatchNorm\tN/A\t\tN/A\t\t\tN/A" % (layer_id))
        elif isinstance(m, nn.ReLU):
            layer_id += 1
            print("%d\t\tReLU\t\tN/A\t\tN/A\t\t\tN/A" % (layer_id))

    print("Total nonzero parameters: %d" %num_total_nonzero_params)
    print("Total parameters: %d" %num_total_params)
    total_sparisty = 1. - num_total_nonzero_params / num_total_params
    print("Total sparsity: %f" %total_sparisty)

