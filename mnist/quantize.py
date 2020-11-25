import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _quantize_layer(weight, bits=8):
    """
    :param weight: A numpy array of any shape.
    :param bits: quantization bits for weight sharing.
    :return quantized weights and centriods.
    """
    shape = weight.shape
    min = np.min(weight)
    max = np.max(weight)
    # Get zero maskings
    zero_masking = (weight == 0)
    # Consider the 0 parameters, we have to reserve a centriod for them.
    num_clusters = 2 ** bits + 1
    kmeans_init = np.linspace(min, max, num_clusters)
    kmeans = KMeans(n_clusters=num_clusters, init=kmeans_init.reshape(-1, 1), n_init=1,
                    precompute_distances=True, algorithm='full')
    kmeans.fit(weight.reshape(-1, 1))
    new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(shape)
    # Recover the zero masking
    new_weight[zero_masking] = 0.
    # Now exclude the zero center
    centers = kmeans.cluster_centers_.copy()
    zero_center_idx = np.argmin(np.abs(centers))
    centers_ = []
    for i in range(len(centers)):
        if i != zero_center_idx:
            centers_.append(centers[i])
    centers_ = np.array(centers_)
    return new_weight, centers_

def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            weight, centers = _quantize_layer(weight, bits=bits)
            centers = centers.flatten()
            cluster_centers.append(centers)
            m.conv.weight.data = torch.from_numpy(weight).to(device)
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            weight, centers = _quantize_layer(weight, bits=bits)
            centers = centers.flatten()
            cluster_centers.append(centers)
            m.linear.weight.data = torch.from_numpy(weight).to(device)
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
    return np.array(cluster_centers)

