import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class HuffmanQueue:
    """
    Queue used for Huffman Coding.
    """
    def __init__(self, keys, values):
        self.keys = np.array(keys)
        self.values = np.array(values)
        self.queue_length = len(self.keys)

    def dequeue_min_values(self):
        id = np.argmin(self.values)
        key, value = self.keys[id], self.values[id]
        self.keys = np.concatenate([self.keys[:id], self.keys[id+1:]])
        self.values = np.concatenate([self.values[:id], self.values[id+1:]])
        self.queue_length -= 1
        return key, value

    def enqueue(self, key, value):
        self.keys = np.concatenate([self.keys, [key]])
        self.values = np.concatenate([self.values, [value]])
        self.queue_length += 1

    def finalizd(self):
        return (self.queue_length == 1)


def _is_not_a_virtual_node(name):
    return name[0] != 'n'


def get_encode(idx, ordered_huffman_sequence, attr, encodings):
    """
    Recursive 'preorder' huffman coding process.
    """
    lch = ordered_huffman_sequence[idx-2]
    rch = ordered_huffman_sequence[idx-1]
    if _is_not_a_virtual_node(lch):
        encodings[idx-2] = attr + '0'
    else:
        for i in range(idx-1):
            if ordered_huffman_sequence[i] == lch:
                get_encode(i, ordered_huffman_sequence, attr + "0", encodings)
                break
        pass
    if _is_not_a_virtual_node(rch):
        encodings[idx-1] = attr + '1'
    else:
        for i in range(idx-1):
            if ordered_huffman_sequence[i] == rch:
                get_encode(i, ordered_huffman_sequence, attr + "1", encodings)
                break

def _encode_huffman_sequence(ordered_huffman_sequence):
    """
    Fetch the huffman coding.
    """
    root_idx = len(ordered_huffman_sequence)-1
    encodings = np.empty_like(ordered_huffman_sequence)
    get_encode(root_idx, ordered_huffman_sequence, "", encodings)

    valid_keys = []
    valid_encodings = []

    for i in range(len(encodings)):
        if _is_not_a_virtual_node(ordered_huffman_sequence[i]):
            valid_encodings.append(encodings[i])
            valid_keys.append(ordered_huffman_sequence[i])
    res = dict(zip(valid_keys, valid_encodings))
    return res

def _huffman_coding_per_layer(weight, centers):
    """
    Huffman coding for each layer
    :param weight: weight parameter of the current layer.
    :param centers: KMeans centroids in the quantization codebook of the current weight layer.
    :return: encoding map and frequency map for the current weight layer.
    """
    num_centers = np.shape(centers)[0]
    freq = np.zeros_like(centers)
    weight_1d = weight.flatten()
    for i in range(num_centers):
        freq[i] = int((weight_1d == centers[i]).sum())

    ordered_huffman_sequence = np.array([])
    queue = HuffmanQueue(centers, freq)
    auxilary_index = 0
    while not queue.finalizd():
        key1, value1 = queue.dequeue_min_values()
        key2, value2 = queue.dequeue_min_values()
        value = value1 + value2
        node_name = "node_%d" %auxilary_index
        auxilary_index += 1
        queue.enqueue(node_name, value)
        ordered_huffman_sequence = np.concatenate([ordered_huffman_sequence, [key1, key2, node_name]])
    encodings = _encode_huffman_sequence(ordered_huffman_sequence)
    frequency = dict(zip(np.array(centers, np.str), freq))
    return encodings, frequency


def compute_average_bits(encodings, frequency):
    """
    Compute the average storage bits of the current layer after Huffman Coding.
    :param encodings: encoding map of the current layer w.r.t. weight (centriod) values.
    :param frequency: frequency map of the current layer w.r.t. weight (centriod) values.
    :return (float) a floating value represents the average bits.
    """
    total = 0
    total_bits = 0
    for key in frequency.keys():
        total += frequency[key]
        total_bits += frequency[key] * len(encodings[key])
    return total_bits / total

def huffman_coding(net, centers):
    """
    Apply huffman coding on a 'quantized' model to save further computation cost.
    :param net: a 'nn.Module' network object.
    :param centers: KMeans centroids in the quantization codebook for Huffman coding.
    :return: frequency map and encoding map of the whole 'net' object.
    """
    assert isinstance(net, nn.Module)
    layer_ind = 0
    freq_map = []
    encodings_map = []
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)

    return freq_map, encodings_map