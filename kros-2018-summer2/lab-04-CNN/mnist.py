import numpy as np
import struct
import array
import os
import gzip

def parse_idx(fd):

    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)
    data_type = DATA_TYPES[data_type]
    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions, fd.read(4 * num_dimensions))

    # print(header)
    # print(zeros)
    # print(data_type)
    # print(num_dimensions)
    # print(dimension_sizes)

    data = array.array(data_type, fd.read())
    data.byteswap()

    return np.array(data).reshape(dimension_sizes)

def open_and_parse_idx3(fname):

    fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
    with fopen(fname, 'rb') as fd:
        images = parse_idx(fd)

    num, rows, cols = images.shape
    images = images.reshape((num,rows*cols))

    return images

def open_and_parse_idx1(fname, one_hot):

    fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
    with fopen(fname, 'rb') as fd:
        labels = parse_idx(fd)

    if one_hot:
        #print(len(labels))
        one_hot = np.zeros((len(labels),10))
        for i in range(len(labels)):
            one_hot[i][labels[i]] = 1
        labels = one_hot

    return labels

def load_train_datasets(target_dir, one_hot=False):

    images_fname = "train-images-idx3-ubyte.gz"
    labels_fname = "train-labels-idx1-ubyte.gz"

    images = open_and_parse_idx3(os.path.join(target_dir, images_fname))
    labels = open_and_parse_idx1(os.path.join(target_dir, labels_fname), one_hot)

    return images, labels

def load_test_datasets(target_dir, one_hot=False):

    images_fname = "t10k-images-idx3-ubyte.gz"
    labels_fname = "t10k-labels-idx1-ubyte.gz"

    images = open_and_parse_idx3(os.path.join(target_dir, images_fname))
    labels = open_and_parse_idx1(os.path.join(target_dir, labels_fname), one_hot)

    return images, labels


