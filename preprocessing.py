from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import config as cfg
import pandas as pd
import numpy as np
import random
import torch
import h5py
import os


def read_mat(mat_filename:str, key:str):
    # read_mat reads .mat file specified and returns data specified by key
    # returns an nx1 numpy array, where n is the length of the original list

    try:
        file = h5py.File(mat_filename, 'r')
        data = np.transpose(np.array(file[key]))
        return data
    except:
        print("Error reading input file")


def random_elements(vec1, vec2, num_seq : int, seq_len : int):
    # random_elements takes in two vectors of the same size, the number of
    # sequences in each, and the number of samples/length in each sequence.
    # Returns two vectors shuffled the same way and the order as lists

    if len(vec1) != len(vec2):
        print('Error: lengths of input vectors do not match.')
        print('Length of first argument: %f' % len(vec1))
        print('Length of second argument: %f' % len(vec2))
        return
    order = [None] * num_seq
    for i in range(num_seq):
        order[i] = i
    random.shuffle(order)

    vec1_rand = [None] * len(vec1)
    vec2_rand = [None] * len(vec1)
    for i in range(num_seq):
        start_old = i * seq_len
        start_new = order[i] * seq_len
        vec1_rand[start_new : start_new + seq_len] = vec1[start_old : start_old + seq_len]
        vec2_rand[start_new : start_new + seq_len] = vec2[start_old : start_old + seq_len]

    return vec1_rand, vec2_rand, order


def divide_dataset(x, y, num_seq, seq_len, pct_training : int):
    # divide_dataset takes in two vectors, the number of sequences in each vector,
    # the number of samples per sequence, and a percentage of how to divide the data
    # and returns x_train, y_train, x_test, y_test as numpy arrays and number of training sequences

    split_group_num = int(num_seq * pct_training / 100)
    split_ind = split_group_num * seq_len

    x_rand, y_rand, _ = random_elements(x, y, num_seq, seq_len)
    x_train = np.array(x_rand[:split_ind])
    x_test  = np.array(x_rand[split_ind:])
    y_train = np.array(y_rand[:split_ind])
    y_test  = np.array(y_rand[split_ind:])
    
    return x_train.reshape(-1, 1), y_train.reshape(-1, 1), x_test.reshape(-1, 1), y_test.reshape(-1, 1)
    

def create_datasets(x, y, num_seq, seq_len, pct_training : int):
    # create_datasets takes in two vectors, the number of sequences in each vector,
    # the number of samples per sequence, and a percentage of how to divide the data
    # and returns training and testing datasets in the form of PyTorch tensors

    x_train, y_train, x_test, y_test = divide_dataset(x, y, num_seq, seq_len, pct_training)
    x_train = torch.tensor(x_train, dtype = torch.float)
    y_train = torch.tensor(y_train, dtype = torch.long)
    x_test  = torch.tensor(x_test, dtype = torch.float)
    y_test  = torch.tensor(y_test, dtype = torch.long)
    return x_train, y_train, x_test, y_test


def create_dataloader(x_train, y_train, x_test, y_test, mini_batch_size):
    # create_dataloader takes in PyTorch tensors and mini batch size and returns
    # PyTorch dataloaders containing training and testing data.

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset  = TensorDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size = mini_batch_size, shuffle = False)
    test_dataloader  = DataLoader(test_dataset, batch_size = mini_batch_size, shuffle = False)
    return train_dataloader, test_dataloader


def reshape_data(x_train, y_train, x_test, y_test, seq_len):
    x_train = x_train.view(-1, seq_len, 1)
    y_train = y_train.view(-1, seq_len, 1)
    x_test  = x_test.view(-1, seq_len, 1)
    y_test  = y_test.view(-1, seq_len, 1)
    return x_train, y_train, x_test, y_test


def prepare_data(x_file_info : tuple, y_file_info : tuple, seq_len, pct_train : int, mini_batch_size, one_based_ind = False):
    # prepare_data transforms the .mat files into PyTorch tensors and saves the data.
    # Returns a flag for success of operation.
    
    try:
        waves = read_mat(x_file_info[0], x_file_info[1])
        masks = read_mat(y_file_info[0], y_file_info[1])
        if one_based_ind: masks = masks - 1  # convert 1-based to 0-based indexing

        num_seq = int(len(waves) / seq_len)
        x_train, y_train, x_test, y_test = create_datasets(waves, masks, num_seq, seq_len, pct_train)
        x_train, y_train, x_test, y_test = reshape_data(x_train, y_train, x_test, y_test, seq_len)
        print("Data successfully transformed to PyTorch tensors.")

        tensors = [x_train, y_train, x_test, y_test]
        torch.save(tensors, 'inhale_exhale_tensor_dat.pt')
        print("Data successfully saved to 'inhale_exhale_tensor_dat.pt'")

    except Exception as e:
        print(e)