# import packages
import torch
# additional certain short functions
from torch.utils.data import Dataset
from random import randint
from copy import deepcopy
from math import floor, ceil
from torch import stack
from torch import cat

# object picking
def pick(dataset: Dataset, name: list, number: int) -> list:
    i = 0   # intitial start searching point
    counter = [0] * len(name)
    score = []
    while True:
        if dataset.classes[dataset.targets[i]] in name and \
            counter[name.index(dataset.classes[dataset.targets[i]])] < number:
                if score == []:
                    score = dataset.data[i].reshape(1, 28, 28)
                else:
                    score = cat((score, dataset.data[i].reshape(1, 28, 28)), dim=0)
                counter[name.index(dataset.classes[dataset.targets[i]])] += 1
        i += 1
        correct = 0
        for j in range(len(name)):
            if counter[j] == number:
                correct += 1
        if correct == len(name):
            return score
        
# sequential album extension
def album(dataset: torch.Tensor, is_rand_stride: bool, is_rand_pos: bool) -> torch.Tensor:
    expansion = []      # list after expanded
    temp1 = []          # list after stride
    temp2 = []          # list after position
    stride = 5          # default stride
    pos = [14, 27]      # default starting position [col, row]
    num_list = [0] * len(dataset)       # record number for each original album
    score_list = [0] * len(dataset)     # record number for each final album
    stride_list = [0] * len(dataset)    # record stride for each final album
    
    # expand and clean
    for i in range(len(dataset)):
        if i == 0:
            expansion = dataset[i].repeat(3, 4).reshape(1, 28*3, 28*4)
        else:
            buffer = dataset[i].repeat(3, 4).reshape(1, 28*3, 28*4)
            expansion = cat((expansion, buffer), dim=0)
        expansion[i][:, 28:] = 0
        expansion[i][28:, :] = 0
    expansion = torch.tensor(expansion, dtype=torch.float32)
    # print("expansion shape =", expansion.shape)
    
    # random stride
    for i in range(len(num_list)):
        # define stride value
        if is_rand_stride == True: stride = randint(3, 7)
        # album generation
        first_frame = deepcopy(expansion[i])
        next_frame = deepcopy(expansion[i])
        y = deepcopy(expansion[i])
        for j in range(floor((112-28)/stride)):
            # moving part (1-dim only)
            for z in range(28):
                next_frame[:, 28+stride*(j+1)-(z+1)] = next_frame[:, 28+stride*j-(z+1)]
            # clean other area
            next_frame[:, :stride*(j+1)+1] = 0
            # sequencing part
            if j == 0: y = stack((first_frame, next_frame)) 
            else: y = cat((y, next_frame.reshape(1, 28*3, 28*4)), dim=0)
        # record number
        num_list[i] = len(y)
        # sequencing part
        if i == 0: temp1 = y
        else: temp1 = cat((temp1, y), dim=0)
        # renew stride_list
        stride_list[i] = stride

    # random position
    if is_rand_pos == True: 
        # intialize temp2
        temp2 = torch.zeros(len(temp1), 28*3, 28*4)
        # start random position moving
        for i in range(len(num_list)):
            # define position value
            pos = [randint(14, 27), randint(27, 28*3-1)]    # [col, row]
            stride = floor((112-28)/(num_list[i]-1))        # re-inference stride value
            # parallel movement
            for j in range(num_list[i]):
                for k in range(28):
                    for l in range(28):
                        temp2[sum(num_list[:i])+j][pos[1]-k][pos[0]+stride*j-l] = \
                            temp1[sum(num_list[:i])+j][27-k][27+stride*j-l]
                # clean other area
                temp2[sum(num_list[:i])+j][:pos[1]-28+1, :] = 0
                temp2[sum(num_list[:i])+j][:, pos[0]+stride*j+1:] = 0
            # fill temp into score
            if i == 0: score = temp2[:num_list[i]]
            else: score = cat((score, temp2[sum(num_list[:i]):sum(num_list[:i+1])]), dim=0)
            # fetch up last frames
            next_frame = deepcopy(temp2[sum(num_list[:(i+1)])-1])
            for j in range(ceil((28-pos[0])/stride)+1):
                for k in range(28):
                    if pos[0]+stride*(num_list[i]+j)-k < 112:
                        next_frame[:, pos[0]+stride*(num_list[i]+j)-k] = \
                            next_frame[:, pos[0]+stride*(num_list[i]+j-1)-k]
                # clean other area
                next_frame[:, :pos[0]+stride*(num_list[i]+j)-28+1] = 0
                score = cat((score, next_frame.reshape(1, 28*3, 28*4)), dim=0)
            # renew score_list
            score_list[i] = num_list[i]+j+1
    
    # # print list shape
    # print("temp shape =", temp.shape)
    # print("numbers of each original album =", num_list)
    # if is_rand_pos == True:
    #     print("score shape =", score.shape)
    #     print("numbers of each final album =", score_list)

    # return setting (add stride_list afterward)
    if is_rand_pos == True: return score, score_list, stride_list
    else: return temp1, num_list, stride_list
    
# frame differences generation function
def dif_frame(dataset: torch.Tensor, num_list: list) -> torch.Tensor:
    score = []                          # return tensor
    node = [0] * len(num_list)          # node for swithing album
    front_frame = deepcopy(dataset[0])
    back_frame = deepcopy(dataset[1])
    # sum the num for node list
    for i in range(len(num_list)):
        node[i] = sum(num_list[:i])
    for i in range(len(dataset)):
        # reset front_frame and back_frame when switching album
        if i in node:
            front_frame = dataset[i]
            back_frame = dataset[i+1]
            continue
        if i == 1: score = abs(back_frame - front_frame).reshape(1, 28*3, 28*4)
        else: score = cat((score, abs(back_frame - front_frame).reshape(1, 28*3, 28*4)), dim=0)
        # renew front_frame and back_frame within one album 
        if i < len(dataset)-1:
            front_frame = back_frame
            back_frame = dataset[i+1]
    return score
            
# vector generation function (x-axis info only)
def vector(num_list: list, stride_list: list) -> torch.Tensor:
    stack = []      # vector stack for each album
    score = []
    for i in range(len(num_list)):
        stack = torch.tensor([stride_list[i]]).repeat(num_list[i]-1, 1)
        if i == 0: score = stack
        else: score = cat((score, stack), dim=0)
    return score
    