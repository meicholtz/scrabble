import numpy as np
from utils import *
import ipdb

my_data = np.load('/Users/Alex/Desktop/Summer-2019/scrabble/scrabble_dataset.npz', allow_pickle=True)
data = np.load('/Users/Alex/Desktop/Summer-2019/scrabble/yad2k-em3d/data.npz', allow_pickle=True)
print('My image shape: {} \nORNL image Shape: {}\n\n'.format(my_data['images'].shape, data['images'].shape))
print('My shape of one image: {} \nORNL shape of one image: {}\n\n'.format(my_data['images'][0].shape, data['images'][0].shape))
print('My boxes shape: {} \nORNL boxes Shape: {}\n\n'.format(my_data['boxes'].shape, data['boxes'].shape))
print('My shape of one box: {} \nORNL shape of one box: {}\n\n'.format(my_data['boxes'][0].shape, data['boxes'][0].shape))
ipdb.set_trace()

