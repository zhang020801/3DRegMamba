import os
import glob
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data



import pickle
def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class Dataset(Data.Dataset):
    def __init__(self, files):
      
        self.files = files

    def __len__(self):
        
        return len(self.files)

    def __getitem__(self, index):
        
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]
        index = self.files[index][59:61]
        
        return img_arr, index


class IXIDataset(Data.Dataset):
    def __init__(self, files):
       
        self.files = files

    def __len__(self):
        
        return len(self.files)

    def __getitem__(self, index):
        
        file_path = self.files[index]
        image, label = pkload(file_path)


        img_arr = image[np.newaxis, ...]
        index = self.files[index][59:61]
        
        return img_arr, index
