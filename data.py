import os
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision import transforms
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.nameL = os.listdir(os.path.join(path, 'original'))

    def __len__(self):
        return len(self.nameL)

    def __getitem__(self, index):
        name = self.nameL[index]  # xx.png
        mat_l_path = os.path.join(self.path, 'original', name)  #读取低晶源文件路径
        mat_m_path = os.path.join(self.path, 'target', name)  #读取多晶源文件路径
        mat_l_data = loadmat(mat_l_path)
        mat_m_data = loadmat(mat_m_path)

        return torch.Tensor(mat_l_data['pics1']).unsqueeze(0), torch.Tensor(mat_m_data['pics2']).unsqueeze(0)

if __name__ == '__main__':
    # from torch.nn.functional import one_hot
        data = MyDataset('data/Train')
    # print(data[0][0].shape)
    # print(data[0][1].shape)
    # out=one_hot(data[0][0].long())
    # print(data[0][1])
    # arrayImg = np.array(data[0][0]) # transfer tensor to array
    # plt.imshow(arrayImg, cmap='gray')  # show image
    # plt.show()

    # data_loader = DataLoader(MyDataset('data'), batch_size=2, shuffle=True)
    # for image, segment_image in data_loader:
    # # data = MyDataset('data')
    # # print(data[0][0])
    #     print(image.shape, segment_image.shape)




