from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from resUnet import *
from data import *

net = ResUNet()
net.eval()

data_path = r'data/Test'
data_loader = DataLoader(MyDataset(data_path), batch_size=1, shuffle=False)

weights='params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')

for i, (original_image, target_image) in enumerate(data_loader):
    # original_image, target_image = image.to(device), segment_image.to(device)
    out = net(original_image)
    print(out)
    # 输出结果
    output_image = out.squeeze(-4).squeeze(-3).detach().numpy()
    input_image = original_image.squeeze(-4).squeeze(-3).numpy()
    target_image = target_image.squeeze(-4).squeeze(-3).numpy()
    # 画图
    fig = plt.figure(figsize=(15, 5))
    rows = 1
    columns = 3

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
    # showing image
    plt.imshow(input_image, cmap='gray')
    plt.axis('off')
    plt.title("input")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    # showing image
    plt.imshow(output_image, cmap='gray')
    plt.axis('off')
    plt.title("output")

    fig.add_subplot(rows, columns, 3)
    # showing image
    plt.imshow(target_image, cmap='gray')
    plt.axis('off')
    plt.title("target")

plt.show()




