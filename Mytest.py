from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import torch
from resUnet import *
from data import *

net = ResUNet()

weights='params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')

_input = input('please input test mat file path:')
test_matpath = loadmat(_input)
testmat = torch.Tensor(test_matpath['pics1']).unsqueeze(0)

img_data = torch.unsqueeze(testmat, dim=0)
print(img_data)
net.eval()
out = net(img_data)
print(out)

# 输出结果
output_image = out.squeeze(-4).squeeze(-3).detach().numpy()
input_image = img_data.squeeze(-4).squeeze(-3).numpy()

# plt.imshow(input_image, cmap='gray')  # 显示输入图像
# plt.imshow(output_image, cmap='gray')  # 显示输出图像

fig = plt.figure(figsize=(10, 5))
rows = 1
columns = 2

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

plt.show()




