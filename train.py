import tqdm
import os
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from resUnet import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = r'data/Train'
save_path = 'train_image'
if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path), batch_size=8, shuffle=True)
    net = ResUNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path, map_location=device))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))
    # loss_fun = nn.CrossEntropyLoss()
    loss_fun = nn.L1Loss()
    epoch = 1
    while epoch < 500:
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(device), segment_image.to(device)
            # print(image.shape, segment_image.shape)
            out_image = net(image)
            # print(out_image.shape)
            train_loss = loss_fun(out_image, segment_image)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 20 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            # _image = image[0]
            # _segment_image = torch.argmax(segment_image[0], 0).unsqueeze(0) * 255
            # # print(_image.shape, _segment_image.shape)
            # _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255
            #
            # img = torch.stack([_segment_image, _out_image], dim=0)
            # # save_image(img, f'{save_path}/{i}.png')
        if epoch % 10 == 0:
            torch.save(net.state_dict(), weight_path)
            print('save successfully!')
        epoch += 1
