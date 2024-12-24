import torch
from torch import nn
from torch.nn import functional as F
from data import *
class Input_Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Input_Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(output_dim),
        )

    def forward(self,x):

        return self.layer(x)+self.conv_skip(x)


class Res_Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Res_Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.layer(x)+self.conv_skip(x)


class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channel, out_channel, kernel_size=3, stride=2, padding=1,output_padding=1
        )

    def forward(self, x):
        return self.upsample(x)


class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()
        self.level1 = Input_Conv_Block(1,32)
        self.maxpool1=nn.MaxPool2d(2,2)
        self.level2 = Res_Conv_Block(32,64)
        self.maxpool2=nn.MaxPool2d(2,2)
        self.level3 = Res_Conv_Block(64,128)
        self.maxpool3=nn.MaxPool2d(2,2)
        self.level4 = Res_Conv_Block(128,256)
        self.maxpool4=nn.MaxPool2d(2,2)

        #bridge
        self.bridge=Res_Conv_Block(256,512) #512

        #upsample
        self.upsample1=UpSample(512,256)
        self.level5 = Res_Conv_Block(512,256)

        self.upsample2=UpSample(256,128) #128
        self.level6 = Res_Conv_Block(256,128)

        self.upsample3=UpSample(128,64) #64
        self.level7 = Res_Conv_Block(128,64)

        self.upsample4=UpSample(64,32) #32
        self.level8 = Res_Conv_Block(64,32)
        self.output_layer = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)



    def forward(self,x):
        #level1
        x1=self.level1(x)
        x2=self.maxpool1(x1)

        #level2
        x3=self.level2(x2)
        x4=self.maxpool2(x3)
        #level3
        x5=self.level3(x4)
        x6=self.maxpool3(x5)
        #level4
        x7=self.level4(x6)
        x8=self.maxpool4(x7)

        #bridge
        bridge=self.bridge(x8)

        #level5
        up1=self.upsample1(bridge)
        up2=torch.cat([up1,x7],dim=1)
        up3=self.level5(up2)

        #level6
        up4=self.upsample2(up3) #128
        up5=torch.cat([up4,x5],dim=1)
        up6=self.level6(up5)

        #level7
        up7=self.upsample3(up6) #64
        up8=torch.cat([up7,x3],dim=1)
        up9=self.level7(up8)

        #level8
        up10=self.upsample4(up9) #32
        up11=torch.cat([up10,x1],dim=1)
        up12=self.level8(up11)

        #output
        output1 = self.output_layer(up12)

        return output1 + x

if __name__ == '__main__':
    # x=torch.randn(2,3,256,256)
    data = MyDataset('data')
    img = data[0][0]
    img = img.unsqueeze(0)
    net = ResUNet()
    imgput = net(img)
    imgo = img.squeeze(-4).squeeze(-3).detach().numpy()
    print(imgput.shape)
    imgshow = imgput.squeeze(-4).squeeze(-3).detach().numpy()

    # plt.imshow(imgshow, cmap='gray')
    plt.imshow(imgo, cmap='gray')
    plt.show()



    # data_loader = DataLoader(MyDataset('data'), batch_size=2, shuffle=True)
    # for image, segment_image in data_loader:
    # # data = MyDataset('data')
    # # print(data[0][0])
    #     print(image.shape, segment_image.shape)
