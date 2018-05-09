import torch
from torch import nn
from torch.nn import functional as F

class SimpleModel(nn.Module):
    def __init__(self,):
        super(SimpleModel, self).__init__()
        pass
    def forward(self,x):
        pass


class TrichannelModel(nn.Module):
    def __init__(self,):
        super( TrichannelModel, self).__init__()
        self.cnn1 = nn.Sequential(
                nn.Conv2d(1,32, kernel_size=5, stride=1, padding =2),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(32,64, 5, 1, 2),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2),
            )

        self.cnn2 = nn.Sequential(
                nn.Conv2d(1,32, kernel_size=5, stride=1, padding =2),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(32,64, 5, 1, 2),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2),
            )

        self.cnn3 = nn.Sequential(
                nn.Conv2d(64,64, kernel_size=3, stride=1, padding =1),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2),
            )
        self.fc = nn.Linear(64*(128/2/2/2)*(88/2/2/2), 256)
        self.criteria_auxi = F.mse_loss
        self.criteria_main = F.mse_loss # dai ding

    def forward(self,pro90, pro00, gal90):
        batch_size = pro90.size(0)
        x1 = self.cnn2(pro90)
        x2 = self.cnn1(pro00)
        x3 = self.cnn2(gal90)
        x2_1 = self.cnn3(x2)
        x3_1 = self.cnn3(x3)
        x2_2 = self.fc( x2_1.view(batch_size,-1))
        x3_2 = self.fc( x3_1.view(batch_size,-1))

        loss1 = self.criteria_auxi(x1, x2)
        loss2 = self.criteria_auxi(x2_2, x3_2)
        return loss1, loss2

if __name__ == "__main__":
    net = TrichannelModel()
    net.cuda()
    optim = torch.optim.Adam(net.parameters(), lr = 0.0001)
    from data import GeiImageFileCSV
    dataset = GeiImageFileCSV(root = '/data/limingyao/data/gei/dataset/',mode='trichannel')
    print len(dataset)
    # for train in one batch
    pro00, pro90, gal00, gal90, id = dataset[0]
    pro00 = torch.from_numpy(pro00).cuda().unsqueeze(0).unsqueeze(0)
    pro90 = torch.from_numpy(pro90).cuda().unsqueeze(0).unsqueeze(0)
    gal90 = torch.from_numpy(gal90).cuda().unsqueeze(0).unsqueeze(0)

    loss1 , loss2 = net(pro90, pro00, gal90)
    alpha = 0.5
    loss = alpha* loss1 + (1-alpha)*loss2
    optim.zero_grad()
    loss.backward()
    optim.step()






