import torch
import torch.nn as nn
import torch.nn.functional as F

class AcrNET(nn.Module):
    def __init__(self):
        super(AcrNET, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 34), 
                               stride=(1, 1))

        self.dnn_fc1 = nn.Linear(2390, 256)
        self.dnn_fc2 = nn.Linear(256, 32)

        self.fc1 = nn.Linear(52, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, seq, ss3, ss8, acc, dnn_feature):
        # seq
        cnn_feature = torch.cat((seq, ss3, ss8, acc), -1)
        cnn_feature = self.conv(cnn_feature).squeeze()
        cnn_feature = F.relu(F.max_pool1d(cnn_feature, cnn_feature.size(2)).squeeze())

        # pssm + transformer
        x = F.relu(self.dnn_fc1(dnn_feature))
        x = self.dnn_fc2(x)

        x = torch.cat((cnn_feature, x), 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        pred = F.log_softmax(x, dim=1)

        return pred

