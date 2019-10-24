
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
import numpy as np
import random



def to_img(x):
    #x = 0.5 * (x + 1)
    #x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 84, 84)
    return x



class NpyDataset(Dataset):
    """Torch Dataset for npy files"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the npy files
            transform (callable, optional): Optional transform to be applied
                on a sample.
            """
        self.root_dir = root_dir
        self.transform = transform

        data_files = os.listdir(root_dir)
        random.shuffle(data_files)

        self.data = None

        for i in range(20):
            new_data = np.load('{}/{}'.format(root_dir, data_files[i]))
            if self.data is None:
                self.data = new_data
            else:
                self.data = np.concatenate((self.data, new_data))
                print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 2, stride=2),
            nn.Sigmoid()
            #nn.ReLU(True),
            #nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),
            #nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    if not os.path.exists('./dc_img'):
        os.mkdir('./dc_img')

    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-3

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_path = './observations/'
    train_dataset = NpyDataset(
            root_dir=data_path,
            transform=None
            )
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            num_workers=0,
            shuffle=True
            )



    model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)

    for epoch in range(num_epochs):
        for batch_idx, img_batch in enumerate(train_loader):
            img_batch = Variable(img_batch).cuda()
            # ===================forward=====================
            output = model(img_batch)
            loss = criterion(output, img_batch)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.item()))
        #if epoch % 10 == 0:
        #    pic = to_img(output.detach().cpu().numpy()[0])
        #    save_image(pic, './dc_img/image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './conv_autoencoder.pth')
