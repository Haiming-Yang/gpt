import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import PIL, io
import zipfile

class CIFAR10Dataset(Dataset):
    def __init__(self, train=True, root_dir='data',download=True):
        super(CIFAR10Dataset, self).__init__()
        self.dataset =  torchvision.datasets.CIFAR10(root=root_dir, train=train , download=download)
        self.label_to_name = ['airplane','automobile', 'bird','cat','deer','dog','frog','horse', 'ship','truck',]

    def __getitem__(self,index):
        img,y0 = self.dataset.__getitem__(index)
        img = np.array(img).transpose(2,0,1)/255.
        # print(img.shape, y0)
        return img, y0

    def __len__(self):
        return self.dataset.__len__()

    def demo_some_images(self):
        plt.figure(figsize=(8,6))
        for i in range(20):
            plt.gcf().add_subplot(4,5,i+1)
            img, y0 = self.__getitem__(i)
            title = '%s:%s'%(str(y0),str(self.label_to_name[y0]))
            plt.gca().imshow(img.transpose(1,2,0))
            plt.gca().set_title(title)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
        plt.tight_layout()
        plt.show()

def prepare_cifarloader(train=True, root_dir='data', batch_size=4, shuffle=True, demo=False, download=False):
    cif = CIFAR10Dataset(train=train, root_dir=root_dir, download=download)
    if demo:  cif.demo_some_images()
    loader = DataLoader(cif, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader


def bytes_to_ndarray(bytes):
    bytes_io = bytearray(bytes)
    img = PIL.Image.open(io.BytesIO(bytes_io))
    return np.array(img)

class CelebADataset(Dataset):
    def __init__(self, PATH_TO_IMG, img_size=None):
        super(CelebADataset, self).__init__()
        self.dir = PATH_TO_IMG
        self.zf = zipfile.ZipFile(PATH_TO_IMG)
        self.itemlist = self.zf.namelist()[1:] # the first entry is the name of the folder
        self.n_data = len(self.itemlist)

        self.img_size = img_size # None, or (H,w)

        if img_size is not None:
            H, W = img_size
            self.transf = transf = torchvision.transforms.Compose([
                torchvision.transforms.Resize((H,W)),
            ])

    def __getitem__(self,index):
        img = self.zf.read(self.itemlist[index])

        # if use cv2
        # img = cv2.imdecode(np.frombuffer(img, np.uint8), 1)    
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = bytes_to_ndarray(np.frombuffer(img, np.uint8))
        img = img/255.
        img = img.transpose(2,0,1)

        if self.img_size is not None:
            img = torch.tensor(img).to(torch.float)
            img = self.transf(img)
            img = np.asarray(img).copy()
          
        return img, 0 # 0 is a dummy label

    def __len__(self):
        return self.n_data


def prepare_celebaloader(img_size=(64,64),train=True, root_dir='data', batch_size=4, shuffle=True):
    dat = CelebADataset(root_dir,img_size=img_size)
    loader = DataLoader(dat, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader
