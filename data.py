import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.image
import os
import torch.utils.data as data
import random

class GeiImageFileCSV(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, shuffle=False, train=True):
        self.root = root
        self.transfrom =transform
        self.target_transform = target_transform
        self.shuffle=False
        self.train=train
        self.list=[]
        self.create_image_list
        self.fr = 1
        self.to = 10307
        self.Xp00, self.Xg00, self.Xp90, self.Xg90, self.idx, self.flag = Xp00, Xg00, Xp90, Xg90, idx, flag
    def __getitem__(self, index):
        pass
    def __len__(self):
        return len(self.ids)
    def create_image_list(self):
        root_dir = self.root
        Xp_00_dir = os.path.join(root_dir,  "000-00")
        Xg_00_dir = os.path.join(root_dir,  "000-01")
        Xp_90_dir = os.path.join(root_dir,  "090-00")
        Xg_90_dir = os.path.join(root_dir,  "090-01")
        flag = []
        ids = []
        Xp_00=[]
        Xg_00=[]
        Xp_90=[]
        Xg_90=[]
        Yp_00=[]
        Yg_00=[]
        Yp_90=[]
        Yg_90=[]

        if self.train:
            for i in xrange(self.fr, self.to, 2):
                try:
                    image_p00 = matplotlib.image.imread(os.path.join(Xp_00_dir,str(id_list[i,0]).zfill(5)+".png"))
                    image_g00 = matplotlib.image.imread(os.path.join(Xg_00_dir,str(id_list[i,0]).zfill(5)+".png"))
                    image_p90 = matplotlib.image.imread(os.path.join(Xp_90_dir,str(id_list[i,0]).zfill(5)+".png"))
                    image_g90 = matplotlib.image.imread(os.path.join(Xg_90_dir,str(id_list[i,0]).zfill(5)+".png"))
                    Xp00.append(image_p00)
                    Xg00.append(image_g00)
                    Xp90.append(image_p90)
                    Xg90.append(image_g90)
                    ids.append(i)
                    flag.append(1)
                except:
                    try:
                        image_g00 = matplotlib.image.imread(os.path.join(Xg_00_dir,str(id_list[i,0]).zfill(5)+".png"))
                        image_p90 = matplotlib.image.imread(os.path.join(Xp_90_dir,str(id_list[i,0]).zfill(5)+".png"))
                        Xp00.append(None)
                        Xg00.append(image_g00)
                        Xp90.append(image_p90)
                        Xg90.append(None)
                        ids.append(i)
                        flag.append(2)
        if self.test:
            for i in xrange(self.fr+1, self.to, 2):
                try:
                    image_g00 = matplotlib.image.imread(os.path.join(Xg_00_dir,str(id_list[i,0]).zfill(5)+".png"))
                    image_p90 = matplotlib.image.imread(os.path.join(Xp_90_dir,str(id_list[i,0]).zfill(5)+".png"))
                    Xp00.append(None)
                    Xg00.append(image_g00)
                    Xp90.append(image_p90)
                    Xg90.append(None)
                    ids.append(i)
                    flag.append(2)
        
        return Xp00, Xg00, Xp90, Xg90, idx, flag

