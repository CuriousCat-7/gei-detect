import numpy as np
import pandas as pd
import matplotlib.image
import os
import torch.utils.data as data
import random

class GeiImageFileCSV(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, shuffle=False, train=True, mode='full'): # mode = 'full', 'trichannel','pair1', 'pair2', '90&00'
        self.root = root
        self.transfrom =transform
        self.target_transform = target_transform
        self.shuffle=False
        self.train=train
        self.list=[]
        self.create_image_list
        self.mode = mode
        self.fr = 1
        self.to = 10307
        self.Xp00, self.Xg00, self.Xp90, self.Xg90, self.ids = self.create_image_list()
    def __getitem__(self, index):
        return self.Xp00[index], self.Xp90[index], self.Xg00[index], self.Xg90[index], self.ids[index]
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
        Xp00=[]
        Xg00=[]
        Xp90=[]
        Xg90=[]
        Yp00=[]
        Yg00=[]
        Yp90=[]
        Yg90=[]

        if self.train: # flag 1 means have all data, flag 2 means only have probe 90 and gallery 00, flag 3 means have probe 00 and gallery 90
            for i in xrange(self.fr, self.to, 2):
                if self.mode == 'full':
                    try:
                        image_p00 = matplotlib.image.imread(os.path.join(Xp_00_dir,str(i).zfill(5)+".png"))
                        image_g00 = matplotlib.image.imread(os.path.join(Xg_00_dir,str(i).zfill(5)+".png"))
                        image_p90 = matplotlib.image.imread(os.path.join(Xp_90_dir,str(i).zfill(5)+".png"))
                        image_g90 = matplotlib.image.imread(os.path.join(Xg_90_dir,str(i).zfill(5)+".png"))
                        Xp00.append(image_p00)
                        Xg00.append(image_g00)
                        Xp90.append(image_p90)
                        Xg90.append(image_g90)
                        ids.append(i)
                    except:
                        pass
                elif self.mode == 'pair1':
                    try:
                        image_g00 = matplotlib.image.imread(os.path.join(Xg_00_dir,str(i).zfill(5)+".png"))
                        image_p90 = matplotlib.image.imread(os.path.join(Xp_90_dir,str(i).zfill(5)+".png"))
                        Xp00.append(None)
                        Xg00.append(image_g00)
                        Xp90.append(image_p90)
                        Xg90.append(None)
                        ids.append(i)
                    except:
                        pass
                elif self.mode == 'pair2':

                    try:
                        image_p00 = matplotlib.image.imread(os.path.join(Xp_00_dir,str(i).zfill(5)+".png"))
                        image_g90 = matplotlib.image.imread(os.path.join(Xg_90_dir,str(i).zfill(5)+".png"))
                        Xp00.append(image_p00)
                        Xg00.append(None)
                        Xp90.append(None)
                        Xg90.append(image_g90)
                        ids.append(i)
                    except:
                        pass
                elif self.mode == 'trichannel':
                    try:
                        image_p00 = matplotlib.image.imread(os.path.join(Xp_00_dir,str(i).zfill(5)+".png"))
                        image_p90 = matplotlib.image.imread(os.path.join(Xp_90_dir,str(i).zfill(5)+".png"))
                        image_g90 = matplotlib.image.imread(os.path.join(Xg_90_dir,str(i).zfill(5)+".png"))
                        Xp00.append(image_p00)
                        Xg00.append(None)
                        Xp90.append(image_p90)
                        Xg90.append(image_g90)
                        ids.append(i)
                    except:
                        pass
                elif self.mode == '90&00':
                    try:
                        image_p00 = matplotlib.image.imread(os.path.join(Xp_00_dir,str(i).zfill(5)+".png"))
                        image_p90 = matplotlib.image.imread(os.path.join(Xp_90_dir,str(i).zfill(5)+".png"))
                        Xp00.append(image_p00)
                        Xg00.append(None)
                        Xp90.append(image_p90)
                        Xg90.append(None)
                    except:
                        try:
                            image_g00 = matplotlib.image.imread(os.path.join(Xg_00_dir,str(i).zfill(5)+".png"))
                            image_g90 = matplotlib.image.imread(os.path.join(Xg_90_dir,str(i).zfill(5)+".png"))
                            Xp00.append(None)
                            Xg00.append(image_g00)
                            Xp90.append(None)
                            Xg90.append(image_g90)
                        except:
                            pass


        else:
            for i in xrange(self.fr+1, self.to, 2):
                try:
                    image_g00 = matplotlib.image.imread(os.path.join(Xg_00_dir,str(i).zfill(5)+".png"))
                    image_p90 = matplotlib.image.imread(os.path.join(Xp_90_dir,str(i).zfill(5)+".png"))
                    Xp00.append(None)
                    Xg00.append(image_g00)
                    Xp90.append(image_p90)
                    Xg90.append(None)
                    ids.append(i)
                except:
                    pass

        return Xp00, Xg00, Xp90, Xg90, ids

if __name__ == "__main__":
    dataset =  GeiImageFileCSV(root = '/data/limingyao/data/gei/dataset/', mode='pair1')
    print len(dataset)
    print dataset[20]

