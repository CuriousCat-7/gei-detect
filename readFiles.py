import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.image
import os

#数据
#sequence 00 as probe, sequence 01 as gallery, both are used for training
#X:data, Y:id, p:probe, g:gallery [00]:angle
Xp_00=[]
Xg_00=[]
Xp_90=[]
Xg_90=[]

Yp_00=[]
Yg_00=[]
Yp_90=[]
Yg_90=[]

#id_list = np.array(pd.read_csv('../input/fashion-mnist_train.csv'))

id_list = np.zeros([15,2]).astype(int)#################################

#路径
root_dir = os.path.join("../input")
Xp_00_dir = os.path.join(root_dir, "000-00", "000-00")
Xg_00_dir = os.path.join(root_dir, "000-01", "000-01")
Xp_90_dir = os.path.join(root_dir, "090-00", "090-00")
Xg_90_dir = os.path.join(root_dir, "090-01", "090-01")

#id_table: 每个id对应GEI在矩阵中的索引，不存在的为-1 [id,Xp_00,Xg_00,Xp_90,Xp_90]
id_table=np.ones([id_list.shape[0],5])*(-1)
id_table[:,0]=id_list[:,0]

#读取数据
for i in range(0,id_list.shape[0]):
    id_list[i,0] = (i*2+1)#################################
    id_table[i,0]=id_list[i,0]#################################
    if os.path.exists(os.path.join(Xp_00_dir,str(id_list[i,0]).zfill(5)+".png")):
        image = matplotlib.image.imread(os.path.join(Xp_00_dir,str(id_list[i,0]).zfill(5)+".png"))
        Xp_00.append(image)
        Yp_00.append(id_list[i,0])
        id_table[i,1]=len(Xp_00)-1
    if os.path.exists(os.path.join(Xg_00_dir,str(id_list[i,0]).zfill(5)+".png")):
        image = matplotlib.image.imread(os.path.join(Xg_00_dir,str(id_list[i,0]).zfill(5)+".png"))
        Xg_00.append(image)
        Yg_00.append(id_list[i,0])
        id_table[i,2]=len(Xg_00)-1
    if os.path.exists(os.path.join(Xp_90_dir,str(id_list[i,0]).zfill(5)+".png")):
        image = matplotlib.image.imread(os.path.join(Xp_90_dir,str(id_list[i,0]).zfill(5)+".png"))
        Xp_90.append(image)
        Yp_90.append(id_list[i,0])
        id_table[i,3]=len(Xp_90)-1
    if os.path.exists(os.path.join(Xg_90_dir,str(id_list[i,0]).zfill(5)+".png")):
        image = matplotlib.image.imread(os.path.join(Xg_90_dir,str(id_list[i,0]).zfill(5)+".png"))
        Xg_90.append(image)
        Yg_90.append(id_list[i,0])
        id_table[i,4]=len(Xg_90)-1

Xp_00 = np.array(Xp_00) #
Xg_00 = np.array(Xg_00)
Xp_90 = np.array(Xp_90)
Xg_90 = np.array(Xg_90)

Yp_00 = np.array(Yp_00)
Yg_00 = np.array(Yg_00)
Yp_90 = np.array(Yp_90)
Yg_90 = np.array(Yg_90)

print(Xp_00.shape)
print(Xg_00.shape)
print(Xp_90.shape)
print(Xg_90.shape)
print(Yp_00.shape)
print(Yg_00.shape)
print(Yp_90.shape)
print(Yg_90.shape)