import os
import random

# 设置数据目录
data_dir = "/home/vipuser/下载/split_59113_new_20/Data"
out_dir = "/home/vipuser/下载/split_59113_new_20/"


# 获取所有 tif 文件的文件名列表
all_files = os.listdir(data_dir)
tif_files = [f[:-4] for f in all_files if f.endswith(".mat")]

# 随机打乱文件列表
random.shuffle(tif_files)

# 按照 7:2:1 的比例划分为训练集、验证集和测试集
train_size = int(len(tif_files) * 0.7)
val_size = int(len(tif_files) * 0.2)
test_size = len(tif_files) - train_size - val_size

train_files = tif_files[:train_size]
val_files = tif_files[train_size : train_size + val_size]
test_files = tif_files[train_size + val_size :]


# 将文件名写入到对应的 txt 文件中
with open(os.path.join(out_dir, "train.txt"), "w") as f:
    f.write("\n".join(train_files))

with open(os.path.join(out_dir, "val.txt"), "w") as f:
    f.write("\n".join(val_files))

with open(os.path.join(out_dir, "test.txt"), "w") as f:
    f.write("\n".join(test_files))

print("Data split and file lists saved successfully!")
