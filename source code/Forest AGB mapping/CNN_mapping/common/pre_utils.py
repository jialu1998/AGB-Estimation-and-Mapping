import torch
import numpy as np
import hdf5storage
import time
import gdal

def get_reconstruction_gpu(opt,input, model):
    """As the limited GPU memory split the input."""
    
    var_input = input.cuda()
    with torch.no_grad():
        start_time = time.time()
        var_output = model(var_input)
        
    if opt.model == 'AeroRIT':
        var_output = var_output.max(1)[1].unsqueeze(0)
        #print(var_output.shape)
        
    end_time = time.time()

    return end_time-start_time, var_output.cpu()


def copy_patch1(x, y):
    x[:] = y[:]


def copy_patch2(stride, h, x, y):
    #print(x.shape)
    #print(y.shape)
    x[:, :, :, :-(h % stride)] = (y[:, :, :, :-(h % stride)] + x[:, :, :, :-(h % stride)]) / 2.0
    x[:, :, :, -(h % stride):] = y[:, :, :, -(h % stride):]


def copy_patch3(stride, w, x, y):
    x[:, :, :-(w % stride), :] = (y[:, :, :-(w % stride), :] + x[:, :, :-(w % stride), :]) / 2.0
    x[:, :, -(w % stride):, :] = y[:, :, -(w % stride):, :]


def copy_patch4(stride, w, h, x, y):
    x[:, :, :-(w % stride), :] = (y[:, :, :-(w % stride), :] + x[:, :, :-(w % stride), :]) / 2.0
    x[:, :, -(w % stride):, :-(h % stride)] = (y[:, :, -(w % stride):, :-(h % stride)] + x[:, :, -(w % stride):, :-(h % stride)]) /2.0
    x[:, :, -(w % stride):, -(h % stride):] = y[:, :, -(w % stride):, -(h % stride):]


def reconstruction_patch_image_gpu(opt,rgb, model, patch, stride):
    all_time = 0
    _, _, w, h = rgb.shape
    rgb = torch.from_numpy(rgb).float()
    temp_hyper = torch.zeros(1, opt.outbands, w, h).float()
    # temp_rgb = torch.zeros(1, 3, w, h).float()
    for x in range(w//stride + 1):
        for y in range(h//stride + 1):
            if x < w // stride and y < h // stride:
                rgb_patch = rgb[:, :, x * stride:x * stride + patch, y * stride:y * stride + patch]
                patch_time, hyper_patch = get_reconstruction_gpu(opt,rgb_patch, model)
                # temp_hyper[:, :, x * stride:x * stride + patch, y * stride:y * stride + patch] = hyper_patch
                copy_patch1(temp_hyper[:, :, x * stride:x * stride + patch, y * stride:y * stride + patch], hyper_patch)
            elif x < w // stride and y == h // stride:
                rgb_patch = rgb[:, :, x * stride:x * stride + patch, -patch:]
                patch_time, hyper_patch = get_reconstruction_gpu(opt,rgb_patch, model)
                # temp_hyper[:, :, x * stride:x * stride + patch, -patch:] = hyper_patch
                copy_patch2(stride, h, temp_hyper[:, :, x * stride:x * stride + patch, -patch:], hyper_patch)
            elif x == w // stride and y < h // stride:
                rgb_patch = rgb[:, :, -patch:, y * stride:y * stride + patch]
                patch_time, hyper_patch = get_reconstruction_gpu(opt,rgb_patch, model)
                # temp_hyper[:, :, -patch:, y * stride:y * stride + patch] = hyper_patch
                copy_patch3(stride, w, temp_hyper[:, :, -patch:, y * stride:y * stride + patch], hyper_patch)
            else:
                rgb_patch = rgb[:, :, -patch:, -patch:]
                patch_time, hyper_patch = get_reconstruction_gpu(opt,rgb_patch, model)
                # temp_hyper[:, :, -patch:, -patch:] = hyper_patch
                copy_patch4(stride, w, h, temp_hyper[:, :, -patch:, -patch:], hyper_patch)
            all_time += patch_time

    img_res = temp_hyper.numpy() * 1.0

    img_res = np.transpose(img_res.squeeze(), [1, 2, 0])
    img_res_limits = np.minimum(img_res, 1.0)
    img_res_limits = np.maximum(img_res_limits, 0)
    return all_time, img_res


def load_img_to_array(img_file_path):
     """
     读取栅格数据，将其转换成对应数组
     :param: img_file_path: 栅格数据路径
     :return: 返回投影，几何信息，和转换后的数组(5888,5888,10)
     """
     dataset = gdal.Open(img_file_path)  # 读取栅格数据
     print('处理图像的栅格波段数总共有：', dataset.RasterCount)

     # 判断是否读取到数据 
     if dataset is None:
         print('Unable to open *.tif')
         sys.exit(1)  # 退出

     projection = dataset.GetProjection()  # 投影
     transform = dataset.GetGeoTransform()  # 几何信息

    # 直接读取dataset
     img_array = dataset.ReadAsArray()

     return projection, transform, img_array



def predit_to_tif( mat, projection, geo_transform , mapfile):
        """
        将数组转成tif文件写入硬盘
        :param mat: 数组
        :param projection: 投影信息
        :param tran: 几何信息
        :param mapfile: 文件路径
        :return:
        """

        row = mat.shape[0]  # 矩阵的行数
        columns = mat.shape[1]  # 矩阵的列数

        print(geo_transform)

        dim_z = mat.shape[2]  # 通道数

        driver = gdal.GetDriverByName('GTiff')  # 创建驱动
        # 创建文件
        dst_ds = driver.Create(mapfile, columns, row, dim_z, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(geo_transform)  # 设置几何信息
        dst_ds.SetProjection(projection)  # 设置投影

        # 将数组的各通道写入tif图片
        for channel in np.arange(dim_z):
            map = mat[:, :, channel]
            dst_ds.GetRasterBand(int(channel + 1)).WriteArray(map)

        dst_ds.FlushCache()  # 写入硬盘
        dst_ds = None


def get_labels():
        return np.asarray(
                [
                        [255, 0, 0],
                        [0, 255, 0],
                        [0, 0, 255],
                        [0, 255, 255],
                        [255, 127, 80],
                        [153, 0, 0],
                        ]
                )
def decode_segmap(label_mask,n_classes, plot=False):
        label_colours = get_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, n_classes ):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        
        return np.uint8(rgb)
