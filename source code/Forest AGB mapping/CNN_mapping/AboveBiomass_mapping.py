import argparse
import logging
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from AWAN.AWAN import AWAN
from common.tool import (
    AverageMeter,
    initialize_logger,
    load_weight,
    save_checkpoint,
    setup_seed,
)
from osgeo import gdal
from tqdm import tqdm


def create_chips(
    agbd_path,
    band_paths_59,
    band_paths_113,
    model,
    output_band,
    chip_size=128,
    overlap=64,
):

    print("Starting chip making now")

    # Open datasets
    in_ds_AGBD = gdal.Open(agbd_path)
    in_ds_bands_59 = [gdal.Open(path) for path in band_paths_59]
    in_ds_bands_113 = [gdal.Open(path) for path in band_paths_113]

    # Get image properties
    width, height = in_ds_AGBD.RasterXSize, in_ds_AGBD.RasterYSize
    col_num = (width - overlap) // (chip_size - overlap) + 1
    row_num = (height - overlap) // (chip_size - overlap) + 1
    print(f"row_num: {row_num}   col_num: {col_num}")

    for i in tqdm(range(row_num)):
        for j in range(col_num):
            # Calculate the offsets with overlap
            offset_x = i * (chip_size - overlap)
            offset_y = j * (chip_size - overlap)
            b_xsize = min(chip_size, width - offset_y)
            b_ysize = min(chip_size, height - offset_x)

            # Read data from bands
            bands_data = [
                band.ReadAsArray(offset_y, offset_x, b_xsize, b_ysize)
                for band in in_ds_bands_59 + in_ds_bands_113
            ]

            # Create a 3D array and preprocess
            mat_array = np.concatenate(
                [np.expand_dims(band, axis=2) for band in bands_data], axis=2
            )
            mat_array[np.isnan(mat_array)] = 0
            mat_array[np.isinf(mat_array)] = 0

            # Convert to tensor and predict
            X_tensor = load_tif_to_tensor(mat_array)

            with torch.no_grad():  # 在预测时不需要计算梯度
                y_pred = model(X_tensor.cuda()) * 1200

            original_data = y_pred.data.cpu().numpy().squeeze()

            extracted_region = original_data[32:96, 32:96]

            # # 创建一个新的数组并填充 0
            # padded_array = np.zeros((128, 128), dtype=original_data.dtype)

            # # 将提取的区域放入新的数组中心
            # padded_array[start_x : start_x + 64, start_y : start_y + 64] = (
            #     extracted_region
            # )

            # Write output
            output_band.WriteArray(extracted_region, offset_y + 32, offset_x + 32)

            # 设置地理参考信息
            output_band.GetDataset().SetGeoTransform(in_ds_AGBD.GetGeoTransform())
            output_band.GetDataset().SetProjection(in_ds_AGBD.GetProjection())

    # Close datasets
    in_ds_AGBD, in_ds_bands_59, in_ds_bands_113 = None, None, None


def load_tif_to_tensor(tif_file):
    X = np.transpose(tif_file, [2, 0, 1])
    rgb = np.float32(np.expand_dims(X, axis=0)) / 100  # 除最大值
    return torch.Tensor(rgb)


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description="SSR")
    parser.add_argument("--seed", type=int, default=25, help="seed")
    parser.add_argument(
        "--weight_path",
        type=str,
        default="/home/vipuser/下载/weight_59113_20/net_68epoch.pth",
        help="path to weights",
    )

    parser.add_argument(
        "--outtif_path",
        type=str,
        default="outtif_59113_20",
        help="path to weights",
    )

    parser.add_argument("--data_part", type=str, default="test", help="data partition")
    parser.add_argument("--Type", type=str, default="rad", help="data type")

    opt = parser.parse_args()

    setup_seed(opt.seed)

    model = AWAN(20, 1, 100, 4)
    print("Parameters number is", sum(param.numel() for param in model.parameters()))

    model = load_weight(model, opt.weight_path)

    # 检查文件是否存在
    if not os.path.exists(opt.outtif_path):
        # 如果不存在，则创建目录
        os.makedirs(opt.outtif_path)
        print(f"目录 '{opt.outtif_path}' 不存在，已创建。")
    else:
        print(f"目录 '{opt.outtif_path}' 已存在。")

    ########### 构建输入x

    agbd_path = "/home/vipuser/下载/AGBD_avg_New_England_ProjectUTM_roi_mask1.tif/AGBD_avg_New_England_ProjectUTM_roi_mask1.tif"
    band_paths_59 = [
        "/home/vipuser/下载/MAPNewEngland59model_RH95_delete_prjutm_roi_mask1.tif/MAPNewEngland59model_RH95_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland59model_RH90_delete_prjutm_roi_mask1.tif/MAPNewEngland59model_RH90_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland59model_RH98_delete_prjutm_roi_mask1.tif/MAPNewEngland59model_RH98_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland59model_RH85_delete_prjutm_roi_mask1/MAPNewEngland59model_RH100_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland59model_RH85_delete_prjutm_roi_mask1/MAPNewEngland59model_RH85_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland59model_CC1_delete_prjutm_roi_mask1/MAPNewEngland59model_RH80_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland59model_CC1_delete_prjutm_roi_mask1/MAPNewEngland59model_RH75_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland59model_CC1_delete_prjutm_roi_mask1/MAPNewEngland59model_CC2_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland59model_CC1_delete_prjutm_roi_mask1/MAPNewEngland59model_h_canopy_q_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland59model_CC1_delete_prjutm_roi_mask1/MAPNewEngland59model_CC1_delete_prjutm_roi_mask1.tif",
    ]
    band_paths_113 = [
       "/home/vipuser/下载/MAPNewEngland113model_RH100_delete_prjutm_roi_mask1/MAPNewEngland113model_RH100_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland113model_RH100_delete_prjutm_roi_mask1/MAPNewEngland113model_RH98_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland113model_RH100_delete_prjutm_roi_mask1/MAPNewEngland113model_RH95_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland113model_RH100_delete_prjutm_roi_mask1/MAPNewEngland113model_canopy_ope_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland113model_RH100_delete_prjutm_roi_mask1/MAPNewEngland113model_RH90_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland113model_RH100_delete_prjutm_roi_mask1/MAPNewEngland113model_RH85_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland113model_RH100_delete_prjutm_roi_mask1/MAPNewEngland113model_h_dif_canopy_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland113model_RH100_delete_prjutm_roi_mask1/MAPNewEngland113model_RH80_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland113model_RH100_delete_prjutm_roi_mask1/MAPNewEngland113model_RH75_delete_prjutm_roi_mask1.tif",
        "/home/vipuser/下载/MAPNewEngland113model_RH100_delete_prjutm_roi_mask1/MAPNewEngland113model_CC1_delete_prjutm_roi_mask1.tif",
    ]

    ########构建输出y

    driver = gdal.GetDriverByName("GTiff")
    # 定义输出文件名并结合路径
    output_file_name = "predicted_59113_20_net68.tif"
    output_file = os.path.join(opt.outtif_path, output_file_name)
    # 判断文件是否存在
    if os.path.isfile(output_file):
        # 删除文件
        os.remove(output_file)
        print(f"文件 '{output_file}' 已被删除。")
    else:
        print(f"文件 '{output_file}' 不存在。")

    # 输出文件的完整路径
    print(f"输出文件的完整路径为: {output_file}")
    in_ds_AGBD = gdal.Open(
        "/home/vipuser/下载/AGBD_avg_New_England_ProjectUTM_roi_mask1.tif/AGBD_avg_New_England_ProjectUTM_roi_mask1.tif"
    )

    width = in_ds_AGBD.RasterXSize  # 获取数据宽度
    height = in_ds_AGBD.RasterYSize  # 获取数据高度
    outbandsize = in_ds_AGBD.RasterCount  # 获取数据波段数
    im_geotrans = in_ds_AGBD.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = in_ds_AGBD.GetProjection()  # 获取投影信息
    datatype = in_ds_AGBD.GetRasterBand(1).DataType
    im_data = in_ds_AGBD.ReadAsArray()  # 获取数据

    output_dataset = driver.Create(output_file, width, height, 1, gdal.GDT_Float32)

    output_dataset.SetGeoTransform(im_geotrans)
    output_dataset.SetProjection(im_proj)

    output_dataset.GetRasterBand(1).WriteArray(
        np.zeros((height, width))
    )  # 修正矩阵大小

    output_band = output_dataset.GetRasterBand(1)

    create_chips(agbd_path, band_paths_59, band_paths_113, model, output_band)


if __name__ == "__main__":
    main()
