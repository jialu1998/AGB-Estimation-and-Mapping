import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import time
import random
from AWAN.AWAN import AWAN
from AWAN.dataset import *
from AWAN.utils import *
from common.tool import setup_seed, save_checkpoint, AverageMeter
import logging


class HyperDataset(udata.Dataset):
    def __init__(self, path, set_type, hsi_sign):

        self.path = path
        self.set_type = set_type
        with open(os.path.join(self.path, self.set_type + ".txt")) as f:
            self.keys = f.read().splitlines()

        random.shuffle(self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # print(os.path.join(self.path,'Data',self.keys[index]+'.mat'))
        mat = h5py.File(
            os.path.join(self.path, "Data", self.keys[index] + ".mat"), "r"
        )  # 放入拆分完简介爱

        rgb = np.float32(np.array(mat["rgb"]))
        # rgb = rgb / 60  # 除最大值，59_10
        rgb = rgb / 100  # 除最大值,113_10

        rgb = np.transpose(rgb, [2, 0, 1])
        rgb = torch.Tensor(rgb)

        label = np.float32(np.array(mat["label"]))
        label = label / 1200

        label = np.transpose(label, [2, 0, 1])
        label = torch.Tensor(label)

        mat.close()

        return rgb, label


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    iteration,
    init_lr,
    decay_power,
    opt,
):
    model.train()
    random.shuffle(train_loader)
    losses = AverageMeter()

    for k, train_data_loader in enumerate(train_loader):
        for i, (images, labels) in enumerate(train_data_loader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)
            lr = poly_lr_scheduler(
                optimizer, init_lr, iteration, max_iter=opt.max_iter, power=decay_power
            )
            iteration = iteration + 1
            output = model(images)
            loss = criterion(output, labels)
            loss_all = loss
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            losses.update(loss.data)
            print(
                "[Epoch:%02d],[Process:%d/%d],[iter:%d],lr=%.9f,train_losses.avg=%.9f"
                % (epoch, k + 1, len(train_loader), iteration, lr, losses.avg)
            )

    return losses.avg, iteration, lr


def validate(val_loader, model, criterion):
    model.eval()
    losses = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
        losses.update(loss.data)

    return losses.avg


def poly_lr_scheduler(
    optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9
):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr * (1 - iteraion / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, val_loss):
    loss_csv.write(
        "%d, %d, %.9f, %.9f, %.9f, %.9f\n"
        % (epoch, iteration, epoch_time, lr, train_loss, val_loss)
    )
    loss_csv.flush()


def initialize_logger(log_dir):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
    file_handler = logging.FileHandler(log_dir)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        "epoch": epoch,
        "iter": iteration,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, os.path.join(model_path, "net_%depoch.pth" % epoch))


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description="SSR")
    parser.add_argument("--seed", type=int, default=25, help="seed")
    parser.add_argument("--batchSize", type=int, default=20, help="batch size")
    parser.add_argument(
        "--end_epoch", type=int, default=100 + 1, help="number of epochs"
    )
    parser.add_argument(
        "--init_lr", type=float, default=1e-4, help="initial learning rate"
    )
    parser.add_argument("--decay_power", type=float, default=1.5, help="decay power")
    parser.add_argument("--max_iter", type=float, default=305000, help="max_iter")
    parser.add_argument(
        "--outf", type=str, default="weight_59113_20", help="path log files"
    )
    opt = parser.parse_args(args=[])

    setup_seed(opt.seed)

    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    print("\nloading dataset ...")
    train_data = HyperDataset("/home/vipuser/下载/split_59113_new_20", "train", "rad")
    print("train_data set samples: ", len(train_data))
    val_data = HyperDataset("/home/vipuser/下载/split_59113_new_20", "val", "rad")
    print("val_data set samples: ", len(val_data))

    train_loader1 = DataLoader(
        dataset=train_data,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    train_loader = [train_loader1]
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = AWAN(20, 1, 100, 4)
    print("Parameters number is ", sum(param.numel() for param in model.parameters()))

    criterion = torch.nn.L1Loss()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    start_epoch = 0
    iteration = 0
    record_val_loss = 0.1
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.init_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
    )

    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    loss_csv = open(os.path.join(opt.outf, "loss.csv"), "a+")
    log_dir = os.path.join(opt.outf, "train.log")
    logger = initialize_logger(log_dir)

    resume_file = ""
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint["epoch"]
            iteration = checkpoint["iter"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])

    for epoch in range(start_epoch + 1, opt.end_epoch):
        start_time = time.time()
        train_loss, iteration, lr = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            iteration,
            opt.init_lr,
            opt.decay_power,
            opt,
        )
        val_loss = validate(val_loader, model, criterion)

        if epoch % 10 == 0:
            save_checkpoint(opt.outf, epoch, iteration, model, optimizer)
            print("epoch Model Saved")

        if val_loss < record_val_loss:
            record_val_loss = val_loss
            save_checkpoint(opt.outf, epoch, iteration, model, optimizer)
            print("loss Model Saved")

        end_time = time.time()
        epoch_time = end_time - start_time
        print(
            "Epoch [%02d], Iter[%06d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f "
            % (epoch, iteration, epoch_time, lr, train_loss, val_loss)
        )
        record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, val_loss)
        logger.info(
            "Epoch [%02d], Iter[%06d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f "
            % (epoch, iteration, epoch_time, lr, train_loss, val_loss)
        )


if __name__ == "__main__":
    main()
