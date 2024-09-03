# -*- coding: utf-8 -*-
import argparse
import datetime
import os.path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging

import torch.nn.functional as F
# from STGCN import *
from STGCN import ST_GCN_only_reg, ST_GCN_only_cls, ST_GCN, ST_GCN_split, ST_GCN_split_plus, ST_GCN_W, \
    ST_GCN_DeepBackbone
from STGCN import ST_GCN_L, ST_GCN_MLPs, ST_GCN_DeepMLPs, ST_GCN_DeepBackbone_MLPs
from STGCN import SST_GCN, STD_GCN, SSTD_GCN, ST_GCN_multi_reg, Siamese_SSTD_GCN_MLPs_only_reg
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import MultivariateNormal as MVN


def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../数据增强/输出文件/3D太极拳第二式",
                        help="npy数据文件的路径")
    parser.add_argument("--model", type=str, default="Siamese_ST_GCN_DWBackbone_MLPs_only_reg",
                        help="ST_GCN_only_reg, ST_GCN_only_cls, ST_GCN, ST_GCN_split, ST_GCN_split_plus, ST_GCN_W, ST_GCN_DeepBackbone, STGCN import ST_GCN_L, ST_GCN_MLPs, ST_GCN_DeepMLPs, ST_GCN_DeepBackbone_MLPs, SST_GCN, STD_GCN, SSTD_GCN, ST_GCN_multi_reg, Siamese_SSTD_GCN_MLPs_only_reg")
    parser.add_argument('--load_weight', type=str, default=None,
                        help="预训练权重")  # 训练日志/第二式分数回归_暹罗网络BMSE/2023-12-20 15:59:40/stgcn_epoch=468_loss=0.001957362177091892.pt
    parser.add_argument("--criterion", type=str, default="BMSE", help="损失函数, MAE、MSE、BMSE")
    parser.add_argument("--output", type=str, default="训练日志", help="训练日志存放路径")
    parser.add_argument('--device', type=str, default="cuda:0", help="cpu, cuda:0, cuda:1 ...")
    parser.add_argument('--random_seed', type=float, default=3407, help='随机数种子')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--val_per_epochs', default=5, type=int, help='训练时每隔多少epochs验证一次')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs(iteration)')
    parser.add_argument('--lr_schedule', default=None, type=str,
                        help='学习率衰减策略: cosine_anneal, reduceonplateau or lambda')
    parser.add_argument('--t_v_rate', default=0.7, type=float, help='训练集占比')
    parser.add_argument('--hop_size', default=4, type=int, help='关键点邻接距离')
    parser.add_argument('--t_kernel_size', default=12, type=int, help='时域滑动窗口')
    parser.add_argument('--num_workers', default=16, type=int, help='多线程, default = 16')
    parser.add_argument('--accelerate', default=False, type=bool, help='pytorch2.0 accurate')
    args = parser.parse_args()

    # 记录日志并同时输出
    try:
        os.makedirs(args.output)
    except:
        pass
    logging.basicConfig(filename=args.output + '/{}.log'.format(datetime.datetime.now().replace(microsecond=0)),
                        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    logging.info("*" * 50 + "\nParsed arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    logging.info("*" * 50)
    return args


def random_seed(seed):
    logging.info("启用torch.manual_seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(opt):
    name_files = os.listdir(opt.data_dir)
    possible_data = [i for i in name_files if "aug_data" and ".npy" in i]
    data, label = [], []
    for file in possible_data:
        label.append(torch.Tensor(np.load(opt.data_dir + "/" + file))) if "label" in file else data.append(
            torch.Tensor(np.load(opt.data_dir + "/" + file)))
    assert len(data) == len(label), ".npy文件存在多个数据和label,且数量不匹配。 load函数将自动读取所有npy文件，且名字中含有label的作为label，否则为data"
    if len(data) == 0:
        logging.info("未检测到数据")
    else:
        logging.info("-" * 30)
        logging.info("load_data:")
        logging.info("检测到{}组数据".format(len(data)))
        logging.info("load data:" + str(possible_data))
        # print(possible_data)
    # logging.info("未检测到数据" if len(data) == 0 else "-" * 30 + "\nload_data:\n检测到{}组数据".format(len(data)),
    #       "\nload data:",
    #       possible_data)
    data, label = torch.cat(data, dim=0).permute(0, 3, 1, 2), torch.cat(label, dim=0).reshape(-1, 2)
    logging.info("data.shape = {}, label.shape = {}".format(data.shape, label.shape))
    return data, label


def choose_model(opt):
    model_class = eval(opt.model)
    model = model_class(in_channels=3, t_kernel_size=opt.t_kernel_size,
                        hop_size=opt.hop_size).to(device)

    if opt.accelerate:
        try:
            logging.info(
                "检测到torch版本高于2.0, 使用torch.compile加速模型, 使用torch.set_float32_matmul_precision加速fp32矩阵乘法")
            model = torch.compile(model)  # >=torch2.0才有的功能
            torch.set_float32_matmul_precision('high')  # “highest” (default), “high”, or “medium”
        except:
            pass
    return model


class BMCLossMD(torch.nn.modules.loss._Loss):
    """
    Multi-Dimension version BMC, compatible with 1-D BMC
    """

    def __init__(self, init_noise_sigma):
        super(BMCLossMD, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss_md(pred, target, noise_var)
        return loss


def bmc_loss_md(pred, target, noise_var):
    I = torch.eye(pred.shape[-1]).to(device)
    logits = MVN(pred.unsqueeze(1), noise_var * I).log_prob(target.unsqueeze(0))
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(device))
    loss = loss * (2 * noise_var).detach()
    return loss


def choose_criterion(opt):
    if "MAE" in opt.criterion:
        logging.info("使用MAE")
        criterion = torch.nn.L1Loss()  # MAE Loss
    elif opt.criterion == "MSE":
        logging.info("使用MSE")
        criterion = torch.nn.MSELoss()
    elif opt.criterion == "BMSE":
        logging.info("使用Balanced MSE")
        criterion = BMCLossMD(init_noise_sigma=2.0)
    else:
        logging.info("使用MAE")
        criterion = torch.nn.L1Loss()
    return criterion


def lr_schadule(opt, optimizer, schedule=None):
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
    if schedule == "cosine_anneal":
        logging.info("使用余弦退火学习率")
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0.0001)  # 余弦退火
    elif schedule == "reduceonplateau":
        logging.info("TODO: 使用平坦时衰减学习率")
        scheduler = None
    elif schedule == "lambda":
        logging.info("TODO: 使用lambda学习率")
        scheduler = None
    else:
        scheduler = None
    return scheduler


def shuffle(seq, label):
    assert seq.shape[0] == label.shape[0], "时间序列样本量 != label样本量"
    idx = np.array(range(label.shape[0]))
    np.random.shuffle(idx)
    seq = seq[idx, ...]
    label = label[idx, ...]
    return seq, label


# TODO: 屏蔽不必要的warning
def ignore_warning():
    warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.utils")


class dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return (self.label.shape[0])

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label


if __name__ == "__main__":

    ignore_warning()
    opt = opt()

    data, label = load_data(opt)
    data, label = shuffle(data, label)
    # random_seed(opt.random_seed)
    device = opt.device if torch.cuda.is_available() else "cpu"
    # 加载模型
    model = choose_model(opt)

    if opt.load_weight:
        logging.info("加载预训练权重: {}".format(opt.load_weight))
        state_dict = torch.load(opt.load_weight)
        model.load_state_dict(state_dict)
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"模型参数量: {total_params}, 即{total_params * 4 / (1024 ** 2)}MB")

    # 配置data_loader
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=dataset(train_data, train_label),
    #     batch_size=opt.batch_size, shuffle=True, pin_memory=True, persistent_workers=False, num_workers=opt.num_workers)
    # val_loader = torch.utils.data.DataLoader(
    #     dataset=dataset(val_data, val_label),
    #     batch_size=opt.batch_size, shuffle=False, pin_memory=True, persistent_workers=False,
    #     num_workers=opt.num_workers)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset(data[:int(opt.t_v_rate * data.shape[0])], label[:int(opt.t_v_rate * data.shape[0])]),
        batch_size=opt.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=dataset(data[int(opt.t_v_rate * data.shape[0]):], label[int(opt.t_v_rate * data.shape[0]):]),
        batch_size=opt.batch_size, shuffle=False)

    # 加载其他48个人的第二式动作
    others_data = torch.load(opt.data_dir + "/source/太极拳第二式48个视频标准化后的时间序列.torch.tensor").permute(0, 3,
                                                                                                                   1, 2)
    others_data = others_data.to(device)

    # 配置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=0.0001)  # 配置岭回归参数0.0001

    scheduler = lr_schadule(opt=opt, optimizer=optimizer, schedule=opt.lr_schedule)  # 学习率衰减策略
    criterion = choose_criterion(opt)
    criterion = criterion.to(device)
    criterion1 = torch.nn.L1Loss()
    if opt.criterion == "BMSE":
        optimizer.add_param_group({'params': criterion.parameters(), 'lr': 0.001})

    # 简单前向传播跑通和输出检查
    # model.to(device)
    # y = model(torch.randn(1,3,200,17).to(device))
    # >>> y: (tensor([[0.5562]]), tensor([[-0.1474,  0.0553,  0.0919]])

    # 训练(边验证边训练)
    make_file = opt.output + "/{}".format(datetime.datetime.now().replace(microsecond=0))
    os.makedirs(make_file)
    writer = {"train_loss": SummaryWriter(log_dir=make_file + '/train_loss'),
              "val_loss": SummaryWriter(log_dir=make_file + '/val_loss')}
    writer4 = {"lr_curve": SummaryWriter(log_dir=make_file + '/lr_curve')}
    loss_between_epochs = []  # 记录每个epoch损失值
    val_loss = []
    lr_curve = []

    best_state_dict = model.state_dict()
    best_epoch = 1
    min_loss = 1
    x_std = torch.Tensor(np.load(opt.data_dir + "/source/source_data.npy")).permute(0, 3, 1, 2)
    # start Train

    for epoch in range(1, opt.epochs + 1):
        loss_between_batchs = []
        # train
        model.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            # print("模型训练开始时")
            data, label = data.to(device), label[:, 0].unsqueeze(1).to(device)
            x_std = x_std.to(device)
            pred_label = model(data, x_std)
            # print("data.shape={},label.shape={},x_std.shape={},pred_label.shape={}".format(data.shape,label.shape,x_std.shape,pred_label.shape))

            # print("data.shape = ",data.shape) # data.shape =  torch.Size([128, 3, 200, 17])
            # print("label.shape = ",label.shape) # label.shape =  torch.Size([128, 2]),其中label[:,0]为回归结果，label[:,1]为分类结果
            # print("pred_label.shape = ",pred_label.shape)

            loss = criterion(pred_label, label)

            others_score = model(others_data[:-10], x_std)
            # print(others_score, "\n", others_score.shape)
            loss1 = criterion(others_score, torch.full((38, 1), 0.8).to(device))
            total_loss = 0.5 * loss + 0.5 * loss1

            optimizer.zero_grad()
            # loss.backward()
            total_loss.backward()

            optimizer.step()
            loss_between_batchs.append(loss.item())
        try:
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()  # 如果配置了学习率衰减策略, 则对学习率衰减
        except:
            # 如果没用衰减的话
            current_lr = optimizer.param_groups[0]['lr']
            pass
        batch_loss = np.mean(loss_between_batchs)

        loss_between_epochs.append(batch_loss)

        lr_curve.append((current_lr))

        writer["train_loss"].add_scalar("Total Loss", batch_loss, epoch)
        writer4["lr_curve"].add_scalar("Lr", current_lr, epoch)
        # print("\033[91mThis is red text\033[0m")# 彩色字体 # \033[91m代表字体颜色 \033[0m代表字体背景底色
        logging.info(
            '\033[95m● Train ● : Epoch: {:4}/{} | Lr: {:.8f} | Total Loss: {:.4f} | {}\033[0m'.format(
                epoch, opt.epochs, current_lr, batch_loss,
                datetime.datetime.now().strftime('%H:%M:%S')))

        # val
        if epoch % opt.val_per_epochs == 0:
            model.eval()
            aaa = 0
            bbb = 0
            with torch.no_grad():
                val_loss_between_batch = []
                for batch_idx, (data, label) in enumerate(val_loader):
                    data, label = data.to(device), label[:, 0].unsqueeze(1).to(device)
                    pred_label = model(data, x_std)
                    # print("eval_data.shape:",data.shape)
                    # print("eval_label.shape:", label.shape)
                    loss = criterion(pred_label, label)
                    val_loss_between_batch.append(loss.item())
                    aaa = pred_label
                    bbb = label
                print('pred_label:', aaa[:12, ].T)
                print('gt___label:', bbb[:12, ].T)
                others_score = model(others_data[-10:], x_std)
                others_mean_score = torch.mean(others_score)
                print("其他人第二式的得分:{}, mean={}".format(others_score.T, others_mean_score))
                val_loss_batch = np.mean(val_loss_between_batch)
                val_loss.append(val_loss_batch)
                writer["val_loss"].add_scalar("Total Loss", val_loss_batch, epoch)

                # print("\033[91mThis is red text\033[0m")# 彩色字体 # 91m代表字体颜色 [0m代表字体背景底色

                # 记录最优模型
                if others_mean_score > 0.5 and val_loss_batch < 0.1:
                    torch.save(best_state_dict,
                               make_file + "/stgcn_epoch={}_loss={:4f}_others_m_score={:4f}.pt".format(best_epoch,
                                                                                                       min_loss,
                                                                                                       others_mean_score))
                if val_loss_batch < min_loss:
                    best_state_dict = model.state_dict()
                    best_epoch = epoch
                    min_loss = val_loss_batch
                    logging.info(
                        '\033[92m○  val  ○ : Epoch: {:4}/{} | Lr: {:.8f} | Total Loss: {:.4f} | {} ☆\033[0m'.format(
                            epoch, opt.epochs, current_lr, val_loss_batch,
                            datetime.datetime.now().strftime('%H:%M:%S')))
                    # 保存当前最优模型
                    torch.save(best_state_dict, make_file + "/stgcn_epoch={}_loss={}.pt".format(best_epoch, min_loss))
                else:
                    logging.info(
                        '\033[92m○  val  ○ : Epoch: {:4}/{} | Lr: {:.8f} | Total Loss: {:.4f} | {}\033[0m'.format(
                            epoch, opt.epochs, current_lr, val_loss_batch,
                            datetime.datetime.now().strftime('%H:%M:%S')))
    # 关闭tensorboard流
    writer["train_loss"].close()
    writer["val_loss"].close()
    writer4["lr_curve"].close()

    # 绘制损失迭代曲线
    # TODO: 绘制各个子图的曲线
    ax = plt.subplot(111)
    plt.title('Train loss iterations')
    plt.xlabel('Epochs')
    plt.ylabel('Loss({})'.format(type(criterion).__name__))
    plt.plot(range(len(loss_between_epochs)), loss_between_epochs, 'b',
             label='Training {}'.format(type(criterion).__name__))
    plt.plot([i * opt.val_per_epochs for i in range(len(val_loss))], val_loss, 'r',
             label='validation {}'.format(type(criterion).__name__))

    ax.legend(fontsize=10, frameon=False)
    plt.savefig(
        make_file + '/{}_epochs={}_optim={}_loss_func={}_min_loss={:.2f}.png'.format(
            datetime.datetime.now().replace(microsecond=0), opt.epochs, type(optimizer).__name__,
            type(criterion).__name__,
            min_loss), dpi=300)
    plt.close()

    logging.shutdown()
