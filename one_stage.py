# -*- coding: utf-8 -*-
import argparse
import datetime
import os.path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
# from STGCN import *
from STGCN import ST_GCN_only_reg, ST_GCN_only_cls, ST_GCN, ST_GCN_split, ST_GCN_split_plus, ST_GCN_W, \
    ST_GCN_DeepBackbone
from STGCN import ST_GCN_L, ST_GCN_MLPs, ST_GCN_DeepMLPs, ST_GCN_DeepBackbone_MLPs
from STGCN import SST_GCN, STD_GCN, SSTD_GCN, ST_GCN_multi_reg, Siamese_SSTD_GCN_MLPs_only_reg
from torch.utils.tensorboard import SummaryWriter


def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../数据增强/输出文件/新3d太极拳训练集和验证集npy",
                        help="npy数据文件的路径")
    parser.add_argument("--model", type=str, default="ST_GCN_multi_reg",
                        help="ST_GCN, ST_GCN_W, ST_GCN_L, ST_GCN_MLPs, ST_GCN_DeepMLPs, ST_GCN_DeepBackbone")
    parser.add_argument("--output", type=str, default="训练日志", help="训练日志存放路径")
    parser.add_argument('--device', type=str, default="cuda:0", help="cpu, cuda:0, cuda:1 ...")
    parser.add_argument('--random_seed', type=float, default=3407, help='随机数种子')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--val_per_epochs', default=5, type=int, help='训练时每隔多少epochs验证一次')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs(iteration)')
    parser.add_argument('--lr_schedule', default=None, type=str,
                        help='学习率衰减策略: cosine_anneal, reduceonplateau or lambda')
    parser.add_argument('--classes', default=24, type=int, help='动作分类数')
    parser.add_argument('--mae_cel_rate', nargs='+', type=float, default=[1, 1],
                        help='MAE和CEL的回归和分类权重比值,默认为1:1, 即1 1')
    parser.add_argument('--hop_size', default=4, type=int, help='关键点邻接距离')
    parser.add_argument('--t_kernel_size', default=12, type=int, help='时域滑动窗口')
    parser.add_argument('--num_workers', default=16, type=int, help='多线程, default = 16')
    parser.add_argument('--accelerate', default=False, type=bool, help='pytorch2.0 accurate')
    args = parser.parse_args()
    print("*" * 50 + "\nParsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("*" * 50)
    return args


def random_seed(seed):
    print("启用torch.manual_seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(opt):
    train_data = torch.Tensor(np.load(opt.data_dir + "/train_inputs.npy")).permute(0, 3, 1, 2)
    train_label = torch.Tensor(np.load(opt.data_dir + "/train_labels.npy"))
    val_data = torch.Tensor(np.load(opt.data_dir + "/val_input.npy")).permute(0, 3, 1, 2)
    val_label = torch.Tensor(np.load(opt.data_dir + "/val_label.npy"))
    print("train_data.shape = {}, train_label.shape = {}".format(train_data.shape, train_label.shape))
    print("val_data.shape = {}, val_label.shape = {}".format(val_data.shape, val_label.shape))

    return train_data, train_label, val_data, val_label


def choose_model(opt):
    # if opt.model == "ST_GCN":
    #     model = ST_GCN(num_classes=opt.classes, in_channels=3, t_kernel_size=opt.t_kernel_size,
    #                    hop_size=opt.hop_size).to(device)
    # elif opt.model == "ST_GCN_L":
    #     model = ST_GCN_L(num_classes=opt.classes, in_channels=3, t_kernel_size=opt.t_kernel_size,
    #                      hop_size=opt.hop_size).to(device)
    # elif opt.model == "ST_GCN_MLPs":
    #     model = ST_GCN_MLPs(num_classes=opt.classes, in_channels=3, t_kernel_size=opt.t_kernel_size,
    #                         hop_size=opt.hop_size).to(device)
    # elif opt.model == "ST_GCN_DeepMLPs":
    #     model = ST_GCN_DeepMLPs(num_classes=opt.classes, in_channels=3, t_kernel_size=opt.t_kernel_size,
    #                             hop_size=opt.hop_size).to(device)
    # elif opt.model == "ST_GCN_DeepBackbone":
    #     model = ST_GCN_DeepBackbone(num_classes=opt.classes, in_channels=3, t_kernel_size=opt.t_kernel_size,
    #                                 hop_size=opt.hop_size).to(device)
    # elif opt.model == "ST_GCN_DeepBackbone_MLPs":
    #     model = ST_GCN_DeepBackbone_MLPs(num_classes=opt.classes, in_channels=3, t_kernel_size=opt.t_kernel_size,
    #                                      hop_size=opt.hop_size).to(device)

    # elif opt.model == "ST_GCN_W":
    #     model = ST_GCN_W(num_classes=opt.classes, in_channels=3, t_kernel_size=opt.t_kernel_size,
    #                      hop_size=opt.hop_size).to(device)
    # else:
    #     print("指定模型有误,强制将模型配置为ST_GCN")
    #     model = ST_GCN(num_classes=opt.classes, in_channels=3, t_kernel_size=opt.t_kernel_size,
    #                    hop_size=opt.hop_size).to(device)
    model_class = eval(opt.model)
    model = model_class(num_classes=opt.classes, in_channels=3, t_kernel_size=opt.t_kernel_size,
                        hop_size=opt.hop_size).to(device)

    if opt.accelerate:
        try:
            print(
                "检测到torch版本高于2.0, 使用torch.compile加速模型, 使用torch.set_float32_matmul_precision加速fp32矩阵乘法")
            model = torch.compile(model)  # >=torch2.0才有的功能
            torch.set_float32_matmul_precision('high')  # “highest” (default), “high”, or “medium”
        except:
            pass
    return model


def lr_schadule(opt, optimizer, schedule=None):
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
    if schedule == "cosine_anneal":
        print("使用余弦退火学习率")
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0.0001)  # 余弦退火
    elif schedule == "reduceonplateau":
        print("TODO: 使用平坦时衰减学习率")
        scheduler = None
    elif schedule == "lambda":
        print("TODO: 使用lambda学习率")
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
    train_data, train_label, val_data, val_label = load_data(opt)
    train_data, train_label = shuffle(train_data, train_label)
    val_data, val_label = shuffle(val_data, val_label)
    # random_seed(opt.random_seed)
    device = opt.device if torch.cuda.is_available() else "cpu"
    # 加载模型
    model = choose_model(opt)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params}, 即{total_params * 4 / (1024 ** 2)}MB")

    # 配置data_loader
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset(train_data, train_label),
        batch_size=opt.batch_size, shuffle=True, pin_memory=True, persistent_workers=False, num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(
        dataset=dataset(val_data, val_label),
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, persistent_workers=False,
        num_workers=opt.num_workers)

    # 配置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=0.0001)  # 配置岭回归参数0.0001
    MAE_Loss = torch.nn.L1Loss()  # MAE Loss
    CE_Loss = torch.nn.CrossEntropyLoss()  # 交叉熵
    scheduler = lr_schadule(opt=opt, optimizer=optimizer, schedule=opt.lr_schedule)  # 学习率衰减策略


    def criterion(pred_y, y):
        """
        计算两个损失
        :param pred_y: 预测的y.包含回归和分类结果
        :param y: 真实的y.包含回归和分类结果
        :return: 返回MAE和交叉熵损失
        """
        pred_score, score = pred_y[0], y[:, 0].unsqueeze(1)
        mae = MAE_Loss(pred_score, score)
        # print("MAE: ", mae)
        pred_cls, cls = pred_y[1], y[:, 1]
        cel = CE_Loss(pred_cls, cls.long())  # 由于这里的y既包含了float的回归数据，又包含了int的类别数据 分不开只好这样了
        # print("CEL: ", cel)
        return mae, cel


    # 简单前向传播跑通和输出检查
    # model.to(device)
    # y = model(torch.randn(1,3,200,17).to(device))
    # >>> y: (tensor([[0.5562]]), tensor([[-0.1474,  0.0553,  0.0919]])

    # 训练(边验证边训练)
    make_file = opt.output + "/{}".format(datetime.datetime.now().replace(microsecond=0))
    os.makedirs(make_file)
    writer = {"train_loss": SummaryWriter(log_dir=make_file + '/train_loss'),
              "val_loss": SummaryWriter(log_dir=make_file + '/val_loss')}
    writer1 = {"train_mae": SummaryWriter(log_dir=make_file + '/train_mae'),
               "val_mae": SummaryWriter(log_dir=make_file + '/val_mae')}
    writer2 = {"train_cel": SummaryWriter(log_dir=make_file + '/train_cel'),
               "val_cel": SummaryWriter(log_dir=make_file + '/val_cel')}
    writer3 = {"train_acc": SummaryWriter(log_dir=make_file + '/train_acc'),
               "val_acc": SummaryWriter(log_dir=make_file + '/val_acc')}
    writer4 = {"lr_curve": SummaryWriter(log_dir=make_file + '/lr_curve')}
    loss_between_epochs = []  # 记录每个epoch损失值
    val_loss = []
    mae_between_epochs = []
    val_mae = []
    cel_between_epochs = []
    val_cel = []
    lr_curve = []

    best_state_dict = model.state_dict()
    best_epoch = 1
    min_loss = 1

    # start Train
    for epoch in range(1, opt.epochs + 1):
        correct = 0
        loss_between_batchs = []
        mae_between_batchs = []
        cel_between_batchs = []
        # train
        model.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            # print("模型训练开始时")
            # print("data.shape = ",data.shape) # data.shape =  torch.Size([128, 3, 200, 17])
            # print("label.shape = ",label.shape) # label.shape =  torch.Size([128, 2]),其中label[:,0]为回归结果，label[:,1]为分类结果
            data, label = data.to(device), label.to(device)
            pred_label = model(data)
            # print("pred_label:")
            # print(pred_label[0].shape) # torch.Size([128, 1])
            # print(pred_label[1].shape) # torch.Size([128, 3])
            # print("label:")
            # print(label[:,0].shape) # torch.Size([128])
            # print(label[:,1].shape) # torch.Size([128]) 需要变成[128,1]

            mae, cel = criterion(pred_label, label)

            w_mae = opt.mae_cel_rate[0] / (opt.mae_cel_rate[0] + opt.mae_cel_rate[1])
            w_cel = opt.mae_cel_rate[1] / (opt.mae_cel_rate[0] + opt.mae_cel_rate[1])
            loss = w_mae * mae + w_cel * cel  # 总损失函数 = w1 *回归损失 + w2 * 分类损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("pred_label【1】.shape",pred_label[1].shape)
            _, predict = torch.max(pred_label[1].data, 1)
            # print("predict",predict)
            # print("label",label[:,1])
            correct += (predict == label[:, 1]).sum().item()

            loss_between_batchs.append(loss.item())
            mae_between_batchs.append(mae.item())
            cel_between_batchs.append(cel.item())
        try:
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()  # 如果配置了学习率衰减策略, 则对学习率衰减
        except:
            # 如果没用衰减的话
            current_lr = optimizer.param_groups[0]['lr']
            pass
        batch_loss = np.mean(loss_between_batchs)
        batch_mae = np.mean(mae_between_batchs)
        batch_cel = np.mean(cel_between_batchs)
        loss_between_epochs.append(batch_loss)
        mae_between_epochs.append(batch_mae)
        cel_between_epochs.append(batch_cel)
        lr_curve.append((current_lr))
        batch_acc = 100. * correct / len(train_loader.dataset)
        writer["train_loss"].add_scalar("Total Loss", batch_loss, epoch)
        writer1["train_mae"].add_scalar("MAE", batch_mae, epoch)
        writer2["train_cel"].add_scalar("CEL", batch_cel, epoch)
        writer3["train_acc"].add_scalar("Acc", batch_acc, epoch)
        writer4["lr_curve"].add_scalar("Lr", current_lr, epoch)
        # print("\033[91mThis is red text\033[0m")# 彩色字体 # \033[91m代表字体颜色 \033[0m代表字体背景底色
        print(
            '\033[95m● Train ● : Epoch: {:4}/{} | Lr: {:.8f} | Total Loss: {:.4f} | MAE Loss: {:.4f} | CE Loss: {:.4f} | Cls Acc: {:.2f}% |{}\033[0m'.format(
                epoch, opt.epochs, current_lr, batch_loss, batch_mae, batch_cel, batch_acc,
                datetime.datetime.now().strftime('%H:%M:%S')))

        # val
        if epoch % opt.val_per_epochs == 0:
            model.eval()
            with torch.no_grad():
                val_correct = 0
                val_loss_between_batch = []
                val_mae_between_batch = []
                val_cel_between_batch = []
                for batch_idx, (data, label) in enumerate(val_loader):
                    data, label = data.to(device), label.to(device)
                    pred_label = model(data)
                    mae, cel = criterion(pred_label, label)
                    w_mae = opt.mae_cel_rate[0] / (opt.mae_cel_rate[0] + opt.mae_cel_rate[1])
                    w_cel = opt.mae_cel_rate[1] / (opt.mae_cel_rate[0] + opt.mae_cel_rate[1])
                    loss = w_mae * mae + w_cel * cel  # 总损失函数 = w1 *回归损失 + w2 * 分类损失
                    val_loss_between_batch.append(loss.item())
                    val_mae_between_batch.append(mae.item())
                    val_cel_between_batch.append(cel.item())
                    _, predict = torch.max(pred_label[1].data, 1)
                    print("ground truth:", label[:, 1])
                    print("predict class:", predict)
                    val_correct += (predict == label[:, 1]).sum().item()
                val_loss_batch = np.mean(val_loss_between_batch)
                val_mae_batch = np.mean(val_mae_between_batch)
                val_cel_batch = np.mean(val_cel_between_batch)
                val_loss.append(val_loss_batch)
                val_mae.append(val_mae_batch)
                val_cel.append(val_cel_batch)
                val_acc_batch = 100. * val_correct / len(val_loader.dataset)
                writer["val_loss"].add_scalar("Total Loss", val_loss_batch, epoch)
                writer1["val_mae"].add_scalar("MAE", val_mae_batch, epoch)
                writer2["val_cel"].add_scalar("CEL", val_cel_batch, epoch)
                writer3["val_acc"].add_scalar("Acc", val_acc_batch, epoch)
                # print("\033[91mThis is red text\033[0m")# 彩色字体 # 91m代表字体颜色 [0m代表字体背景底色

                # 记录最优模型
                if val_loss_batch < min_loss:
                    best_state_dict = model.state_dict()
                    best_epoch = epoch
                    min_loss = val_loss_batch
                    print(
                        '\033[92m○  val  ○ : Epoch: {:4}/{} | Lr: {:.8f} | Total Loss: {:.4f} | MAE Loss: {:.4f} | CE Loss: {:.4f} | Cls Acc: {:.2f}% |{} ☆\033[0m'.format(
                            epoch, opt.epochs, current_lr, val_loss_batch, val_mae_batch, val_cel_batch,
                            val_acc_batch,
                            datetime.datetime.now().strftime('%H:%M:%S')))
                    # 保存当前最优模型
                    torch.save(best_state_dict, make_file + "/stgcn_epoch={}_loss={}.pt".format(best_epoch, min_loss))
                else:
                    print(
                        '\033[92m○  val  ○ : Epoch: {:4}/{} | Lr: {:.8f} | Total Loss: {:.4f} | MAE Loss: {:.4f} | CE Loss: {:.4f} | Cls Acc: {:.2f}% |{}\033[0m'.format(
                            epoch, opt.epochs, current_lr, val_loss_batch, val_mae_batch, val_cel_batch,
                            val_acc_batch,
                            datetime.datetime.now().strftime('%H:%M:%S')))
    # 关闭tensorboard流
    writer["train_loss"].close()
    writer["val_loss"].close()
    writer1["train_mae"].close()
    writer1["val_mae"].close()
    writer2["train_cel"].close()
    writer2["val_cel"].close()
    writer3["train_acc"].close()
    writer3["val_acc"].close()
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
