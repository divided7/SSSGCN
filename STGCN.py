import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Graph_structure import Graph_17 as Graph


class SpatialGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size):
        super().__init__()
        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * s_kernel_size,
                              kernel_size=1)

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        # print("SGC变形之前：", n, kc, t, v, "  变形之后：", n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        # 对邻接矩阵进行GC，相加特征。
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()


class SpatialSeparateGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size):
        super().__init__()
        self.s_kernel_size = s_kernel_size
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=3,
                               padding=1,
                               groups=in_channels,
                               )
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels * s_kernel_size, kernel_size=1)

    def forward(self, x, A):
        x = self.conv1(x)
        x = self.conv2(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()


class STGC_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t_kernel_size, A_size, dropout=0.6):
        super().__init__()
        self.sgc = SpatialGraphConvolution(in_channels=in_channels,
                                           out_channels=out_channels,
                                           s_kernel_size=A_size[0])

        # Learnable weight matrix M 给边缘赋予权重。学习哪个边是重要的。
        self.M = nn.Parameter(torch.ones(A_size))

        self.tgc = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.SiLU(),
                                 nn.Dropout(dropout),
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (t_kernel_size, 1),
                                           (stride, 1),
                                           ((t_kernel_size - 1) // 2, 0)),
                                 # nn.Dropout(dropout),
                                 nn.BatchNorm2d(out_channels),
                                 nn.SiLU())

    def forward(self, x, A):
        x = self.tgc(self.sgc(x, A * self.M))
        return x


class SSTGC_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t_kernel_size, A_size, dropout=0.5):
        super().__init__()
        self.sgc = SpatialSeparateGraphConvolution(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   s_kernel_size=A_size[0])

        self.M = nn.Parameter(torch.ones(A_size))

        self.tgc = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (t_kernel_size, 1),  # kernel_size
                                           (stride, 1),  # stride
                                           ((t_kernel_size - 1) // 2, 0),  # padding
                                           ),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU())

    def forward(self, x, A):
        x = self.tgc(self.sgc(x, A * self.M))
        return x


class STDGC_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t_kernel_size, A_size, dropout=0.5):
        super().__init__()
        self.sgc = SpatialGraphConvolution(in_channels=in_channels,
                                           out_channels=out_channels,
                                           s_kernel_size=A_size[0])

        # Learnable weight matrix M 给边缘赋予权重。学习哪个边是重要的。
        self.M = nn.Parameter(torch.ones(A_size))

        self.tgc = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (t_kernel_size, 1),  # kernel_size
                                           (stride, 1),  # stride
                                           ((t_kernel_size - 1) // 2, 0),  # padding
                                           dilation=2,
                                           ),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU())

    def forward(self, x, A):
        x = self.tgc(self.sgc(x, A * self.M))
        return x


class SSTDGC_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t_kernel_size, A_size, dropout=0.5):
        super().__init__()
        self.sgc = SpatialSeparateGraphConvolution(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   s_kernel_size=A_size[0])

        # Learnable weight matrix
        self.M = nn.Parameter(torch.ones(A_size))

        self.tgc = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (t_kernel_size, 1),  # kernel_size
                                           (stride, 1),  # stride
                                           ((t_kernel_size - 1) // 2, 0),  # padding
                                           dilation=2,
                                           ),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU())

    def forward(self, x, A):
        x = self.tgc(self.sgc(x, A * self.M))
        return x


class ST_GCN_only_reg(nn.Module):
    def __init__(self, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])
        # STGC_blocks
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Prediction
        self.fc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        # x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        # x = self.stgc6(x, self.A)
        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        # print("x.shape:::",x.shape) # torch.Size([1, 64, 1, 1])
        score = self.fc(x)
        score = self.sigmoid(score)
        # print("score.shape:::",score.shape) # score.shape::: torch.Size([1, 1, 1, 1])
        score = score.view(score.size(0), -1)

        return score


class ST_GCN_only_cls(nn.Module):
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])
        # STGC_blocks
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Prediction
        self.cls = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        # x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        # x = self.stgc6(x, self.A)
        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        # print("x.shape:::",x.shape) # torch.Size([1, 64, 1, 1])
        cls = self.cls(x)
        # print("cls.shape:::", cls.shape)
        cls = cls.view(cls.size(0), -1)
        return cls


class ST_GCN(nn.Module):
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])
        # STGC_blocks
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Prediction
        self.fc = nn.Conv2d(64, 1, kernel_size=1)
        self.cls = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        # x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        # x = self.stgc6(x, self.A)
        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        # print("x.shape:::",x.shape) # torch.Size([1, 64, 1, 1])
        score = self.fc(x)
        score = self.sigmoid(score)
        # print("score.shape:::",score.shape) # score.shape::: torch.Size([1, 1, 1, 1])
        score = score.view(score.size(0), -1)
        cls = self.cls(x)
        # print("cls.shape:::", cls.shape)
        cls = cls.view(cls.size(0), -1)
        return score, cls


class ST_GCN_split(nn.Module):
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])
        # STGC_blocks
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Prediction
        self.fc = nn.Linear(32, 1)
        self.cls = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        # x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        # x = self.stgc6(x, self.A)
        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        # print("x.shape:::",x.shape) # torch.Size([1, 64, 1, 1])
        score = x[:, 32:, :, :].view(x[:, 32:, :, :].size(0), -1)
        # print("score.shape = ",score.shape) # [1,32]
        score = self.fc(score)
        score = self.sigmoid(score)
        # print("score.shape:::",score.shape) # score.shape::: torch.Size([1, 1, 1, 1])
        score = score.view(score.size(0), -1)
        cls = self.cls(x[:, 32:, :, :])
        # print("cls.shape:::", cls.shape)
        cls = cls.view(cls.size(0), -1)
        return score, cls


class ST_GCN_split_plus(nn.Module):
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])
        # STGC_blocks
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Prediction
        self.fc1 = nn.Linear(32 + num_classes, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(64, 1)
        self.cls = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        # x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        # x = self.stgc6(x, self.A)
        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        # print("x.shape:::",x.shape) # torch.Size([1, 64, 1, 1])

        cls = self.cls(x[:, 32:, :, :])
        cls = cls.view(cls.size(0), -1)
        # print("cls.shape:",cls.shape) # torch.Size([1, num_classes])
        score = x[:, 32:, :, :].view(x[:, 32:, :, :].size(0), -1)
        # print("score.shape",score.shape) # torch.Size([1, 32])
        score = torch.cat((score, cls), dim=1).contiguous()
        score = self.fc2(self.tanh(self.bn1(self.fc1(score))))
        score = self.sigmoid(score)
        score = score.view(score.size(0), -1)
        # print("score.shape",score.shape)
        return score, cls


class ST_GCN_W(nn.Module):
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])

        # STGC_blocks
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = STGC_block(32, 64, 1, t_kernel_size, A_size)
        self.stgc3 = STGC_block(64, 128, 1, t_kernel_size, A_size)
        self.stgc4 = STGC_block(128, 128, 2, t_kernel_size, A_size)
        self.stgc5 = STGC_block(128, 128, 1, t_kernel_size, A_size)
        self.stgc6 = STGC_block(128, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Prediction
        self.fc = nn.Conv2d(64, 1, kernel_size=1)
        self.cls = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)
        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        # print("x.shape:::",x.shape) # torch.Size([1, 64, 1, 1])
        score = self.fc(x)
        score = self.sigmoid(score)
        # print("score.shape:::",score.shape) # score.shape::: torch.Size([1, 1, 1, 1])
        score = score.view(score.size(0), -1)
        cls = self.cls(x)
        # print("cls.shape:::", cls.shape)
        cls = cls.view(cls.size(0), -1)
        return score, cls


class ST_GCN_DeepBackbone(nn.Module):
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])

        # STGC_blocks
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Prediction
        self.fc = nn.Conv2d(64, 1, kernel_size=1)
        self.cls = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)
        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        # print("x.shape:::",x.shape) # torch.Size([1, 64, 1, 1])
        score = self.fc(x)
        score = self.sigmoid(score)
        # print("score.shape:::",score.shape) # score.shape::: torch.Size([1, 1, 1, 1])
        score = score.view(score.size(0), -1)
        cls = self.cls(x)
        # print("cls.shape:::", cls.shape)
        cls = cls.view(cls.size(0), -1)
        return score, cls


class ST_GCN_L(nn.Module):
    # 替换regression head中的卷积层为Linear层
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])

        # STGC_blocks
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Prediction
        self.fc = nn.Linear(64, 1)
        self.cls = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        # x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        # x = self.stgc6(x, self.A)
        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        # print("x.shape:::",x.shape) # torch.Size([batch size, 64, 1, 1])
        score = x.view(x.size(0), -1)
        score = self.fc(score)
        score = self.sigmoid(score)
        # print("score.shape:::",score.shape) # score.shape::: torch.Size([batch size, 1])
        cls = self.cls(x)
        # print("cls.shape:::", cls.shape)
        cls = cls.view(cls.size(0), -1)
        return score, cls


class ST_GCN_MLPs(nn.Module):
    # 替换regression head中的卷积层为MLPs
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])

        # STGC_blocks
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Prediction
        self.fc = nn.Linear(64, 64)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(64, 1)
        self.cls = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        # x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        # x = self.stgc6(x, self.A)
        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        # print("x.shape:::",x.shape) # torch.Size([batch size, 64, 1, 1])
        score = x.view(x.size(0), -1)
        score = self.fc1(self.tanh(self.fc(score)))
        score = self.sigmoid(score)
        # print("score.shape:::",score.shape) # score.shape::: torch.Size([batch size, 1])
        cls = self.cls(x)
        # print("cls.shape:::", cls.shape)
        cls = cls.view(cls.size(0), -1)
        return score, cls


class ST_GCN_DeepMLPs(nn.Module):
    # 替换regression head中的卷积层为深层MLPs
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])

        # STGC_blocks
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Prediction
        self.fc = nn.Linear(64, 64)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.cls = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        # x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        # x = self.stgc6(x, self.A)
        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        # print("x.shape:::",x.shape) # torch.Size([batch size, 64, 1, 1])
        score = x.view(x.size(0), -1)
        score = self.fc3(self.tanh(self.fc2(self.tanh(self.fc1(self.tanh(self.fc(score)))))))
        score = self.sigmoid(score)
        # print("score.shape:::",score.shape) # score.shape::: torch.Size([batch size, 1])
        cls = self.cls(x)
        # print("cls.shape:::", cls.shape)
        cls = cls.view(cls.size(0), -1)
        return score, cls


class ST_GCN_DeepBackbone_MLPs(nn.Module):
    # 替换regression head中的卷积层为深层MLPs
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])

        # STGC_blocks
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Prediction
        self.fc = nn.Linear(64, 64)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.cls = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)
        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        # print("x.shape:::",x.shape) # torch.Size([batch size, 64, 1, 1])
        score = x.view(x.size(0), -1)
        score = self.fc3(self.tanh(self.fc2(self.tanh(self.fc1(self.tanh(self.fc(score)))))))
        score = self.sigmoid(score)
        # print("score.shape:::",score.shape) # score.shape::: torch.Size([batch size, 1])
        cls = self.cls(x)
        # print("cls.shape:::", cls.shape)
        cls = cls.view(cls.size(0), -1)
        return score, cls


class SST_GCN(nn.Module):
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])

        # STGC_blocks
        self.stgc1 = SSTGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = SSTGC_block(32, 64, 1, t_kernel_size, A_size)
        self.stgc3 = SSTGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc4 = SSTGC_block(64, 128, 2, t_kernel_size, A_size)
        self.stgc5 = SSTGC_block(128, 128, 1, t_kernel_size, A_size)
        self.stgc6 = SSTGC_block(128, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Prediction
        self.fc = nn.Linear(64, 64)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.cls = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)
        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        # print("x.shape:::",x.shape) # torch.Size([batch size, 64, 1, 1])
        score = x.view(x.size(0), -1)
        score = self.fc3(self.tanh(self.fc2(self.tanh(self.fc1(self.tanh(self.fc(score)))))))
        score = self.sigmoid(score)
        # print("score.shape:::",score.shape) # score.shape::: torch.Size([batch size, 1])
        cls = self.cls(x)
        # print("cls.shape:::", cls.shape)
        cls = cls.view(cls.size(0), -1)
        return score, cls


class STD_GCN(nn.Module):
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])

        # STGC_blocks
        self.stgc1 = STDGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = STDGC_block(32, 64, 1, t_kernel_size, A_size)
        self.stgc3 = STDGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc4 = STDGC_block(64, 128, 2, t_kernel_size, A_size)
        self.stgc5 = STDGC_block(128, 128, 1, t_kernel_size, A_size)
        self.stgc6 = STDGC_block(128, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Prediction
        self.fc = nn.Linear(64, 64)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.cls = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)
        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        # print("x.shape:::",x.shape) # torch.Size([batch size, 64, 1, 1])
        score = x.view(x.size(0), -1)
        score = self.fc3(self.tanh(self.fc2(self.tanh(self.fc1(self.tanh(self.fc(score)))))))
        score = self.sigmoid(score)
        # print("score.shape:::",score.shape) # score.shape::: torch.Size([batch size, 1])
        cls = self.cls(x)
        # print("cls.shape:::", cls.shape)
        cls = cls.view(cls.size(0), -1)
        return score, cls


class SSTD_GCN(nn.Module):
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])

        # STGC_blocks
        self.stgc1 = SSTDGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = SSTDGC_block(32, 64, 1, t_kernel_size, A_size)
        self.stgc3 = SSTDGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc4 = SSTDGC_block(64, 128, 2, t_kernel_size, A_size)
        self.stgc5 = SSTDGC_block(128, 128, 1, t_kernel_size, A_size)
        self.stgc6 = SSTDGC_block(128, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Prediction
        self.fc = nn.Linear(64, 64)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.cls = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)
        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        # print("x.shape:::",x.shape) # torch.Size([batch size, 64, 1, 1])
        score = x.view(x.size(0), -1)
        score = self.fc3(self.tanh(self.fc2(self.tanh(self.fc1(self.tanh(self.fc(score)))))))
        score = self.sigmoid(score)
        # print("score.shape:::",score.shape) # score.shape::: torch.Size([batch size, 1])
        cls = self.cls(x)
        # print("cls.shape:::", cls.shape)
        cls = cls.view(cls.size(0), -1)
        return score, cls


class ST_GCN_multi_reg(nn.Module):
    def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
        super().__init__()
        self.reg_model_list = []
        # reg model:
        for i in range(num_classes):
            self.reg_model_list.append(ST_GCN_only_reg(in_channels, t_kernel_size, hop_size))
        # cls model:
        self.cls_model = ST_GCN_only_cls(num_classes, in_channels, t_kernel_size, hop_size)

    def forward(self, x):
        cls = self.cls_model(x)
        classes = torch.argmax(cls, dim=1)
        print("classes:", classes)
        # print("torch.max(cls.data, 1)[1].item():",torch.max(cls.data, 1)[1].item())
        # reg = self.reg_model_list[torch.max(cls.data, 1)[1].item()](x)
        reg = self.reg_model_list[0](x)
        return cls, reg


class Siamese_SSTD_GCN_MLPs_only_reg(nn.Module):
    # 替换regression head中的卷积层为深层MLPs
    def __init__(self, in_channels, t_kernel_size, hop_size):
        super().__init__()
        # graph制作
        graph = Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()
        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])
        self.alpha = torch.nn.Parameter(torch.rand(1))
        self.beta = torch.nn.Parameter(torch.rand(1))
        # STGC_blocks
        self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size)
        self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size)
        self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size)
        self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward_once(self, x):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)
        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)

        score_feature_seq = x.view(x.size(0), -1)
        return score_feature_seq

    def relu1(self, x):
        return torch.clamp(x, min=0, max=1)

    def CoLU(self, x):
        x = torch.clamp(x, min=0, max=1)
        return (1 - torch.cos(torch.pi * x)) / 2

    def forward(self, x1, std_x, act=None):
        """
        :param x1: 用户运动的动作
        :param std_x: 标准的满分动作
        :return:
        """
        y1 = self.forward_once(x1)
        std_y = self.forward_once(std_x)
        #
        cos_sim = nn.functional.cosine_similarity(y1, std_y, dim=1, eps=1e-6)  # 这里将数据由[bs,1]变形为[bs]
        euclidean_dist = nn.functional.pairwise_distance(y1, std_y, p=2)  # p范数 这里将数据由[bs,1]变形为[bs]
        # alpha = 0  # 角度权重
        # beta = 1  # 距离权重
        score = self.alpha * cos_sim + self.beta * (1 / (1 + euclidean_dist))  # 使用可学习的参数
        # score = alpha * cos_sim + beta * (1 / (1 + euclidean_dist)) # 使用不可学习的参数
        score = score.view(-1, 1)
        score = score.contiguous()
        if act == "relu1":
            score = self.relu1(score)
        elif act == "colu":
            score = self.CoLU(score)
        else:
            score = self.sigmoid(score)
        # # additional : score + MLPs
        # return score
        return score


def 算子拆解():
    x = torch.randn([1, 2, 32, 128, 17])
    A = torch.randn([2, 17, 17])
    einsum_result = torch.einsum('nkctv,kvw->nctw', (x, A))
    print(einsum_result[0][0][0][:5])
    n_dim, k_dim, c_dim, t_dim, v_dim = x.size()
    _, _, w_dim = A.size()

    a_reshaped = x.permute(0, 2, 3, 1, 4).reshape(n_dim, c_dim, t_dim, k_dim * v_dim)
    b_reshaped = A.reshape(k_dim * v_dim, w_dim)
    result = torch.matmul(a_reshaped, b_reshaped)
    result = result.reshape(n_dim, c_dim, t_dim, w_dim)
    print(result.shape)
    print(result[0][0][0][:5])
    return 0
