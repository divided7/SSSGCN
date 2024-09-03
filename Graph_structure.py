import numpy as np


class Graph_17():
    """
    Human3.6m数据集,17个关键点
        0: 'root (pelvis)',
        1: 'right_hip',
        2: 'right_knee',
        3: 'right_foot',
        4: 'left_hip',
        5: 'left_knee',
        6: 'left_foot',
        7: 'spine',
        8: 'thorax',
        9: 'neck_base',
        10: 'head',
        11: 'left_shoulder',
        12: 'left_elbow',
        13: 'left_wrist',
        14: 'right_shoulder',
        15: 'right_elbow',
        16: 'right_wrist'
    """

    def __init__(self, hop_size):
        # 规定边缘排列，作为集合{{起点,终点},{起点,终点},{起点,终点……}这样规定一个边为元素。
        self.get_edge()

        # hop: hop数连接几个分离的关节
        # 例如hop=2的话，手腕不仅和胳膊肘连在一起，还和肩膀连在一起。
        self.hop_size = hop_size
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, hop_size=hop_size)

        # 创建一个相邻矩阵。在这里，根据hop数创建一个相邻矩阵。
        # hop是2的时候，0hop, 1hop, 2hop这三个相邻的矩阵被创建。
        # 论文中提出了多种生成方法。这次使用了简单易懂的方法。
        self.get_adjacency()

    def __str__(self):
        return self.A

    def get_edge(self):
        self.num_node = 17
        self_link = [(i, i) for i in range(self.num_node)]  # Loop
        neighbor_link = [(0, 7), (7, 8), (8, 9), (9, 10),
                         (8, 11), (11, 12), (12, 13),
                         (8, 14), (14, 15), (15, 16),
                         (0, 1), (1, 2), (2, 3),
                         (0, 4), (4, 5), (5, 6)]

        self.edge = self_link + neighbor_link

    def get_adjacency(self):
        valid_hop = range(0, self.hop_size + 1, 1)
        adjacency = np.zeros((self.num_node, self.num_node))  # 邻接矩阵
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        self.A = A

    def get_hop_distance(self, num_node, edge, hop_size):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop_size + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(hop_size, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        DAD = np.dot(A, Dn)
        return DAD


class Graph_25():
    """


    """

    def __init__(self, hop_size):
        # 规定边缘排列，作为集合{{起点,终点},{起点,终点},{起点,终点……}这样规定一个边为元素。
        self.get_edge()

        # hop: hop数连接几个分离的关节
        # 例如hop=2的话，手腕不仅和胳膊肘连在一起，还和肩膀连在一起。
        self.hop_size = hop_size
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, hop_size=hop_size)

        # 创建一个相邻矩阵。在这里，根据hop数创建一个相邻矩阵。
        # hop是2的时候，0hop, 1hop, 2hop这三个相邻的矩阵被创建。
        # 论文中提出了多种生成方法。这次使用了简单易懂的方法。
        self.get_adjacency()

    def __str__(self):
        return self.A

    def get_edge(self):
        self.num_node = 25
        self_link = [(i, i) for i in range(self.num_node)]  # Loop
        neighbor_base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                         (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                         (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                         (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                         (22, 23), (23, 8), (24, 25), (25, 12)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
        self.edge = self_link + neighbor_link

    def get_adjacency(self):
        valid_hop = range(0, self.hop_size + 1, 1)
        adjacency = np.zeros((self.num_node, self.num_node))  # 邻接矩阵
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        self.A = A

    def get_hop_distance(self, num_node, edge, hop_size):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop_size + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(hop_size, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        DAD = np.dot(A, Dn)
        return DAD
