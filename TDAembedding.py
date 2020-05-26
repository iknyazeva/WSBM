import numpy as np
import re
import pandas as pd
from connected import UnionFind


class TDAembedding:
    def __init__(self, filepath, num_nodes):
        self.filepath = filepath
        self.num_nodes = num_nodes
        self.h0embedding = []
        self.h1embedding = []

    def read_intervals(self, h0=False, h1=True):
        # sorted list
        intervals = {'h0': [], 'h1': []}

        section = False
        with open(self.filepath) as fp:
            line = fp.readline()
            while line:
                if ("Dimension: 0" in line) and h0:
                    section = -1
                    line = fp.readline()
                if ("Dimension: 1" in line):
                    if h1:
                        section = 1
                        line = fp.readline()
                    else:
                        section = 2
                        line = False

                elif "Dimension: 2" in line:
                    section = 2
                    line = False
                else:
                    line = fp.readline()

                if section == -1:
                    res = read_h0(line)
                    if res is not None:
                        intervals['h0'].append(res)
                elif section == 1:
                    res = read_h1(line)
                    if res is not None:
                        intervals['h1'].append(res)
            intervals['h1'] = sorted(intervals['h1'])
            intervals['h0'] = sorted(intervals['h0'])

        return intervals

    def create_h1_embedding(self, intervals = None):
        """

        """
        if intervals is None:
            intervals = self.read_intervals(h0=False, h1=True)
        cycle_vec, columns = create_cycle_mapping(intervals['h1'], self.num_nodes)
        df = pd.DataFrame((cycle_vec > 0).astype('int'), columns=columns)
        df['num_ints'] = np.sum(cycle_vec>0, axis=1)
        df['mean_int'] = np.mean(cycle_vec, axis=1)
        self.h1embedding = df
        return df

    def create_h0_embedding(self, intervals = None):
        """

        :return:
        """
        if intervals is None:
            intervals = self.read_intervals(h0=True, h1=False)
        conn_vec, columns = create_conn_mapping(intervals['h0'], self.num_nodes)
        conn_emb = np.zeros_like(conn_vec)
        for i in range(len(intervals['h0'])):
            for j in range(self.num_nodes):
                if conn_emb[j, i] == 0:
                    idxs = np.where(conn_vec[:, i] == conn_vec[j, i])[0]
                    conn_emb[idxs, i] = len(idxs)
        df = pd.DataFrame(conn_emb, columns=columns)
        self.h0embedding = df
        return df





def create_cycle_mapping(intervals, num_nodes = 20):
    """

    :param intervals: intervals['h1] in format (start,end), [cycle members]
    :param num_nodes: number of nodes in network
    :return:
    """
    cycle_vec = np.zeros((num_nodes, len(intervals)))
    columns = []
    for idx, interval in enumerate(intervals):
        cycle_vec[interval[1], idx] = interval[0][1] - interval[0][0]
        columns.append(f"{interval[0][0] :.2f}-{interval[0][1] :.2f}")
    return cycle_vec, columns


def create_conn_mapping(intervals, num_nodes = 20):
    """

    :param intervals: intervals['h1] in format (start,end), [cycle members]
    :param num_nodes: number of nodes in network
    :return:
    """
    conn_vec = np.zeros((num_nodes, len(intervals))).astype('int')
    columns = []
    uf = UnionFind(num_nodes)
    for idx, interval in enumerate(intervals):
        uf.union(interval[1][0] - 1, interval[1][1] - 1)
        uf.update_parents()
        conn_vec[:, idx] = np.array(uf.parent).astype('int')
        columns.append(f"{interval[0] :.3f}")
    return conn_vec, columns


def read_h0(line):
    result = re.findall('[\d\.]+', line)
    if len(result) == 4:
        level = float(result[1])
        members = [int(result[2])-1, int(result[3])-1]
    else:
        return None
    return level, members


def read_h1(line):
    """

    :param line: intervals with representative in format [start, end ): [node1, node2],....
    :return: tuple with interval and node list
    """

    # result = re.findall('[\(\[][\d\.]+,\s?[\d\.]+[\]\)]', line)#numbers with bracers
    result = re.findall('[\d\.]+', line)  # extract numbers only
    # result = re.findall('[\d\.]+,\s?[\d\.]+', line)#tuples with spaces
    # result = re.search('\[(\d+),(\d+)\]', line)
    if len(result) > 3:
        interval = (float(result[0]), float(result[1]))
        members = list(set([int(node)-1 for node in result[2:]]))
    else:
        return None

    return interval, members


if __name__ == "__main__":
    # line = '[0.122, 0.144): [18,20]+[14,17]+[17,20]+[14,18]'
    line  = '[0.000, 0.306): [20]+[12]'
    res = read_h0(line)
    #line = "Dimension: 1"
    #h1line = read_h1(line)
    filepath = 'ints_20_0.4_0.3_0.3_3-1.txt'
    embd = TDAembedding(filepath)
    #embd.create_h1_embedding()
    df = embd.create_h0_embedding()
    intervals = embd.read_intervals(h0=True, h1=False)
    conn_vec, columns = create_conn_mapping(intervals['h0'], num_nodes=20)

    #cycle_vec, columns = create_cycle_mapping(intervals['h1'], num_nodes=20)
   # (cycle_vec > 0).astype('int')
    print('Cool')
