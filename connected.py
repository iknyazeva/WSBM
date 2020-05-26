import re


class UnionFind(object):
    def __init__(self, n):
        self.n = n
        self.parent = [-1] * n
        self.link = [-1] * n
        for i in range(n):
            self.parent[i] = i
            self.link[i] = i

    def find(self, i):
        # Path Compression
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def update_parents(self):
        for i in range(self.n):
            self.find(i)


    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot != yroot:
            self.parent[yroot] = xroot
            self.create_link(x, y)

    def create_link(self, x, y):
        temp = self.link[y]
        self.link[y] = self.link[x]
        self.link[x] = temp

    def print_set(self, x):
        # type: (object) -> object
        root = x
        conn_set = [x]
        while self.link[x] != root:
            x = self.link[x]
            conn_set.append(x)
        return [conn + 1 for conn in conn_set]


def read_data(filename, node_numbers=20):
    data = []
    with open(filename, 'r') as f:
        for x in range(1,node_numbers+1):
            line = next(f)
            pattern = re.search("^\[0.000, (\d.\d+)\):\s\[(\d+)\]\+\[(\d+)\]", line)
            # pattern = re.search("(\d+)", line)
            # record = [float(pattern.group(1), int(pattern.group(2),int(pattern.group(3))]
            if pattern is not None:
                record = float(pattern.group(1)), int(pattern.group(2)), int(pattern.group(3))
                data.append(record)
    return sorted(data, key = lambda x: x[0])


class ProcessConnected(object):
    def __init__(self, data, n):
        self.n = n
        self.data = data
        self._node_of_interest = []
        self._node_appearance = []
        self._compt = []

    def find_connected_component_at_level(self, node, level):

        uf = UnionFind(self.n)
        for record in self.data:
            if record[0] > level:
                break
            uf.union(record[1] - 1, record[2] - 1)
        return uf.print_set(node - 1)


    def find_first_appearance(self, node):
        """

        :param data: list of b0 intervals, where each record is tuple (level, start, end)
        :param node: interested node
        :return:  level and line_number where node first appeared
        """
        self._node_of_interest = node

        assert node <= self.n, "Current node couldn't be less then the number of nodes"

        isFind = False
        for idx, record in enumerate(self.data):
            if (record[1] == node) | (record[2] == node):
                isFind = True
                break
        if isFind:

            self._node_appearance = (idx + 1, record[0])

            return (idx + 1, record[0])
        else:
            return None

    def find_connected_comp(self, node):
        i, level = self.find_first_appearance(node)
        uf = UnionFind(self.n)
        for record in self.data[:i]:
            uf.union(record[1] - 1, record[2] - 1)
        self._compt = uf.print_set(node - 1)


if __name__ == "__main__":
    node_numbers = 20
    data = read_data('ints_20_0.4_0.3_0.3_3-1.txt', node_numbers)
    pc = ProcessConnected(data, 20)
    level = 0.25
    comps17 = pc.find_connected_component_at_level(17, level)
    comps1 = pc.find_connected_component_at_level(1, level)
    comps9 = pc.find_connected_component_at_level(1, level)


    i, level = find_first_appearance(data, node)
    # uf = UnionFind(207)
    # for record in data[:i]:
    #    uf.union(record[1]-1,record[2]-1)
    # compt = uf.print_set(node-1)
    pc.find_connected_comp(128)
    pc.find_connected_comp(124)

    print('Hi')
