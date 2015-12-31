import numpy as np

class NN():
    """
        struct => (2, 1)
        number of units in each layer
    """
    def __init__(self, struct, cost, sigma):
        self.N = len(struct)
        self.struct = struct

        self.b = [np.ones(shape=(1,x), dtype=np.float64) for x in struct[1:]]
        self.w = [np.random.randn(n, m) / np.sqrt(n) for n,m in zip(struct, struct[1:])]

        self.cost = cost
        self.sigma = sigma

    def ff(self, x):
        self.a = [x]
        self.z = []
        for w, b in zip(self.w, self.b):
            x = np.dot(x, w) + b
            self.z.append(x)
            x = self.sigma.f(x)
            self.a.append(x)
        return x

    def grad(self, x, y):
        self.ff(x)
        self.d = [0 for _ in range(self.N - 1)]
        self.d[-1] = self.cost.df(y, self.a[-1]) * self.sigma.df(self.z[-1])
        for l in range(2, self.N):
            self.d[-l] = np.dot(self.d[-l + 1], self.w[-l + 1].T) * self.sigma.df(self.z[-l])

        wgrad = []
        bgrad = []
        for i in range(self.N - 1):
            # summing all errors
            # They are already averaged in cost function
            bgrad.append(np.sum(self.d[i], axis = 0))
            wgrad.append(np.dot(self.a[i].T,self.d[i]))

        return (wgrad, bgrad)

    def train(self, x, y, minibatch = 10, learn = 0.001):
        assert x.shape[1] == self.struct[0]
        assert y.shape[1] == self.struct[-1]
        assert x.shape[0] == y.shape[0]

        sz = self.struct[0]
        joinedData = np.hstack((x, y))
        self.trainJoinedData(joinedData, sz, minibatch=minibatch, learn=learn)

    def trainJoinedData(self, data, separatingColumn, minibatch, learn):
        np.random.shuffle(data)
        sz = separatingColumn
        for batch in np.split(data, minibatch):
            (gw, gb) = self.grad(batch[:, :sz], batch[:, sz:])
            for w, dw in zip(self.w, gw):
                w -= learn * dw

            for b, db in zip(self.b, gb):
                b -= learn * db


    def avgCost(self, x, y):
        return self.cost.f(y, self.ff(x))

    def ngrad(self, x, y):
        eps = 1e-4
        wgrad = list(map(np.zeros_like, self.w))
        bgrad = list(map(np.zeros_like, self.b))

        for i in range(len(self.w)):
            rr, cc = self.w[i].shape
            for r in range(rr):
                for c in range(cc):
                    o = self.w[i][r,c]
                    self.w[i][r,c] = o + eps
                    uc = self.avgCost(x, y)
                    self.w[i][r,c] = o - eps
                    lc = self.avgCost(x, y)
                    self.w[i][r,c] = o

                    wgrad[i][r,c] = (uc - lc) / (2*eps)

        for i in range(len(self.b)):
            rr, cc = self.b[i].shape
            for r in range(rr):
                for c in range(cc):
                    o = self.b[i][r,c]
                    self.b[i][r,c] = o + eps
                    uc = self.avgCost(x, y)

                    self.b[i][r,c] = o - eps
                    lc = self.avgCost(x, y)
                    self.b[i][r,c] = o

                    bgrad[i][r,c] = (uc - lc) / (2*eps)

        return (wgrad, bgrad)
