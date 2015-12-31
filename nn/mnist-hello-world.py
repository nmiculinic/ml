from models import *
from utilityFunctions import *
from pylab import *
from sklearn.datasets import fetch_mldata
from sklearn import preprocessing

mnist = fetch_mldata('MNIST original')

mnist.data.shape
mnist.target.shape
np.unique(mnist.target)
n = mnist.data.shape[1]

lb = preprocessing.LabelBinarizer()
lb.fit(np.arange(10))

ds = np.concatenate((mnist.data, lb.transform(mnist.target)), axis = 1)

np.random.shuffle(ds)

xy_train = ds[:5000]
xy_vc = ds[50000:60000]
xy_test = ds[60000:]

x_cv = xy_vc[:, :n]
y_cv = xy_vc[:, n:]
xy_vc.shape, x_cv.shape, y_cv.shape

nn = NN((n, 30, 10), crossEntropy, sigmoid)

epoh = []
c_train = []
c_cv = []

for i in range(400):
    nn.trainJoinedData(xy_train, n, 50, 0.01)
    c_train.append(nn.avgCost(xy_train[:, :n], xy_train[:, n:]))
    c_cv.append(nn.avgCost(x_cv, y_cv))
    print ("{0:2d}#: train = {1:.3f} cv = {2:.3f}".format(i, c_train[-1], c_cv[1]))
