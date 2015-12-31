from IPython.display import clear_output
from models import *
from utilityFunctions import *
from pylab import *

nn = NN((1,3,3), crossEntropy, sigmoid)

x_train = np.array([[1], [0]])
y_train = np.array([[0,0,0], [1,0, 0]])

np.set_printoptions(precision = 10)

def t1():
    print (nn.ff(x_train[[0], :]))
    print ("=== z & a ===")
    print (nn.z)
    print (nn.a)

    print ("=== numerical gradient ===")
    nw, nb = nn.ngrad(x_train[[0], :], y_train[[0], :])
    print (nw[0].shape, nw)
    print(nb[0].shape, nb)

    print ("=== backprop gradient ===")
    w, b = nn.grad(x_train[[0], :], y_train[[0], :])
    print (w[0].shape, w)
    print(b[0].shape, b)

    print ("=== d ===")
    print (nn.d)

    print ("=== diffs ===")
    diffw = np.max([np.max(np.abs(a-b)) for a,b in zip(nw, w)])
    diffb = np.max([np.max(np.abs(a-b)) for a,b in zip(nb, b)])

    if diffw < 1e-7 and diffb < 1e-7:
        clear_output(True)

    print ("max w diff: ", diffw)
    print ("max b diff: ", diffb)

def t2():
    print (nn.ff(x_train[0, :]))
    print ("=== z & a[0] ===")
    print (nn.z)
    print (nn.a)

    z0 = nn.z
    a0 = nn.a

    print (nn.ff(x_train[1, :]))
    print ("=== z & a[1] ===")
    print (nn.z)
    print (nn.a)

    z1 = nn.z
    a1 = nn.a

    print (nn.ff(x_train))
    print ("=== z & a ===")
    zzz = [np.vstack([a, b]) for a,b in zip(z0, z1)]
    aaa = [np.vstack([a, b]) for a,b in zip(a0, a1)]
    print(zzz)
    print (nn.z)
    print (nn.a)

    for a,b in zip(nn.z, zzz):
        print (a.shape, b.shape)

    for a,b in zip(nn.a, aaa):
        print (a.shape, b.shape)

    zcond = np.all([np.allclose(a,b) for a,b in zip(zzz, nn.z)])
    acond = np.all([np.allclose(a,b) for a,b in zip(aaa, nn.a)])

    if zcond and acond:
        clear_output(True)
        print ("Correctly calculation a/z in batch vs singular case")

def t3():
    l = 1

    w, b = nn.grad(x_train[[0], :], y_train[[0], :])
    w1, b1 = nn.ngrad(x_train[[0], :], y_train[[0], :])

    print ("=== d[0] ===")
    print ("w\n", w[l])
    print ("b\n", b[l])

    print("=== num ===")
    print (w1[l])
    print (b1[l])

    w, b =nn.grad(x_train[[1], :], y_train[[1], :])
    w1, b1 =nn.ngrad(x_train[[1], :], y_train[[1], :])

    print ("=== d[1] ===")
    print ("w\n", w[l])
    print ("b\n", b[l])

    print("=== num ===")
    print (w1[l])
    print (b1[l])

    w, b = nn.grad(x_train, y_train)
    w1, b1 =nn.ngrad(x_train, y_train)

    print ("=== full ===")
    print ("dw\n", w[l])
    print ("db\n", b[l])

    print ("ww\n", nn.w[l])

    print("=== num ===")
    print ("dw\n", w1[l])
    print ("")
    print ("db\n", b1[l])

    wclose = np.allclose(w[l], w1[l])
    bclose = np.allclose(b[l], b1[l])

    print ("dw [all close]:", wclose)
    print ("db [all close]:", bclose)

    if wclose and bclose:
        clear_output(True)
        print("everything is fine :)")


if __name__ == "__main__":
    t1()
    t2()
    t3()
