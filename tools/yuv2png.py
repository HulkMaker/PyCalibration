# -*- coding: utf-8 -*-

from numpy import *
import Image
screenLevels = 255.0


def yuv_import(filename, dims, numfrm, startfrm):
    fp = open(filename, 'rb')
    blk_size = prod(dims) * 3 / 2
    fp.seek(blk_size * startfrm, 0)
    Y = []
    U = []
    V = []
    print dims[0]
    print dims[1]
    d00 = dims[0] // 2
    d01 = dims[1] // 2
    print d00
    print d01
    Yt = zeros((dims[0], dims[1]), uint8, 'C')
    Ut = zeros((d00, d01), uint8, 'C')
    Vt = zeros((d00, d01), uint8, 'C')
    for i in range(numfrm):
        for m in range(dims[0]):
            for n in range(dims[1]):
                # print m,n
                Yt[m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                Ut[m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                Vt[m, n] = ord(fp.read(1))
        Y = Y + [Yt]
        U = U + [Ut]
        V = V + [Vt]
    fp.close()
    return (Y, U, V)


if __name__ == '__main__':
    data = yuv_import('refocus_capture_input_2560x1920_0_20170514153242.yuv', (2560, 1920), 1, 0)
# print data
# im=array2image(array(data[0][0]))
    YY = data[0][0]
    print YY.shape
    for m in range(2):
        print m, ': ', YY[m, :]

    im = Image.fromstring('L', (2560, 1920), YY.tostring())
    im.show()
    im.save('a.png')
