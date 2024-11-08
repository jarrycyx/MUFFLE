import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as Interp


def curve_single(img, nodes, show_curve=False):
    imshape = img.shape
    nodes = np.array(nodes)
    itp = Interp.splrep(nodes[:,0], nodes[:,1], k=3)
    new_img = Interp.splev(img.reshape(-1), itp)

    ### Show curve
    if show_curve:
        plt.figure()
        plt.title("L Curve")
        plt.plot(np.arange(0,1,0.01), Interp.splev(np.arange(0,1,0.01), itp))
        # plt.show()

    return np.clip(new_img.reshape(imshape), 0, 1)


def curve(img, l_nodes, r_nodes=None, g_nodes=None, b_nodes=None, show_curve=False):
    hsvimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    l_curve_img = curve_single(hsvimg[:,:,2], l_nodes, show_curve=show_curve)

    if r_nodes is not None:
        pass
    if g_nodes is not None:
        pass
    if b_nodes is not None:
        pass

    hsvimg[:,:,2] = l_curve_img
    return cv2.cvtColor(hsvimg, cv2.COLOR_HSV2RGB)

def saturation(img, adj):
    hsvimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s_img = hsvimg[:,:,1] * (2**(adj/100))

    hsvimg[:,:,1] = np.clip(s_img, 0, 255)
    return cv2.cvtColor(hsvimg, cv2.COLOR_HSV2RGB)

def natural_saturation(img, adj):
    hsvimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s_img =  curve_single(hsvimg[:,:,1], nodes=[ [0,     0                   ],
                                                 [0.25,  0.25+0.25*adj/100   ],
                                                 [0.75,  0.75+0.1*adj/100    ],
                                                 [1,     1                   ]], show_curve=False)

    hsvimg[:,:,1] = np.clip(s_img, 0, 255)
    return cv2.cvtColor(hsvimg, cv2.COLOR_HSV2RGB)


def contrast(img, adj):
    return curve(img, l_nodes=[ [0,     0                   ],
                                [0.25,  0.25-0.15*adj/100   ],
                                [0.75,  0.75+0.15*adj/100   ],
                                [1,     1                   ]], show_curve=False)

def exposure(img, exp):
    return np.clip(img*(2**exp), 0, 1)


def color_temp(img, adj):
    labimg = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    b_img = labimg[:,:,2] + adj / 100 * 32
    labimg[:,:,2] = np.clip(b_img, -127, 128)

    return cv2.cvtColor(labimg, cv2.COLOR_LAB2RGB)

