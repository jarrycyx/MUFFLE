import numpy as np
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

USE_SAVED_LUT = True

## Clipping Correction Lookup Table

def clip_gauss_poisson(mean, var, k=1, size=1000):
    poisson = np.random.poisson(lam=mean, size=size)
    gaussian = np.random.normal(loc=0, scale=(var-mean)**0.5, size=size)
    # Var(P + G) = Var(P) + sigma^2 = E(P) + sigma^2 = E(P + G) + sigma^2
    
    result = k*(gaussian + poisson)
    res_mean = np.mean(result)
    res_var = np.var(result)
    
    clip_result = np.clip(result, 0, np.inf)
    clip_mean = np.mean(clip_result)
    clip_var = np.var(clip_result)
    
    return res_mean, res_var, clip_mean, clip_var

def make_clipping_correction_lut():
    lut = []
    max_mean = 10.0
    max_var = 100.0
    size = 500 # lut node num: size^2
    for i in tqdm.trange(size):
        # print(i)
        for j in range(size):
            mean = max_mean/size*i
            var = max_var/size*j
            if var > mean:
                lut_node = clip_gauss_poisson(mean, var)
                # print(lut_node)
                lut.append(list(lut_node))

    clip_correction_lut = np.array(lut)
    print("Generate LUT Complete!")
    return clip_correction_lut

def look_in_lut(mean, var, lut=None):
    if lut == None:
        lut=clip_correction_lut
    max_mean = np.max(lut[:,0])
    max_var = np.max(lut[:,1])
    size = int(lut.shape[0]**0.5)
    
    neighbor = []
    for i in range(lut.shape[0]):
        if abs(lut[i][2]-mean) < max_mean/size*2 \
            and abs(lut[i][3]-var) < max_var/size*2:
            neighbor.append(lut[i])
    # print(neighbor)
    if len(neighbor) > 0:
        return np.mean(neighbor, axis=0)
    else:
        print("No Neighbors in LUT!")


if not USE_SAVED_LUT:
    clip_correction_lut = make_clipping_correction_lut()
    np.save("clip_correction_lut.npy", clip_correction_lut)

clip_correction_lut = np.load("clip_correction_lut.npy")

if __name__ == "__main__":
    print(clip_correction_lut)
    fig = plt.figure()
    ax = Axes3D(fig)
    # fig.add_axes(ax)
    ax.scatter(clip_correction_lut[:,1],clip_correction_lut[:,0],clip_correction_lut[:,2], s=0.01)
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    # fig.add_axes(ax)
    ax.scatter(clip_correction_lut[:,1],clip_correction_lut[:,0],clip_correction_lut[:,3], s=0.01)
    plt.show()

    plt.scatter(clip_correction_lut[:,2], clip_correction_lut[:,3], s=0.01)
    plt.show()

    print(look_in_lut(1, 2, clip_correction_lut))