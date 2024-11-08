from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import os, cv2
# import Clip_Correction

def calibrate_beta_c_r(dark_img_list):
    beta_c_r = np.array([])
    no_dsn_img_list = []
    for dark_img in dark_img_list:
        imgmean = np.mean(dark_img)
        imgvar = np.var(dark_img)
        # print(imgmean, imgvar)
        
        rowmeans = np.mean(dark_img, axis=1)
        beta_c_r = np.concatenate([beta_c_r, rowmeans/imgmean], axis=0)
        
        imgsize = dark_img.shape
        beta_factor = np.reshape(rowmeans/imgmean, (1,-1)).transpose().repeat(imgsize[1], axis=1)
        no_dsn_img_list.append(dark_img / beta_factor)
    
    plt.hist(beta_c_r, bins=40)
    plt.show()
    sigma_beta_c = np.var(beta_c_r)**0.5
    return sigma_beta_c, no_dsn_img_list
    
    
def calibrate_k_c(dark_img_list):
    var_list = []
    mean_list = []
    for dark_img in dark_img_list:
        var, mean = np.var(dark_img), np.mean(dark_img)
        print("mean = {:.4f}, var = {:.4f}".format(mean, var))
        mean_list.append(mean)
        var_list.append(var)
        # lut_res = Clip_Correction.look_in_lut(mean, var)
        # if lut_res is not None:
        #     mean_list.append(lut_res[0])
        #     var_list.append(lut_res[1])
    
    var_list = np.array(var_list)
    mean_list = np.array(mean_list)
    
    
    linreg = LinearRegression()
    linreg.fit(mean_list.reshape(-1,1), var_list.reshape(-1,1))
    
    plt.scatter(mean_list, var_list)
    plt.plot(mean_list.reshape(-1,1), 
             mean_list.reshape(-1,1) * linreg.coef_+linreg.intercept_)
    plt.show()
    
    return linreg.coef_[0][0]

if __name__ == "__main__":
    imgs_dir = "/Volumes/WD-SN550/noise_model_par/nir"

    # imgsize = [400, 800]
    np.random.seed(6)
    
    for channel in range(3):
        dark_img_list = []
        for imgname in sorted(os.listdir(imgs_dir)):
            if ".jpg" in imgname or ".png" in imgname or ".bmp" in imgname:
                img_dir = os.path.join(imgs_dir, imgname)
                img = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)
                img = img[:,:,channel]
                print(imgname, img.shape, np.mean(img))
                #img = img[1000:2000,1500:2500]
                # img = cv2.resize(img, (imgsize[1],imgsize[0])).astype(float) / 5
                dark_img_list.append(img)
                # plt.imshow(img*10, cmap='gray')
                # plt.show()
        
        
        sigma_beta_c, no_dsn_img_list = calibrate_beta_c_r(dark_img_list)
        print("sigma_beta_c = {:.15f}".format(sigma_beta_c))

        k_c = calibrate_k_c(dark_img_list)
        print("k_c = {:.15f}".format(k_c))
        
        '''
        plt.imshow(dark_img_list[0] * 100, cmap='gray')
        plt.show()
        plt.imshow(no_dsn_img_list[0], cmap='gray')
        plt.show()
        '''