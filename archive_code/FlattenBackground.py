import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def GaussianKernel(sigma):
    srad=0
    result=1
    while result>0.001:
        result=np.exp(-(srad*srad)/(2*sigma*sigma))
        srad=srad+1
    k=np.arange(2*srad-1)
    k=k-srad+1
    k=np.exp(-(k*k)/(2*sigma*sigma))
    ksum=sum(k)
    k=k/ksum
    return k

def UnsharpMask(array2d, sigma):
    kernel = GaussianKernel(sigma)
    array2d = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, array2d)
    array2d = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, array2d)
    return array2d

def LoadFITSData(file_name):
    fits_file = fits.open(file_name)
    hdu = fits_file[0]
    ImArr = np.array(hdu.data)
    ImArr= ImArr.astype(np.float32)
    return ImArr

file_path = 'C:\\Users\\15035\\Documents\\Images\\AMO 2017 Eclipse Data\\2017-08-27 Raw Images, Renamed\\'
#file_path = 'C:\\Users\\15035\\Documents\\Images\\AMO 2017 Eclipse Data\\2018-02-20 SixMonths_Calibrated\\'


data_name = 'Eclipse_17-17-46_0.1s.fts'
#data_name = 'Eclipse_17-18-31_0.6s.fts'
#data_name = 'Eclipse-17-18-56_0.6s.fts'
#data_name = 'Eclipse_17-18-23_1.6s.fts'
#data_name =  'Ref_17-19-33_1.6s.fts'
#data_name = 'MasterDark_1p6s_2017-08-21_(150x1p6s).fts'
#data_name = 'Master_Flat_Frame_(2107-08-26)(128x0p16s).fts'
#data_name = 'Ref_17-19-33_1.6s.fts'
#data_name = 'LeoSixMonths-000110s.fts'


dark_name = 'MasterDark_1p6s_2017-08-21_(150x1p6s).fts'
flat_name = 'Master_Flat_Frame_(2107-08-26)(128x0p16s).fts'

imgdata = LoadFITSData(file_path+data_name)

dark = LoadFITSData(file_path+dark_name)
flat = LoadFITSData(file_path+flat_name)

imgdata = imgdata-dark
imgdata = imgdata/flat

Blur = UnsharpMask(imgdata,16)
imgdata = imgdata - Blur + 1000
 
ImAvg = np.mean(imgdata)
ImStd = np.std(imgdata)

blk_pt = ImAvg-1.*ImStd
wht_pt = ImAvg+1.*ImStd

fig, ax = plt.subplots()
im = ax.imshow(imgdata,vmin=blk_pt, vmax=wht_pt, cmap='viridis')

ax.set_title('Pan colorbar to shift\n'
             'Zoom colorbar to scale')

fig.colorbar(im, ax=ax, label='Interactive colorbar')

plt.show()



#plt.imshow(imgdata, vmin=blk_pt, vmax=wht_pt, cmap='gray')
#plt.title(data_name)
#plt.show()

print("Program ran")
