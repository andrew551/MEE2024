import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
from photutils.datasets import load_star_image
from photutils.detection import DAOStarFinder

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

#file_path = 'C:\\Users\\15035\\Documents\\Images\\AMO 2017 Eclipse Data\\2017-08-27 Raw Images, Renamed\\'
#data_name = 'Eclipse_17-17-46_0.1s.fts'
#dark_name = 'MasterDark_1p6s_2017-08-21_(150x1p6s).fts'
#flat_name = 'Master_Flat_Frame_(2107-08-26)(128x0p16s).fts'
file_path = 'C:\\Users\\15035\\Documents\\Images\\AMO 2017 Eclipse Data\\2018-02-20 SixMonths_Calibrated\\'
data_name = 'LeoSixMonths-000110s.fts'

data_name = 'ZenithCenteredStack_20x3s.fit'
file_path = 'E:/stardata/'

imgdata = LoadFITSData(file_path+data_name)
#dark = LoadFITSData(file_path+dark_name)
#flat = LoadFITSData(file_path+flat_name)

#imgdata = imgdata-dark
#imgdata = imgdata/flat

#Blur = UnsharpMask(imgdata,16)
#imgdata = imgdata - Blur + 1000
 
ImAvg = np.mean(imgdata)
ImStd = np.std(imgdata)

blk_pt = ImAvg-0.3*ImStd
wht_pt = ImAvg+2.0*ImStd

#hdu = load_star_image()
# data = LoadFITSData(file_path + data_name)

print(imgdata)
plt.imshow(imgdata, cmap='gray', vmin=blk_pt, vmax=wht_pt, interpolation='nearest')
plt.show()

mean, median, std = sigma_clipped_stats(imgdata, sigma=3.0)
daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
sources = daofind(imgdata - median)
for col in sources.colnames:
    sources[col].info.format = '%.8g'
print(sources)

sources.write('StarList.csv',format='csv',overwrite=True)

positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=10.0)
#norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(imgdata, cmap='gray', vmin=blk_pt, vmax=wht_pt, 
           interpolation='nearest')
apertures.plot(color='yellow', lw=2.0, alpha=0.8)
plt.show()
