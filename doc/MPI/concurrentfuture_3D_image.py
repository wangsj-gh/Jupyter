import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
#from osgeo import gdal
#from osgeo import osr
import gdal
import osr
import os
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor



class Dataset:
    def __init__(self, in_file):
        self.in_file = in_file  # Tiff或者ENVI文件

        dataset = gdal.Open(self.in_file)
        self.XSize = dataset.RasterXSize  # 网格的X轴像素数量
        self.YSize = dataset.RasterYSize  # 网格的Y轴像素数量
        self.Bands = dataset.RasterCount  # 波段数
        self.GeoTransform = dataset.GetGeoTransform()  # 投影转换信息
        self.ProjectionInfo = dataset.GetProjection()  # 投影信息
    
    def get_data(self):
        #band: 读取第几个通道的数据
        dataset = gdal.Open(self.in_file)
        data = dataset.ReadAsArray(0,0,self.XSize,self.YSize)
        return data


    def get_lon_lat_minmax(self):
        #获取经纬度信息
        gtf = self.GeoTransform
        x_range = range(0, self.XSize)
        y_range = range(0, self.YSize)
        x, y = np.meshgrid(x_range, y_range)
        longitude = gtf[0] + x * gtf[1] + y * gtf[2]
        latitude = gtf[3] + x * gtf[4] + y * gtf[5]
        return longitude,latitude 
        
def get_image(longitude,latitude,res):
    xmin,ymin,xmax,ymax = [longitude.min(),latitude.min(),longitude.max(),latitude.max()]
    nrows,ncols,bandsNum = res.shape
    xres = (xmax-xmin)/float(ncols)
    yres = (ymax-ymin)/float(nrows)
    geotransform=(xmin,xres,0,ymax,0, -yres)
    output_raster = gdal.GetDriverByName('GTiff').Create('C:\Users\student\Desktop\concurrentfuture_3D_999_image.tif',ncols, nrows, bandsNum ,gdal.GDT_Float32)
    output_raster.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    output_raster.SetProjection(srs.ExportToWkt())
    for k in range(bandsNum):
        data=res[:,:,k]
        output_raster.GetRasterBand(k+1).WriteArray(data)
        output_raster.GetRasterBand(k+1).SetDescription("p%s"%k)
    
    return output_raster.FlushCache()

def func(x, m1, m2, m3, m4, m5, m6):
    return m1 + m2 /(1 + np.exp(-m3 * (x-m4))) - m2/(1 + np.exp(-m5 * (x-m6)))

#注意初值
def get_param(yData):
#     yData=yData[2::]
    if yData.all()==0:
        Parameters=-999
    else:
        xData=np.linspace(1, 365, 92)
        Parameters, pcov = curve_fit(func, xData, yData, p0=[5,40,0.1,140,0.1,270], maxfev=100000000)#,method='trf', maxfev=1000000)
    return Parameters

def fitting(yData):
    result=np.array(list(map(get_param,yData)))
    return result

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    start_time=time.time()
    dir_path = r"C:\Users\student\Desktop"
    filename = "gee-LAI-297571.tif"
    #filename = "gee-LAI-Land-Caribbean-mask.tif"
    file_path = os.path.join(dir_path, filename)
    data_path = Dataset(file_path)
    data = data_path.get_data( ).transpose(1,2,0) 
    lon,lat = data_path.get_lon_lat_minmax()
    
    pool = ProcessPoolExecutor(max_workers=8)
    res = np.array(list(pool.map(fitting,data)))
    pool.shutdown(wait=True)
    end_time=time.time()
    #np.savetxt('D:\Desktop\concurrentfuture_3D_image.txt',res,fmt='%1.3f')
    imagecollection=get_image(lon,lat,res)
    print('pool time :',end_time-start_time,'seconds')
    print('image:',time.time()-end_time,'seconds')
   # print ("Thread pool execution in " + str(time.time() - start_time), "seconds")
