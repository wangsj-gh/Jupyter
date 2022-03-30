from mpi4py import MPI
import numpy as np
from osgeo import gdal
from osgeo import osr
import os
from scipy.optimize import curve_fit


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

    def get_lon_lat(self):
        #获取经纬度信息
        gtf = self.GeoTransform
        x_range = range(0, self.XSize)
        y_range = range(0, self.YSize)
        x, y = np.meshgrid(x_range, y_range)
        lon = gtf[0] + x * gtf[1] + y * gtf[2]
        lat = gtf[3] + x * gtf[4] + y * gtf[5]
        lon_lat=np.array(list(zip(lon,lat)))
        return lon_lat

def func(x, m1, m2, m3, m4, m5, m6):
    return m1 + m2 /(1 + np.exp(-m3 * (x-m4))) - m2/(1 + np.exp(-m5 * (x-m6)))

#注意初值
def get_param(yData):
    yData1=yData[0:2]
    yData2=yData[2::]
    if yData2.all()==0:
        Parameters=[-999]*6
    else:
        xData=np.linspace(1, 365, 92)
        Parameters, pcov = curve_fit(func, xData, yData2, p0=[5,40,0.05,140,0.05,270], maxfev=100000000)#,method='trf', maxfev=1000000)
    result=np.hstack((yData1,Parameters))
    return result


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    if rank == 0:

        dir_path = r"D:\Desktop\MPI-document"
        filename = "gee-LAI-12656.tif"
        file_path = os.path.join(dir_path, filename)
        data_path = Dataset(file_path)
        data = data_path.get_data( ).transpose(1,2,0) 
        lon_lat = data_path.get_lon_lat().transpose(0,2,1)
        lon_lat_data=np.hstack((lon_lat.transpose(0,2,1),data.transpose(0,2,1))).transpose(0,2,1)
        sendbuf=lon_lat_data.reshape(-1,lon_lat_data.shape[2])
        colum2=sendbuf.shape[1]
        # np.savetxt('D:\Desktop\gee-LAI-12656-total.txt',sendbuf,fmt='%1.3f')
        # each nprocess total number: the size of each sub-task
        ave, res = divmod(sendbuf.size, nprocs*colum2)
        ave1, res1 = divmod(res, colum2)
        each_nprocess_row = np.array([ave + 1 if p < ave1 else ave  for p in range(nprocs)])
        total_number = np.array(each_nprocess_row)*colum2
        
        # each nprocess star index: the starting index of each sub-task
        star_index = np.array([sum(total_number[:p]) for p in range(nprocs)])

        sendbuf = np.ascontiguousarray(sendbuf)
        star_index = np.ascontiguousarray(star_index)

    else:
        sendbuf = None
        star_index = None
        colum2=None
        
        # initialize on worker processes
        total_number = np.zeros(nprocs, dtype=np.int)
        each_nprocess_row = np.zeros(nprocs, dtype=np.int)

    # broadcast total_number,each_nprocess_row
    comm.Bcast(total_number, root=0)
    comm.Bcast(each_nprocess_row, root=0)
    colum2=comm.bcast(colum2, root=0)
    
    # initialize recvbuf on all processes
    recvbuf = np.zeros((each_nprocess_row[rank],colum2))
    comm.Scatterv([sendbuf, total_number, star_index, MPI.FLOAT], recvbuf, root=0)
    # res = np.array(list(map(get_param,recvbuf)))
    # recvbuf2 = np.zeros((each_nprocess_row[rank],8))
    recvbuf2 = np.zeros(sum(each_nprocess_row)*colum2)
    comm.Gatherv(recvbuf, [recvbuf2, total_number, star_index, MPI.DOUBLE], root=0) 
    # print(recvbuf.shape)
    # print('After Scatterv, process {} has data:'.format(rank))
    if rank == 0:
        np.savetxt('D:\Desktop\gee-LAI-297571-remove.txt',recvbuf,fmt='%1.3f')
        # print(recvbuf2)