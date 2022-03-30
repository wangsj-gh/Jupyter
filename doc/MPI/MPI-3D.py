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
        get_dataset = gdal.Open(self.in_file)
        data = get_dataset.ReadAsArray(0,0,self.XSize,self.YSize)
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

def func(x, m1, m2, m3, m4, m5, m6):
    return m1 + m2 /(1 + np.exp(-m3 * (x-m4))) - m2/(1 + np.exp(-m5 * (x-m6)))

#注意初值
def get_param(yData):
    if yData.all()==0:
        Parameters=[-999]*6
    else:
        xData=np.linspace(1, 365, 92)
        Parameters, pcov = curve_fit(func, xData, yData, \
            p0=[1,40,0.05,140,0.05,270], maxfev=10000000)
            #,method='trf', maxfev=1000000)
    return Parameters

def fitting(yData):
    result=np.array(list(map(get_param,yData)))
    return result

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    if rank == 0:
        dir_path = r"D:\Desktop\MPI-document"
        filename = "gee-LAI-12656.tif"
        
        # dir_path = r"D:\Desktop\cloud-compute\USA"
        # filename = "gee-LAI-USA-0000006912-0000055296.tif"

        file_path = os.path.join(dir_path, filename)
        data_path = Dataset(file_path)
        sendbuf = data_path.get_data( ).transpose(1,2,0)
        lon,lat = data_path.get_lon_lat_minmax()
        sendbuf = np.ascontiguousarray(sendbuf)
       
        ## sendbuf = np.random.random((13,12,10))
        colum1=sendbuf.shape[0]
        colum2=sendbuf.shape[1]
        colum3=sendbuf.shape[2]
        
        ##each nprocess total number: the size of each sub-task
        ave, res = divmod(sendbuf.size, nprocs*colum2*colum3)
        ave1, res1 = divmod(res, colum2*colum3)
        each_nprocess_row = np.array([ave + 1 if p < ave1 else ave  for p in range(nprocs)])
        total_number = each_nprocess_row*colum2*colum3
        
        ##each nprocess star index: the starting index of each sub-task
        star_index = np.array([sum(total_number[:p]) for p in range(nprocs)])

    else:
        sendbuf = None
        star_index = None
        colum1=None
        colum2=None
        colum3=None
        ## initialize on worker processes
        total_number = np.zeros(nprocs, dtype=np.int)
        each_nprocess_row = np.zeros(nprocs, dtype=np.int)

    ##broadcast total_number,each_nprocess_row
    comm.Bcast(total_number, root=0)
    comm.Bcast(each_nprocess_row, root=0)
    colum1=comm.bcast(colum1, root=0)
    colum2=comm.bcast(colum2, root=0)
    colum3=comm.bcast(colum3, root=0)
 
    ##initialize recvbuf on all processes
    recvbuf = np.zeros((each_nprocess_row[rank],colum2,colum3))
    comm.Scatterv([sendbuf, total_number, star_index, MPI.FLOAT], recvbuf, root=0)
    res = np.array(list(map(fitting,recvbuf)))
    # print(res.shape)
    # comm.Barrier()
    recvbuf2 = np.zeros((each_nprocess_row[rank],colum2,6))
    comm.Gatherv(res, [recvbuf2, total_number, star_index, MPI.DOUBLE], root=0)  

    if rank == 0:
        print(recvbuf2.shape)
