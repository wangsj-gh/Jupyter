from mpi4py import MPI
import numpy as np
from osgeo import gdal
from osgeo import osr
import os

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
    xData=np.linspace(1, 365, 92)
    Parameters, pcov = curve_fit(func, xData, yData, p0=[5,40,0.1,140,0.1,270], maxfev=100000000)
    #method='trf'
    return Parameters

def fitting(yData):
    result=np.array(list(map(get_param,yData)))
    return result

def MPI_partition(sendbuf,nprocs,colum):
    ##each nprocess total number: the size of each sub-task
    ave, res = divmod(sendbuf.size, nprocs*colum)
    ave1, res1 = divmod(res, colum)
    each_nprocess_row = np.array([ave + 1 if p < ave1 else ave  for p in range(nprocs)])
    total_number = each_nprocess_row*colum

    ##each nprocess star index: the starting index of each sub-task
    star_index = np.array([sum(total_number[:p]) for p in range(nprocs)])
    return each_nprocess_row,total_number,star_index

def get_reduce_max(recvbuf):

    partial_max = np.zeros(1)
    partial_max[0] = np.max(recvbuf)
    total_max = np.zeros(1)
    comm.Reduce(partial_max,total_max,op=MPI.MAX,root=0)
    return total_max

    
def get_reduce_min(recvbuf):

    partial_min = np.zeros(1)
    partial_min[0] = np.min(recvbuf)
    total_min = np.zeros(1)
    comm.Reduce(partial_min,total_min,op=MPI.MIN,root=0)
    return total_min


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    if rank == 0:
        # dir_path = r"D:\Desktop\mypaper\data"
        # filename = "gee-LAI-108.tif"
        
        # dir_path = r"D:\Desktop\cloud-compute\USA"
        # filename = "gee-LAI-USA-0000006912-0000055296.tif"

        dir_path = r"D:\Desktop"
        filename = "gee-LAI-12656.tif"

        file_path = os.path.join(dir_path, filename)
        data_path = Dataset(file_path)
        sendbuf = data_path.get_data( ).transpose(1,2,0)
        lon,lat = data_path.get_lon_lat_minmax()
        sendbuf = np.ascontiguousarray(sendbuf)
  
        colum2=sendbuf.shape[1]
        # colum3=sendbuf.shape[2]
        
        # each_nprocess_row,total_number,star_index=MPI_partition(sendbuf,nprocs,colum2*colum3)
        
        each_nprocess_row_lon,total_number_lon,star_index_lon=MPI_partition(lon,nprocs,colum2)
    else:
        # sendbuf = None
        # star_index = None
        lon = None
        star_index_lon = None
        colum2=None
        # colum3=None
        # initialize on worker processes
        # total_number = np.zeros(nprocs, dtype=np.int)
        # each_nprocess_row = np.zeros(nprocs, dtype=np.int)
        total_number_lon = np.zeros(nprocs, dtype=np.int)
        each_nprocess_row_lon = np.zeros(nprocs, dtype=np.int)
    #broadcast total_number,each_nprocess_row
    # comm.Bcast(total_number, root=0)
    # comm.Bcast(each_nprocess_row, root=0)
    colum2=comm.bcast(colum2, root=0)
    # colum3=comm.bcast(colum3, root=0)
    
    comm.Bcast(total_number_lon, root=0)
    comm.Bcast(each_nprocess_row_lon, root=0)

    #initialize recvbuf on all processes
    # recvbuf = np.zeros((each_nprocess_row[rank],colum2,colum3))
    # comm.Scatterv([sendbuf, total_number, star_index, MPI.FLOAT], recvbuf, root=0)
    
    #initialize recvbuf on all processes lon and lat
    recvbuf_lon = np.zeros((each_nprocess_row_lon[rank],colum2))
    comm.Scatterv([lon, total_number_lon, star_index_lon, MPI.FLOAT], recvbuf_lon, root=0)

    resutmax=get_reduce_max(recvbuf_lon)
    resutmin=get_reduce_min(recvbuf_lon)
    if rank == 0:
        

        # print(lon_max,lon_min)
        print(resutmax,resutmin)
    # print('After Scatterv, process {} has data:'.format(rank))
    # print(recvbuf.shape)
