import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import multiprocessing 
import h5py
from memory_profiler import profile
import re
import datetime
import sys

class Topography:
    def __init__(self, x, z, ContA):
        self.x = x
        self.z = z
        self.ContA = ContA
        
class InputTopography:
    
    """
    QGISから抽出したx, z, ContAを使う。xはindex0がx=0になるように設定。ｚはそのままでContAはピクセル単位のままexcelファイルに格納しておく。U_flag=Trueの時はDEMから抽出した隆起速度を使う。
    """
    def __init__(self, dirpath, fname, U_flag=False):
        self.fpath = os.path.join(dirpath, fname)
        self.U_flag = U_flag 
        self.df = pd.read_excel(self.fpath, "for simulation", index_col=None)
        self.x = self.df.iloc[:, 0].to_numpy(dtype=np.float64)
        self.z = self.df.iloc[:, 1].to_numpy(dtype=np.float64)
        self.ContA = self.df.iloc[:, 2].to_numpy(dtype=np.float64)*100
        
        if self.U_flag:
            print(f"use Uplift rate from {self.fpath}")
            self.U = self.df.iloc[:, 3].to_numpy(dtype=np.float64)
            
        if self.z[0] > self.z[-1]:
            # index0でのxが0になっている前提
            self.z = np.flipud(self.z)
            self.ContA = np.flipud(self.ContA)
            if self.U_flag:
                self.U = np.flipud(self.U)
#             self.x = np.flipud(self.x)
            
    def PitRemove(self):
        for i in range(1, len(self.z)-1):
            self.z[i] = np.mean(self.z[i-1:i+2])
    
    def MultipulPitRm(self):
        iterNum = 1000
        for i in range(iterNum):
            self.PitRemove()            
    
class Simulator:
    def __init__(self, InputTopography, U_maker, **kwargs):
        self.initTopography = InputTopography
        self.U_maker = U_maker
        self.dt = kwargs['dt']
        self.nmax = kwargs['nmax']
        self.n = kwargs['n']
        self.m = kwargs['m']
        self.kb = kwargs['kb']
        self.UpliftModel = kwargs['UpliftModel']
        self.U = kwargs['U']
        self.DatasetNum = kwargs['DatasetNum']
        self.TrunkName = kwargs['TrunkName']
        self.dirpath = kwargs['dirpath']
        self.DatasetPointer = 0
        self.hdfname = "HDF5_" + self.TrunkName + ".h5"
        self.hdfpath = os.path.join(self.dirpath, self.hdfname)
        self.boundary_xlist = kwargs['boundary_xlist']
        self.boundary_names = kwargs['boundary_names']
        self.boundary_xindex = []
        self._get_boundary_xindex()
        
    def _get_boundary_xindex(self):
        
        for x in self.boundary_xlist:
            self.boundary_xindex.append(np.where((self.initTopography.x<=x)&(x<self.initTopography.x+1))[0][0])
    
    @profile
    def main(self):
        steady_state = False # 定常状態かを判定するフラッグ
        to_wrrite = False # 定常状態を達成し、一度テキストファイルに書き込んだか否かを判定するフラッグ
        for i in range(self.DatasetNum):
            self.textpath = os.path.join(self.dirpath, self.TrunkName+".txt")
            
            # 繰り返し数の進行状況を外部テキストファイルに書き込み（multiprocessingのpoolを使うと結果が返ってくるまで標準出力に表示されないため）
            if i == 0:
                TopographyInst = self.initTopography
                if os.path.isfile(self.textpath):
                    os.remove(self.textpath)
                    
                with open(self.textpath, "w") as f:
                    print(f"now {i+1}/{self.DatasetNum}, {datetime.datetime.now()}", file=f)                
                
            else:
                with open(self.textpath, "a") as f:
                    print(f"now {i+1}/{self.DatasetNum}, {datetime.datetime.now()}", file=f)
            
            # 定常状態か否かで計算を分岐
            if not steady_state:
                # 非定常状態なら計算続行
                TopographyInst = self.do_computing_ver2(TopographyInst)
                self.save_to_hdf(TopographyInst.z, TopographyInst.x)

                for boundary_pointer, index in enumerate(self.boundary_xindex):
                    # boundary_pointerはいくつめの境界かを示す。外側のループで i を使用しているので明示的な単語を使用。
                    self.save_to_hdf_for_boundary(TopographyInst.z.T[index], boundary_pointer)
            else:
                # 定常状態に達した場合は計算をせずに境界の標高値のみを保存する
                for boundary_pointer in range(len(self.boundary_xindex)):
                    # boundary_pointerはいくつめの境界かを示す。外側のループで i を使用しているので明示的な単語を使用。
                    self.save_to_hdf_for_boundary(boundary_z[boundary_pointer], boundary_pointer)
                
            self.DatasetPointer  += 1
            
            # 定常状態かどうかを判定
            if (not steady_state) and (not to_wrrite):
                z_diviation = TopographyInst.z[-2] - TopographyInst.z[-1]
                if z_diviation.sum() == 0:
                    boundary_z = self._steady_boundaryZ(TopographyInst.z[-1])
                    with open(self.textpath, "a") as f:
                        # 赤色で出力。
                        print('\033[31m'+'\nAchieved steady state!!!!!\n'+'\033[0m', file=f)
                        to_wrrite = True 
                    
        print("\nFinish caluclation!")
        
    def _steady_boundaryZ(self, z):
        
        """
        引数ｚ: 定常状態に達したタイムステップでの河口から分水界までの一連の標高値。複数タイムステップでの標高値のndarryではない事に注意。
        返り値：境界での標高値をself.nmax分だけ複成したndarrayを要素に持つリスト。境界の個数だけリストの要素がある。
        """
        
        boundary_z = []
        for xindex in self.boundary_xindex:
            boundary_z.append(np.repeat([z[xindex]], self.nmax))
        
        return boundary_z
        
    def xz_plot(self, TopographyInst):
        
        for i in range(len(TopographyInst.z)):
            plt.plot(TopographyInst.x, TopographyInst.z[i])
    
#     @profile
#     def do_computing(self, TopographyInst):
#         if self.DatasetPointer == 0:
#             init_z = TopographyInst.z
#         else:
#             init_z = TopographyInst.z[-1]
        
#         ContA = TopographyInst.ContA
#         x = TopographyInst.x
# #         z_allstep = np.zeros((self.nmax, x.shape[1]))
#         for i in range(self.nmax):
# #             print(f"i : {i}")
#             if i == 0:
#                 z_old = init_z
#                 z_allstep = init_z.reshape(1, -1)
#             z_new = self.stream_power_model(x, z_old, ContA, i)
#             z_new = self.Add_U(z_new, x)
# #             print(f"z_new : {z_new}")
# #             print(f"before z_allstep: {z_allstep.shape}")
#             z_allstep = np.concatenate((z_allstep, z_new.reshape(1, -1)))
# #             print(f"after z_allstep: {z_allstep.shape}")
#             z_old = np.ravel(z_new)
        
#         calcd_Topography = Topography(x, z_allstep, ContA)
#         return calcd_Topography

#     @profile
    def do_computing_ver2(self, TopographyInst):
        if self.DatasetPointer == 0:
            init_z = TopographyInst.z
        else:
            init_z = TopographyInst.z[-1]
        
        ContA = TopographyInst.ContA
        x = TopographyInst.x
        z_allstep = np.zeros((self.nmax, x.shape[0]))
        U_maker = self.U_maker
        for i in range(self.nmax):
#             print(f"i : {i}")
            if i == 0:
                z_allstep[i] = init_z
            else:
                z_allstep[i] = U_maker.Add_U(self.stream_power_model(x, z_allstep[i-1], ContA, i))
       
        calcd_Topography = Topography(x, z_allstep, ContA)
        del x, z_allstep, ContA, init_z
        return calcd_Topography       

    def stream_power_model(self, x, z, ContA, calc_pointer):
        
        dt = self.dt
        kb = self.kb
        m = self.m
        n = self.n
        riverse_z = np.flipud(z)
        dx = np.concatenate([np.array([10], dtype=np.float64), np.diff(x)])
        z_diff = np.abs(np.flipud(np.concatenate([np.diff(riverse_z), np.zeros(1)])))
        Erate = kb * (ContA ** m) * (((z_diff) / dx) ** n) #侵食速度
#         if calc_pointer==1:
#             print(f"Erate: {Erate}")
        z = z - (dt*Erate)
        
        return z
    
    def save_to_hdf(self, z, x):
        group_name = self.TrunkName
        timeitval = int(self.nmax/365)
#         dataset_name = str(self.DatasetPointer*timeitval)+"~"+str((self.DatasetPointer+1)*timeitval)
        dataset_name = str(self.DatasetPointer)+"_"+str(self.DatasetPointer*timeitval)+"~"+str((self.DatasetPointer+1)*timeitval)
        if self.DatasetPointer == 0:
            if os.path.isfile(self.hdfpath):
                os.remove(self.hdfpath)
            with h5py.File(self.hdfpath, "a") as f:
                g = f.create_group(group_name)
                g.create_dataset(dataset_name, data=z, compression="gzip", compression_opts=4)
                gx = f.create_group("x")
                gx.create_dataset("x", data=x, compression="gzip", compression_opts=4)
                #                 g.create_dataset("param_dict", data=param_dict, compression="gzip", compression_opts=4)
        
        else:
            with h5py.File(self.hdfpath, "a") as f:
                g = f[group_name]
                #print(type(g))
                g.create_dataset(dataset_name, data=z, compression="gzip", compression_opts=4)
                
    def save_to_hdf_for_boundary(self, z, boundary_pointer):
        group_name = self.boundary_names[boundary_pointer]
        timeitval = int(self.nmax/365)
        boundary_fname = group_name + "_" + self.TrunkName + ".h5"
        boundary_fpath = os.path.join(self.dirpath, boundary_fname)
#         dataset_name = str(self.DatasetPointer*timeitval)+"~"+str((self.DatasetPointer+1)*timeitval)
        dataset_name = str(self.DatasetPointer)+"_"+str(self.DatasetPointer*timeitval)+"~"+str((self.DatasetPointer+1)*timeitval)
        if self.DatasetPointer == 0:
            if os.path.isfile(boundary_fpath):
                os.remove(boundary_fpath)
            with h5py.File(boundary_fpath, "a") as f:
                g = f.create_group(group_name)
                g.create_dataset(dataset_name, data=z, compression="gzip", compression_opts=4)
        else:
            with h5py.File(boundary_fpath, "a") as f:
                g = f[group_name]
                #print(type(g))
                g.create_dataset(dataset_name, data=z, compression="gzip", compression_opts=4)

class DEM_Uplift_maker:
    
    def __init__(self, InputTopography, dt=3600*24):
        self.InputTopography = InputTopography
        self.U = InputTopography.U
        self.dt = dt
        
    def Add_U(self, z):
        dt = self.dt
        U = self.U    
        return np.hstack((z[0], z[1:] + U[1:]*dt))

# class Uni_Uplift_maker:
#     def __init__(self, U=0.5, dt=3600*24):
#         self.U = U
        
#     def Add_U(self, z, x):
#         U = self.U
#         dt = self.dt
        
#         ex_boundary_z = z[1:]
#         z_add_uplift_ex = ex_boundary_z + (U * dt)
#         z_add_uplift = np.hstack((z[0], z_add_uplift_ex))
        
#         return z_add_uplift

# class Tilt_Uplift_maker:
#     def __init__(self, U=0.5, dt=3600*24):
        
#     def Add_U(self, z, x):
        
#         """
#         変動地形学に載っている過去１００万年間の平均隆起速度と、名村川の流路長18kmから算出した直線式で傾動隆起を表現している。
#         coefが直線の傾きでinterceptが切片の値。
#         """
#         coef = 6e-16 # ((0.5-0.2)mm/yr / 15.872km)+0.2mm/yr (Yasudaagawa Area)
#         intercept = 6.3e-12 # 0.2mm/yr (Yasudagawa Area)
#         U = x[1:]*coef + intercept
#         dt = self.dt
        
#         ex_boundary_z = z[1:]
#         z_add_uplift_ex = ex_boundary_z + (U * dt)
#         z_add_uplift = np.hstack((z[0], z_add_uplift_ex))
        
#         return z_add_uplift
       
                
class Multi_Simulator:
    
    def __init__(self, initTp1, initTp2, U_maker1, U_maker2, simulation_param1, simulation_param2, ):
        self.instanceList = [Simulator(initTp1, U_maker1, **simulation_param1), Simulator(initTp2, U_maker2, **simulation_param2)]
                    
    def main(self):
        p = multiprocessing.Pool(2)
        print("Now doing main calculation......................")
        p.map(self.does_computing, self.instanceList)
        print("--------Finish--------------------")
        
    def does_computing(self, Tpinstance):
        return Tpinstance.main()
        
                
class ZHDFreader:
    
    def __init__(self, dirpath, TrunkName):
        self.TrunkName = TrunkName
        self.dirpath = dirpath
        self.hdfname = "HDF5_" + self.TrunkName + ".h5"
        self.hdfpath = os.path.join(self.dirpath, self.hdfname)
        print(f"isfile :{os.path.isfile(self.hdfpath)}")
        self._get_xzgroup()
        self.zdataset_names = self.sorting_dataset(self.zgroup_name, self.zdataset_names)
        
    def _get_xzgroup(self):
        
        with h5py.File(self.hdfpath, "r") as f:
            for group in f:
                if group=="x":
                    self.xgroup_name = group
                else:
                    self.zgroup_name = group
                    
            self.zdataset_names = []
            for value in f[self.zgroup_name].values():
#                     print(value.name)
                    self.zdataset_names.append(value.name)                    
        
    def zarray(self, zdataset_id):
        with h5py.File(self.hdfpath, "r") as f:
            return np.array(f[self.zdataset_names[zdataset_id]])
        
    def xarray(self):
        with h5py.File(self.hdfpath, "r") as f:
            return np.array(f['/x/x'])
    
    def sorting_dataset(self, group_name, dataset_names):
        
        interval = 0
         
        pointers = []
        j = 0
        print(dataset_names)
        for name in dataset_names:
            m = re.match('\d+_', name[len(group_name)+2:])
            pointer = m.group()
            pointers.append(int(pointer[:-1]))
            if j == 0:
                start_yr_moj = re.match('\d+~', name[len(group_name)+len(pointer)+2:])
                start_yr = start_yr_moj.group()
                end_yr = name[len(group_name)+len(pointer)+len(start_yr)+2:]
                interval = int(end_yr)-int(start_yr[:-1])
            j += 1
        pointers.sort()
        print(pointers)
        sorted_datasets = []
        for i in pointers:
            start_yr = i*interval
            end_yr = start_yr+interval
            name = '/'+group_name+'/'+str(i)+'_'+str(start_yr)+'~'+str(end_yr)
            sorted_datasets.append(name)
            
        return sorted_datasets
    
class BoundaryHDFreader(ZHDFreader):
    
    def __init__(self, dirpath, TrunkName, boundaryName):
        self.TrunkName = TrunkName
        self.dirpath = dirpath
        self.boundaryName = boundaryName
        self.hdfname = self.boundaryName + "_" + self.TrunkName + ".h5"
        self.hdfpath = os.path.join(self.dirpath, self.hdfname)
        print(f"isfile :{os.path.isfile(self.hdfpath)}")
        self._get_xzgroup()
        self.zdataset_names = self.sorting_dataset(self.zgroup_name, self.zdataset_names)

          
if __name__ == "__main__":
    
    # ２河川の並列実行
    
    dirpath = r"G:\miyata\Simulation Model\result\Truk"
    fname_yasuda = r"area-length law yasudariver.xlsx"
    fname_namura = r"area-length law namurariver.xlsx"
    initTp_yasuda = InputTopography(dirpath, fname_yasuda, U_flag=True)
    initTp_yasuda.MultipulPitRm()
    initTp_namura = InputTopography(dirpath, fname_namura, U_flag=True)
    initTp_namura.MultipulPitRm()
    
    simulation_yasuda = {
        "dt" : 3600*24, #1タイムステップ当たりの時間 [s]
        "nmax": 365*100, #時間積分の繰り返し数
        "DatasetNum": 5, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
        "n" : 1,  # ストリームパワーモデルにおける急峻度指数 n
        "m" : 0.5, #　ストリームパワーモデルにおける流域面積指数 m
        "kb" : 3.3e-12, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5) 3.3e-13(n=1, m=0.5)
        "UpliftModel"  : "uniform_uplift", # 隆起様式の選択
        "U" : 1.58e-11, # 隆起速度 [m/s] 0.8mm/yr = 2.54e-11, 0.5mm/yr = 1.58e-11,  0.1mm/yr = 3.1e-12
        "TrunkName" : "yasuda", 
        "dirpath" : r"G:\miyata\Simulation Model\result\Truk",
        "boundary_xlist" : [10542, 15873], # 目的の支流との合流点のx座標値を指定。
        "boundary_names" : ["NY1", "NY2"], # 目的の支流との合流点を識別するための名前を指定。ファイル名はboundary_names[i]_Trunkname

    }
    
    simulation_namura = {
        "dt" : 3600*24, #1タイムステップ当たりの時間 [s]
        "nmax": 365*100, #時間積分の繰り返し数
        "DatasetNum": 5, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
        "n" : 1,  # ストリームパワーモデルにおける急峻度指数 n
        "m" : 0.5, #　ストリームパワーモデルにおける流域面積指数 m
        "kb" : 3.3e-12, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5) 3.3e-13(n=1, m=0.5)
        "UpliftModel"  : "uniform_uplift", # 隆起様式の選択
        "U" : 1.58e-11, # 隆起速度 [m/s] 0.8mm/yr = 2.54e-11, 0.5mm/yr = 1.58e-11,  0.1mm/yr = 3.1e-12
        "TrunkName" : "namura", 
        "dirpath" : r"G:\miyata\Simulation Model\result\Truk",
        "boundary_xlist" : [17205, 17895] , # 目的の支流との合流点のx座標値を指定。
        "boundary_names" : ["NY1", "NY2"], # 目的の支流との合流点を識別するための名前を指定。ファイル名はboundary_names[i]_Trunkname

    }
    
    param_path_yasuda = dirpath +'\\'+ simulation_yasuda['TrunkName'] +'_param_dict.h5'
    df_simulator_param_yasuda = pd.DataFrame(simulation_yasuda.values(), index=simulation_yasuda.keys(), dtype=str)
    df_simulator_param_yasuda = df_simulator_param_yasuda.T
    if os.path.isfile(param_path_yasuda):
        os.remove(param_path_yasuda)
    df_simulator_param_yasuda.to_hdf(param_path_yasuda, key = "param_dict")
    
    param_path_namura = dirpath +'\\'+ simulation_namura['TrunkName'] +'_param_dict.h5'
    df_simulator_param_namura = pd.DataFrame(simulation_namura.values(), index=simulation_namura.keys(), dtype=str)
    df_simulator_param_namura = df_simulator_param_namura.T
    if os.path.isfile(param_path_namura):
        os.remove(param_path_namura)
    df_simulator_param_namura.to_hdf(param_path_namura, key = "param_dict")
    
    U_maker_yasuda = DEM_Uplift_maker(initTp_yasuda, dt=simulation_yasuda['dt'])
    U_maker_namura = DEM_Uplift_maker(initTp_namura, dt=simulation_namura['dt'])
    
    MultiSimulator = Multi_Simulator(initTp_yasuda, initTp_namura, U_maker_yasuda, U_maker_namura, simulation_yasuda, simulation_namura)
    MultiSimulator.main()
    
    
    # 単一河川実行
#     dirpath = r"G:\miyata\Simulation Model\result\Truk"
#     fname_namura = r"area-length law namurariver.xlsx"
#     initTp_namura = InputTopography(dirpath, fname_namura)
#     initTp_namura.MultipulPitRm()
    
#     simulation_namura = {
#         "dt" : 3600*24, #1タイムステップ当たりの時間 [s]
#         "nmax": 365, #時間積分の繰り返し数
#         "DatasetNum": 2, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
#         "n" : 1,  # ストリームパワーモデルにおける急峻度指数 n
#         "m" : 0.5, #　ストリームパワーモデルにおける流域面積指数 m
#         "kb" : 3.3e-12, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5) 3.3e-13(n=1, m=0.5)
#         "UpliftModel"  : "uniform_uplift", # 隆起様式の選択
#         "U" : 1.58e-11, # 隆起速度 [m/s] 0.8mm/yr = 2.54e-11, 0.5mm/yr = 1.58e-11,  0.1mm/yr = 3.1e-12
#         "TrunkName" : "namura", 
#         "dirpath" : r"G:\miyata\Simulation Model\result\Truk", 
#         "boundary_xlist" : [17205, 17895], # 目的の支流との合流点のx座標値を指定。
#         "boundary_names" : ["NY1", "NY2"], # 目的の支流との合流点を識別するための名前を指定。ファイル名はboundary_names[i]_Trunkname
#     }
    
#     param_path = dirpath +'\\'+ simulation_namura['TrunkName'] +'_param_dict.h5'
#     df_simulator_param = pd.DataFrame(simulation_namura.values(), index=simulation_namura.keys(), dtype=str)
#     df_simulator_param = df_simulator_param.T
#     if os.path.isfile(param_path):
#         os.remove(param_path)
#     df_simulator_param.to_hdf(param_path, key = "param_dict")
#     streamSimulator = Simulator(initTp_namura, **simulation_namura)
#     TopographyInst = streamSimulator.main()   
