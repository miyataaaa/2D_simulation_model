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
    def __init__(self, x, z):
        self.x = x
        self.z = z
        
class InputTopography:
    
    """
    QGISから抽出したx, z, ContAを使う。xはindex0がx=0になるように設定。ｚはそのままでContAはピクセル単位のままexcelファイルに格納しておく。U_flag=Trueの時はDEMから抽出した隆起速度を使う。
    """
    def __init__(self, dirpath, fname, U_flag=False):
        self.fpath = os.path.join(dirpath, fname)
        self.U_flag = U_flag 
        self.df = pd.read_excel(self.fpath, "xzCU", index_col=None)
        self.x = self.df.iloc[:, 0].to_numpy(dtype=np.float64)
        self.z = self.df.iloc[:, 1].to_numpy(dtype=np.float64)
        self.ContA = self.df.iloc[:, 2].to_numpy(dtype=np.float64)
        
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
            
#     def PitRemove(self):
#         for i in range(1, len(self.z)-1):
#             self.z[i] = np.mean(self.z[i-1:i+2])
    
#     def MultipulPitRm(self):
#         iterNum = 1000
#         for i in range(iterNum):
#             self.PitRemove()            
    
class Simulator:
    def __init__(self, InputTopography, U_maker, **kwargs):
        self.initTopography = InputTopography
        self.Dlambda = self.initTopography.x.max()
        self.U_maker = U_maker
        self.dt = kwargs['dt']
        self.nmax = kwargs['nmax']
        self.n = kwargs['n']
        self.m = kwargs['m']
        self.kb = kwargs['kb']
        self.ka = kwargs['ka'] 
        self.a = kwargs['a'] 
        self.dx = 10
        self.xth = 600
        self.degree = 26
        self.DatasetNum = kwargs['DatasetNum']
        self.TrunkName = kwargs['TrunkName']
        self.dirpath = kwargs['dirpath']
        self.DatasetPointer = 0
        self.hdfname = "HDF5_" + self.TrunkName + ".h5"
        self.hdfpath = os.path.join(self.dirpath, self.hdfname)
    
    @profile
    def main(self):
        self.DatasetPointer = 0
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
            
            TopographyInst, exitFrag = self.do_computing(TopographyInst)
            if exitFrag == False:
                with open(self.textpath, "w") as f:
                    print(f"No river {i+1}/{self.DatasetNum}, {datetime.datetime.now()}", file=f)       
                break
            self.save_to_hdf(TopographyInst.z, TopographyInst.x)
            self.DatasetPointer  += 1

                    
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
    def do_computing(self, TopographyInst):
        if self.DatasetPointer == 0:
            init_z = TopographyInst.z
        else:
            init_z = TopographyInst.z[-1]
        
        x = TopographyInst.x
        z_allstep = np.zeros((self.nmax, x.shape[0]))
        U_maker = self.U_maker
        for i in range(self.nmax):
#             print(f"i : {i}")
            if i == 0:
                z_allstep[i] = init_z
                # print(f"zshape={z_allstep.shape}")
            else:
                DividePointer, LheadPointer, RheadPointer = self._get_BoundaryPointer(z_allstep[i-1])
                # print(DividePointer, LheadPointer, RheadPointer)
                ContA = self._CalcCA(DividePointer)
                # print(f"ContA, shape={ContA.shape} DivideContA={ContA[DividePointer]} Lefthead={ContA[LheadPointer-2:LheadPointer+4]} Righthead={ContA[RheadPointer]}")
                if (LheadPointer < 0) or (RheadPointer >= init_z.shape[0] ):
                    return Topography(x, z_allstep), False
                # z_allstep[i] = U_maker.Add_U(np.hstack((self.stream_power_model(z_allstep[i-1][:LheadPointer+1], ContA[:LheadPointer+1], i), 
                #                                         self.threshhold_hillslope(z_allstep[i-1][LheadPointer:RheadPointer+1]), self.stream_power_model(z_allstep[i-1][RheadPointer:], ContA[RheadPointer:], i))))
                z_allstep[i] = U_maker.Add_U(z_allstep[i-1])
                z_allstep[i][:LheadPointer+1] = self.stream_power_model(z_allstep[i][:LheadPointer+1], ContA[:LheadPointer+1], i)
                z_allstep[i][RheadPointer:] = np.flipud(self.stream_power_model(np.flipud(z_allstep[i][RheadPointer:]), np.flipud(ContA[RheadPointer:]), i))
                z_allstep[i][LheadPointer:RheadPointer+1] = self.threshhold_hillslope(z_allstep[i][LheadPointer:RheadPointer+1])
                
       
        return Topography(x, z_allstep), True

    def stream_power_model(self, z, ContA, calc_pointer):
        
        dt = self.dt
        kb = self.kb
        m = self.m
        n = self.n
        dx = self.dx
        riverse_z = np.flipud(z)
        
        z_diff = np.abs(np.flipud(np.concatenate([np.diff(riverse_z), np.zeros(1)])))
        Erate = kb * (ContA ** m) * (((z_diff) / dx) ** n) #侵食速度
        if calc_pointer==1:
            print(f"zdiff: {z_diff}")
            print(f"Erate: {Erate}")
        
        return z - (dt*Erate)
    
    def _CalcCA(self, DividePointer):
        
        ka = self.ka
        a = self.a
        dx = self.dx
        Routlet =(self.Dlambda) - (DividePointer*dx) # 分水界を０にしたときの右側河口までの距離
        LContA = ka * ((np.arange(DividePointer*dx, -1*dx, -1*dx))**a)
        RContA = ka * ((np.arange(0, Routlet+dx, dx))**a)
        
        return np.hstack((LContA[:-1], RContA))
        
    
    def threshhold_hillslope(self, z):
        
        # 閾値標高
        x_hill = np.arange(0, z.shape[0]*self.dx, self.dx)
        zThresh = x_hill*np.tan(np.radians(self.degree))
        zThresh_Left = zThresh+z[0]
        zThresh_Right = np.flipud(zThresh)+z[-1]
        # 閾値標高を超えている所を置き換える。
        z = np.where(z > zThresh_Left, zThresh_Left, z)
        z = np.where(z > zThresh_Right, zThresh_Right, z)
        return z
        
    def _get_BoundaryPointer(self, z):
        # 分水界と左右の河川の谷頭位置のポインター（インデックス）を返す
        return z.argmax(), int(z.argmax()-(self.xth/self.dx)-1), int(z.argmax()+(self.xth/self.dx)+1)
    
    def save_to_hdf(self, z, x):
        zgroup_name = "Z"+self.TrunkName
        # CAgroup_name = "CA"+self.TrunkName
        timeitval = self.nmax*self.dt
#         dataset_name = str(self.DatasetPointer*timeitval)+"~"+str((self.DatasetPointer+1)*timeitval)
        dataset_name = str(self.DatasetPointer)+"_"+str(self.DatasetPointer*timeitval)+"~"+str((self.DatasetPointer+1)*timeitval)
        if self.DatasetPointer == 0:
            if os.path.isfile(self.hdfpath):
                os.remove(self.hdfpath)
            with h5py.File(self.hdfpath, "a") as f:
                gz = f.create_group(zgroup_name)
                gz.create_dataset(dataset_name, data=z, compression="gzip", compression_opts=4)
                gx = f.create_group("x")
                gx.create_dataset("x", data=x, compression="gzip", compression_opts=4)
#                 gca = f.create_group(CAgroup_name)
#                 gca.create_dataset(dataset_name, data=CA, compression="gzip", compression_opts=4)
                
        else:
            with h5py.File(self.hdfpath, "a") as f:
                gz = f[zgroup_name]
                gz.create_dataset(dataset_name, data=z, compression="gzip", compression_opts=4)
                # gca = f[CAgroup_name]
                # gca.create_dataset(dataset_name, data=CA, compression="gzip", compression_opts=4)

class DEM_Uplift_maker:
    
    def __init__(self, InputTopography, dt):
        self.InputTopography = InputTopography
        self.U = InputTopography.U
        self.dt = dt
        
    def Add_U(self, z):
        dt = self.dt
        U = self.U    
        return np.hstack((z[0], z[1:-1] + U[1:-1]*dt, z[-1]))

class Uni_Uplift_maker:
    def __init__(self, U=0.5, dt=3600*24):
        self.U = U
        
    def Add_U(self, z, x):
        U = self.U
        dt = self.dt
        
        ex_boundary_z = z[1:]
        z_add_uplift_ex = ex_boundary_z + (U * dt)
        z_add_uplift = np.hstack((z[0], z_add_uplift_ex))
        
#         return z_add_uplift

class Tilt_Uplift_maker:
    def __init__(self, InputTopography, coef, intercept, dt):
        
        self.coef = coef
        self.intercept = intercept
        self.x = InputTopography.x
        self.dt = dt
        
    def Add_U(self, z):
        
        """
        変動地形学に載っている過去１００万年間の平均隆起速度と、名村川の流路長18kmから算出した直線式で傾動隆起を表現している。
        coefが直線の傾きでinterceptが切片の値。
        """
        x = self.x
        coef = self.coef
        intercept = self.intercept
#         coef = 6e-16*(365*24*3600) # ((0.5-0.2)mm/yr / 15.872km)+0.2mm/yr (Yasudaagawa Area)
#         intercept = 6.3e-12*(365*24*3600) # 0.2mm/yr (Yasudagawa Area)
        dt = self.dt

        return np.hstack((z[0], (z[1:]+(x[1:]*coef + intercept)*dt)))
       
                
class Multi_Simulator:
    
    def __init__(self, initTPList, U_makerList, ParamList):
        self.processingNum = len(initTPList)
        self.instanceList = []
        for initTp, U_maker, Param in zip(initTPList, U_makerList, ParamList):
            self.instanceList.append(Simulator(initTp, U_maker, **Param))
        
                    
    def main(self):
        p = multiprocessing.Pool(self.processingNum)
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
                elif group.startswith('Z'):
                    self.zgroup_name = group
                else:
                    self.CAgroup_name = group
                    
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
    
    # def CAarray(self, zdataset_id):
    #     with h5py.File(self.hdfpath, "r") as f:
    #         return np.array(f[self.zdataset_names[zdataset_id]])
    
    def sorting_dataset(self, group_name, dataset_names):
        
        interval = 0
         
        pointers = []
        j = 0
        # print(dataset_names)
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
        # print(pointers)
        sorted_datasets = []
        for i in pointers:
            start_yr = i*interval
            end_yr = start_yr+interval
            name = '/'+group_name+'/'+str(i)+'_'+str(start_yr)+'~'+str(end_yr)
            sorted_datasets.append(name)
            
        return sorted_datasets
    
# class BoundaryHDFreader(ZHDFreader):
    
#     def __init__(self, dirpath, TrunkName, boundaryName):
#         self.TrunkName = TrunkName
#         self.dirpath = dirpath
#         self.boundaryName = boundaryName
#         self.hdfname = self.boundaryName + "_" + self.TrunkName + ".h5"
#         self.hdfpath = os.path.join(self.dirpath, self.hdfname)
#         print(f"isfile :{os.path.isfile(self.hdfpath)}")
#         self._get_xzgroup()
#         self.zdataset_names = self.sorting_dataset(self.zgroup_name, self.zdataset_names)

class Multi_SimulatorManager:
    
    def __init__(self, dirpath, ExcelList, ParamList, UflagList, UmodelList):
        self.dirpath = dirpath
        self.ExcelList = ExcelList
        self.ParamList = ParamList
        self.UflagList = UflagList
        self.UmodelList = UmodelList
        
    def _Make_InitTpInst(self):
        self.InitTpInsts = []
        for TPexcel, U_flag in zip(self.ExcelList, self.UflagList):
            initTP = InputTopography(self.dirpath, TPexcel, U_flag=U_flag)
#             initTP.MultipulPitRm()
            self.InitTpInsts.append(initTP)
            
    def _save_Param(self):
        
        for param_set in self.ParamList:
            param_path = self.dirpath +'\\'+ param_set['TrunkName'] +'_param_dict.h5'
            df_param = pd.DataFrame(param_set.values(), index=param_set.keys(), dtype=str)
            df_param = df_param.T
            if os.path.isfile(param_path):
                os.remove(param_path)
            df_param.to_hdf(param_path, key = "param_dict")
            
    def _Make_UmakerInst(self):
        self.UmakerInsts = []
        for param_set, UmakerName, initTPinst in zip(self.ParamList, self.UmodelList, self.InitTpInsts):
            if UmakerName == "DEM_Uplift_maker":
                self.UmakerInsts.append(DEM_Uplift_maker(initTPinst, dt=param_set['dt']))
                
    def main(self):
        
        self._Make_InitTpInst()
        self._save_Param()
        self._Make_UmakerInst()
        
        MultiSimulator = Multi_Simulator(self.InitTpInsts, self.UmakerInsts, self.ParamList)
        MultiSimulator.main()
    
if __name__ == "__main__":
    
    # 複数河川の並列実行    

    simulation_1 = {
        "dt" : 10, #1タイムステップ当たりの時間 [yr]
        "nmax": 1000, #時間積分の繰り返し数
        "DatasetNum": 200, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
        "n" : 1,  # ストリームパワーモデルにおける急峻度指数 n
        "m" : 0.5, #　ストリームパワーモデルにおける流域面積指数 m
        "kb" : 1.2e-5, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5) 3.3e-13(n=1, m=0.5), 1.2e-5(n=1, m=0.5, yrスケール)
        "ka" : 3.43,      
        "a" : 1.67,
        "TrunkName" : "99X150U", 
        "dirpath" : r"G:\miyata\Simulation Model\result\Truk\Double\99X",
    }
    
    simulation_2 = {
        "dt" : 10, #1タイムステップ当たりの時間 [yr]
        "nmax": 1000, #時間積分の繰り返し数
        "DatasetNum": 200, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
        "n" : 1,  # ストリームパワーモデルにおける急峻度指数 n
        "m" : 0.5, #　ストリームパワーモデルにおける流域面積指数 m
        "kb" : 1.2e-5, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5) 3.3e-13(n=1, m=0.5), 1.2e-5(n=1, m=0.5, yrスケール)
        "ka" : 3.43,      
        "a" : 1.67,
        "TrunkName" : "99X200U", 
        "dirpath" : r"G:\miyata\Simulation Model\result\Truk\Double\99X",
    }
    
    simulation_3 = {
        "dt" : 10, #1タイムステップ当たりの時間 [yr]
        "nmax": 1000, #時間積分の繰り返し数
        "DatasetNum": 200, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
        "n" : 1,  # ストリームパワーモデルにおける急峻度指数 n
        "m" : 0.5, #　ストリームパワーモデルにおける流域面積指数 m
        "kb" : 1.2e-5, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5) 3.3e-13(n=1, m=0.5), 1.2e-5(n=1, m=0.5, yrスケール)
        "ka" : 3.43,      
        "a" : 1.67,
        "TrunkName" : "99X250U", 
        "dirpath" : r"G:\miyata\Simulation Model\result\Truk\Double\99X",
    }
    
    simulation_4 = {
        "dt" : 10, #1タイムステップ当たりの時間 [yr]
        "nmax": 1000, #時間積分の繰り返し数
        "DatasetNum": 200, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
        "n" : 1,  # ストリームパワーモデルにおける急峻度指数 n
        "m" : 0.5, #　ストリームパワーモデルにおける流域面積指数 m
        "kb" : 1.2e-5, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5) 3.3e-13(n=1, m=0.5), 1.2e-5(n=1, m=0.5, yrスケール)
        "ka" : 3.43,      
        "a" : 1.67,
        "TrunkName" : "99X300U", 
        "dirpath" : r"G:\miyata\Simulation Model\result\Truk\Double\99X",
    }
    
    dirpath = r"G:\miyata\Simulation Model\result\Truk\Double\99X"
    ExcelList = [r"Namura_0.99X1.5U.xlsx", r"Namura_0.99X2.0U.xlsx", r"Namura_0.99X2.5U.xlsx", r"Namura_0.99X3.0U.xlsx"]
    ParamList = [simulation_1, simulation_2, simulation_3, simulation_4]
    UflagList = [True for i in range(len(ExcelList))]
    UmodelList = ["DEM_Uplift_maker" for i in range(len(ExcelList))]
    MultiSimulatorMgInst = Multi_SimulatorManager(dirpath, ExcelList, ParamList, UflagList, UmodelList)
    MultiSimulatorMgInst.main()

#     #単一河川実行
#     dirpath = r"G:\miyata\Simulation Model\result\Truk\Terrace"
#     fname_namura = r"NamuraTerraceProfile.xlsx"
#     initTp_namura = InputTopography(dirpath, fname_namura, U_flag=True)
#     # initTp_namura.MultipulPitRm()
    
#     simulation_namura = {
#         "dt" : 10, #1タイムステップ当たりの時間 [yr]
#         "nmax": 1000, #時間積分の繰り返し数
#         "DatasetNum": 100, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
#         "n" : 1,  # ストリームパワーモデルにおける急峻度指数 n
#         "m" : 0.5, #　ストリームパワーモデルにおける流域面積指数 m
#         "kb" : 1.2e-5, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5) 3.3e-13(n=1, m=0.5), 1.2e-5(n=1, m=0.5, yrスケール)
#         "ka" : 3.43,      
#         "a" : 1.67,
#         "TrunkName" : "Terrace", 
#         "dirpath" : r"G:\miyata\Simulation Model\result\Truk\Terrace"
#     }
    
#     param_path = dirpath +'\\'+ simulation_namura['TrunkName'] +'_param_dict.h5'
#     df_simulator_param = pd.DataFrame(simulation_namura.values(), index=simulation_namura.keys(), dtype=str)
#     df_simulator_param = df_simulator_param.T
#     if os.path.isfile(param_path):
#         os.remove(param_path)
#     df_simulator_param.to_hdf(param_path, key = "param_dict")
#     # 隆起様式を変化させる。
#     U_maker_namura = DEM_Uplift_maker(initTp_namura, dt=simulation_namura['dt'])
#     streamSimulator = Simulator(initTp_namura, U_maker_namura, **simulation_namura)
#     TopographyInst = streamSimulator.main()   
    
#     # 単一河川実行
#     dirpath = r"E:\研究関連\simulation\Trunk\DoubleTrunk"
#     fname_namura = r"Namura_0.99X2U.xlsx"
#     initTp_namura = InputTopography(dirpath, fname_namura, U_flag=True)
    # initTp_namura.MultipulPitRm()
    
#     simulation_namura = {
#         "dt" : 10, #1タイムステップ当たりの時間 [yr]
#         "nmax": 10, #時間積分の繰り返し数
#         "DatasetNum": 2, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
#         "n" : 1,  # ストリームパワーモデルにおける急峻度指数 n
#         "m" : 0.5, #　ストリームパワーモデルにおける流域面積指数 m
#         "kb" : 1.2e-5, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5) 3.3e-13(n=1, m=0.5), 1.2e-5(n=1, m=0.5, yrスケール)
        # "ka" : 3.43,      
        # "a" : 1.67,
#         "TrunkName" : "99X200U", 
#         "dirpath" : r"E:\研究関連\simulation\Trunk\DoubleTrunk"
#     }
    
#     param_path = dirpath +'\\'+ simulation_namura['TrunkName'] +'_param_dict.h5'
#     df_simulator_param = pd.DataFrame(simulation_namura.values(), index=simulation_namura.keys(), dtype=str)
#     df_simulator_param = df_simulator_param.T
#     if os.path.isfile(param_path):
#         os.remove(param_path)
#     df_simulator_param.to_hdf(param_path, key = "param_dict")
#     # 隆起様式を変化させる。
#     U_maker_namura = DEM_Uplift_maker(initTp_namura, dt=simulation_namura['dt'])
#     streamSimulator = Simulator(initTp_namura, U_maker_namura, **simulation_namura)
#     TopographyInst = streamSimulator.main()   
        
        
        