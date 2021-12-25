import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from DoubleTrunk import ZHDFreader
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class ExcelTopography:
    
    """Excelファイルに格納された初期地形プロファイルを読み込むクラス。
    対象のExcelファイルが格納されたディレクトリの絶対パスと拡張子込みのExcelファイル名を引数に指定。
    """
    
    def __init__(self, dirpath, fname):
        self.fpath = os.path.join(dirpath, fname) 
        self.df = pd.read_excel(self.fpath, "xzCU", index_col=None)
        self.x = self.df.iloc[:, 0].to_numpy(dtype=np.float64)
        self.z = self.df.iloc[:, 1].to_numpy(dtype=np.float64)
        self.ContA = self.df.iloc[:, 2].to_numpy(dtype=np.float64)
        self.U = self.df.iloc[:, 3].to_numpy(dtype=np.float64) * 1000 # タイムスケールをm/yr -> mm/yrに変更
        self.xth = 600
        self.dx = 10
        self.m = 0.5
        self.n = 1
        
    def Left_River_x(self):
        return self.x[:self.z.argmax()-int(self.xth/self.dx)]
    
    def Left_River_z(self):
        return self.z[:self.z.argmax()-int(self.xth/self.dx)]

    def Left_River_CA(self):
        return self.ContA[:self.z.argmax()-int(self.xth/self.dx)]
    
    def Left_River_u(self):
        return self.U[:self.z.argmax()-int(self.xth/self.dx)]
  
    # def Left_River_xzCAu(self):
    #     return self.x[:self.z.argmax()-int(self.xth/self.dx)], self.z[:self.z.argmax()-int(self.xth/self.dx)], self.ContA[:self.z.argmax()-int(self.xth/self.dx)], self.U[:self.z.argmax()-int(self.xth/self.dx)]
    
    def Right_River_x(self):
        return self.x[self.z.argmax()+int(self.xth/self.dx):]
    
    def Right_River_z(self):
        return self.z[self.z.argmax()+int(self.xth/self.dx):]

    def Right_River_CA(self):
        return self.ContA[self.z.argmax()+int(self.xth/self.dx):]
    
    def Right_River_u(self):
        return self.U[self.z.argmax()+int(self.xth/self.dx):]
    
    def whole_xz(self):
        return self.x, self.z
    # def Right_River_xzCAu(self):
    #     return self.x[self.z.argmax()+int(self.xth/self.dx):], self.z[self.z.argmax()+int(self.xth/self.dx):], self.ContA[self.z.argmax()+int(self.xth/self.dx):], self.U[self.z.argmax()+int(self.xth/self.dx):]
    
    def Chi_Left(self):
        ContA, U = self.Left_River_CA(), self.Left_River_u()
        tempChi = ((U/(ContA**self.m))**(1/self.n)) * self.dx       
        return tempChi.cumsum()
    
    def Chi_Right(self):
        ContA, U = np.flipud(self.Right_River_CA()), np.flipud(self.Right_River_u())
        tempChi = ((U/(ContA**self.m))**(1/self.n)) * self.dx       
        return tempChi.cumsum()
        
class CalcdHDFTopography:
    
    """
    Ver3_Trunk.pyを使用して計算され、HDFファイルに保存された地形データのインスタンスを生成するためのクラス。
    
    Ver3_Trunk.pyで使用したそれぞれの地形を識別するためのTrunkName, 
    HDFファイルが格納されたディレクトリの絶対パスをコンストラクタの引数に渡す。計算された地形HDFファイルとその時のパラメータHDFファイルは同じ階層にある前提。
    """
    
    def __init__(self, dirpath, TrunkName):
        self.reader = ZHDFreader(dirpath, TrunkName)
        self.TrunkName = TrunkName
        self.dirpath = dirpath
        self._get_Param()
        self.dx = 10
        self.xth = 600
        self.x = self.xarray()
        
    def _get_Param(self):
    
        self.Parampath = os.path.join(self.dirpath, self.TrunkName+"_param_dict.h5")
        self.paramDf = pd.read_hdf(self.Parampath, key="param_dict", index=False)
        self.dt = int(self.paramDf['dt'].values[0])
        self.nmax = int(self.paramDf['nmax'].values[0])
        print(f"dt, nmax: {self.dt, self.nmax}")
        self.n = float(self.paramDf['n'].values[0])
        self.m = float(self.paramDf['m'].values[0])
        self.kb = float(self.paramDf['kb'].values[0])
        self.ka = float(self.paramDf['ka'].values[0])
        self.a = float(self.paramDf['a'].values[0])
        self.DatasetNum = int(self.paramDf['DatasetNum'].values[0])
        
    def zMatrix(self, dataset_pointer):
        return self.reader.zarray(int(dataset_pointer))
    
    def xarray(self):
        return self.reader.xarray()
    
    def zarray(self, dataset_pointer, row_index):
        return self.zMatrix(int(dataset_pointer))[int(row_index)]
    
    def _Left_River(self, dataset_pointer, row_index):
        z = self.zarray(int(dataset_pointer), int(row_index))
        return self.x[:z.argmax()-int(self.xth/self.dx)], z[:z.argmax()-int(self.xth/self.dx)]
    
    def _Right_River(self, dataset_pointer, row_index):
        z = self.zarray(int(dataset_pointer), int(row_index))
        return self.x[z.argmax()+int(self.xth/self.dx):], z[z.argmax()+int(self.xth/self.dx):]
    
    def Left_River(self, yr):
        dataset_pointer, row_index = divmod(yr, (self.nmax-1)*self.dt)
        # print(f"dataset_pointer, row_index {dataset_pointer} {row_index}")
        return self._Left_River(int(dataset_pointer), int(row_index/self.dt))
    
    def Right_River(self, yr):
        dataset_pointer, row_index = divmod(yr, (self.nmax-1)*self.dt)
        return self._Right_River(int(dataset_pointer), int(row_index/self.dt))
    
    def whole_TopograpyZ(self, yr):
        dataset_pointer, row_index = divmod(yr, (self.nmax-1)*self.dt)
        # print(f"dataset_pointer, row_index: {dataset_pointer, row_index}")
        # print(f"dataset_pointer*(self.nmax-1)*self.dt + row_index: {dataset_pointer*(self.nmax-1)*self.dt + row_index}")
        return self.zarray(dataset_pointer, row_index/self.dt)
    
class Calculator:
    """
    3つの地形データを読み込む。
    1. 定常状態の地形データ　2. 地形パラメータ変化直後の地形データ　3. 地形パラメータ変化後から新たな定常状態まで計算された地形データ
    
    1.定常状態の地形データを格納したInitTpInst(ExcelTopographyクラスのインスタンス)
    2.パラメータ変化直後の状態の地形データを格納したDisequilibriumTpInst(ExcelTopographyクラスのインスタンス)
    3. Ver3_Trunk.pyを使用して計算され、HDFファイルに保存された、。(CalcdHDFTopographyクラスのインスタンス)
    
    2の地形を読み込むのは隆起速度と流域面積を読み込むため。
    """
    
    def __init__(self, InitTpInst, DisequilibriumTpInst, CalcdTpInst):
        self.InitTpInst = InitTpInst
        self.DisequilibriumTpInst = DisequilibriumTpInst
        self.CalcdTpInst = CalcdTpInst
        self.dt = self.CalcdTpInst.dt
        self.nmax = self.CalcdTpInst.nmax
        self.n = self.CalcdTpInst.n
        self.m = self.CalcdTpInst.m
        self.kb = self.CalcdTpInst.kb
        self.DatasetNum = self.CalcdTpInst.DatasetNum
        self.dx = self.CalcdTpInst.dx
        self.ka = self.CalcdTpInst.ka
        self.a = self.CalcdTpInst.a
        self.xth = self.CalcdTpInst.xth
        self.totalYr = int(self.DatasetNum*(self.nmax-1)*self.dt)
        print(f"totalYr: {self.totalYr}")
        
    def _InitSteadyChiZ_Left(self):    
        return self.InitTpInst.Chi_Left(), self.InitTpInst.Left_River_z()
    
    def _InitSteadyChiZ_Rigth(self):    
        return self.InitTpInst.Chi_Right(), self.InitTpInst.Right_River_z()
        
    def _InitDisequilibriumChi_Left(self):    
        return self.DisequilibriumTpInst.Chi_Left()
    
    def _InitDisequilibriumChi_Rigth(self):    
        return self.DisequilibriumTpInst.Chi_Right()
        
    def Chi_Left(self, x):
        """
        引数x: 河川部分のｘ値
        """
        ContA = self.ka * (np.flipud(x+self.xth)**self.a)
        U = self.DisequilibriumTpInst.U[:x.shape[0]]  
        tempChi = ((U/(ContA**self.m))**(1/self.n)) * self.dx
        return tempChi.cumsum()
    
    def Chi_Right(self, x):
        """
        引数x: 河川部分のｘ値
        """
        x -= (x.min()+self.xth) # 分水界のx座標を0にする。
        ContA = np.flipud(self.ka * (x**self.a))
        U = np.flipud(self.DisequilibriumTpInst.U[-1*x.shape[0]:] ) 
        tempChi = ((U/(ContA**self.m))**(1/self.n)) * self.dx
        return tempChi.cumsum()
    
    def Left_River_xz(self, yr):
        return self.CalcdTpInst.Left_River(yr)
    
    def Right_River_xz(self, yr):
        return self.CalcdTpInst.Right_River(yr)
    
    def whole_TopograpyZ(self, yr):
        return self.CalcdTpInst.whole_TopograpyZ(yr)
    
    def whole_TopograpyX(self):
        return self.CalcdTpInst.xarray()
    
class Single_PlotMaker(Calculator):
    
    def __init__(self, InitTpInst, DisequilibriumTpInst, CalcdTpInst):
        super().__init__(InitTpInst, DisequilibriumTpInst, CalcdTpInst)
        self.outpathOrigin = os.path.join(self.CalcdTpInst.dirpath, self.CalcdTpInst.TrunkName)
        self.IterNum = 20
        self.Interval = self.totalYr/self.IterNum
        plt.rcParams["xtick.minor.visible"] = True #x軸補助目盛りの追加
        plt.rcParams["ytick.minor.visible"] = True #y軸補助目盛りの追加
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams["xtick.major.width"] = 2 #X軸の主目盛の太さ
        plt.rcParams["ytick.major.width"] = 2 #Y軸の主目盛の太さ
        plt.rcParams["xtick.minor.width"] = 1.2 #X軸の副目盛の太さ
        plt.rcParams["ytick.minor.width"] = 1.2 #Y軸の副目盛の太さ
        plt.rcParams["xtick.major.size"] = 10 #X軸の主目盛の長さ
        plt.rcParams["ytick.major.size"] = 10 #Y軸の主目盛の長さ
        plt.rcParams["xtick.minor.size"] = 5 #X軸の副目盛の長さ
        plt.rcParams["ytick.minor.size"] = 5 #Y軸の副目盛の長さ
        plt.rcParams["xtick.labelsize"] = 14.0 #X軸の目盛りラベルのフォントサイズ
        plt.rcParams["ytick.labelsize"] = 14.0 #Y軸の目盛ラベルのフォントサイズ
        plt.rcParams['xtick.top'] = False #x軸の上部目盛り
        plt.rcParams['ytick.right'] = False #y軸の右部目盛り
        plt.rcParams['axes.linewidth'] = 2# 軸の線幅edge linewidth。囲みの太さ 
        
    def Left_xz_chiplot(self):
        
        initChi, initZ = self._InitSteadyChiZ_Left()
        initX = self.InitTpInst.Left_River_x() 
        U_list = [self.InitTpInst.Left_River_u(), self.DisequilibriumTpInst.Left_River_u()]
        x_list = [self.InitTpInst.Left_River_x(), self.DisequilibriumTpInst.Left_River_x()]
        U_color = ['slategray', 'navy']
        U_label = ['Steady Uplift', 'Disequilibrium Uplift']
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey = "all", squeeze=False)
        # 縦断形図に２軸目を追加して、隆起速度の変化を可視化する。
        u_ax = axes[0, 0].twinx()
        for x2, u, c, l in zip(x_list, U_list, U_color, U_label):
#             u *= 1000
            u_ax.plot(x2, u, color=c, ls="--", lw=2.0, label=l)
            
        i = 0
        for yr in range(0, self.totalYr, int(self.Interval)):
            x, z = self.Left_River_xz(yr)
            if yr == 0:
#                 print(f"diff initz: {z-initZ}")
                chi = self.Chi_Left(x)
                axes[0, 0].plot(initX, initZ, color="black", ls="-", lw=2.0, label="Init Steady state")
                axes[0, 0].plot(x, z, color="black", ls="-", label="Init Disequilibrium state")
                axes[0, 1].plot(initChi, initZ, color="black", ls="-", label=f"{yr}yr: Init Steady state")
                axes[0, 1].plot(chi, z, color=cm.gist_rainbow(i/self.IterNum), label=f"{yr}yr: Init Disequilibrium state")
                x_size = x.shape[0]
            else:
                if x.shape[0] != x_size:
                    x_size = x.shape[0]
                    chi = self.Chi_Left(x)
                    
                axes[0, 0].plot(x, z, color=cm.gist_rainbow(i/self.IterNum), label=f"{yr}yr")
                axes[0, 1].plot(chi, z, color=cm.gist_rainbow(i/self.IterNum), label=f"{yr}yr")
                    
            i += 1
                    
        axes[0, 0].set_xlabel("X [m]")
        axes[0, 0].set_ylabel("Z [m]")
        axes[0, 1].set_xlabel("Chi [m]")
        axes[0, 1].set_ylabel("Z [m]")
        u_ax.set_ylabel("U [mm/yr]")
        
        h1, l1 = axes[0, 0].get_legend_handles_labels()
        h2, l2 = u_ax.get_legend_handles_labels()
        axes[0, 0].legend(h1+h2, l1+l2, loc="upper left", ncol=3, fontsize=8)
        axes[0, 1].legend(loc="lower right", ncol=3, fontsize=8)
        
        plt.show()
        fig.savefig(self.outpathOrigin + "_Left-chiz.png", bbox_inches='tight')
        fig.clear()
        plt.close(fig)
        
    def whole_xz(self):
        
        initX, initZ = self.InitTpInst.whole_xz()
        
        U_list = [self.InitTpInst.U, self.DisequilibriumTpInst.U]
        x_list = [self.InitTpInst.x, self.DisequilibriumTpInst.x]
        U_color = ['slategray', 'navy']
        U_label = ['Steady Uplift', 'Disequilibrium Uplift']
        
        fig = plt.figure(figsize=(20, 6))
        z_ax = fig.add_subplot(111)
        # 縦断形図に２軸目を追加して、隆起速度の変化を可視化する。
        u_ax = z_ax.twinx()
        for x2, u, c, l in zip(x_list, U_list, U_color, U_label):
#             u *= 1000
            u_ax.plot(x2, u, color=c, ls="--", lw=2.0, label=l)
            
        i = 0
        x = self.whole_TopograpyX()
        for yr in range(0, self.totalYr, int(self.Interval)):
            z = self.whole_TopograpyZ(yr)
            if yr == 0:
                z_ax.plot(initX, initZ, color="black", ls="-", lw=2.0, label="Init Steady state")
                z_ax.plot(x, z, color="black", ls="-", label="Init Disequilibrium state")
            else:
                z_ax.plot(x, z, color=cm.gist_rainbow(i/self.IterNum), label=f"{yr}yr")
                    
            i += 1
                    
        z_ax.set_xlabel("X [m]")
        z_ax.set_ylabel("Z [m]")
        u_ax.set_ylabel("U [mm/yr]")
        
        h1, l1 = z_ax.get_legend_handles_labels()
        h2, l2 = u_ax.get_legend_handles_labels()
        z_ax.legend(h1+h2, l1+l2, loc="upper right", ncol=3, fontsize=8)
        
        plt.show()
        fig.savefig(self.outpathOrigin + "_wholeArea_xzu.png", bbox_inches='tight')
        fig.clear()
        plt.close(fig)        

class Animation_Plot(Calculator):
    
    """縦断形とχプロットを時系列変化にそってアニメーション表示する為のクラス"""
    
    def __init__(self, InitTpInst, DisequilibriumTpInst, CalcdTpInst, IterNum=10):
        super().__init__(InitTpInst, DisequilibriumTpInst, CalcdTpInst)
        self.fig, self.axes = plt.subplots(1, 1, figsize=(10, 6), sharey = "all", squeeze=False)
        self.u_ax = self.axes[0, 0].twinx()
        # self.xzLine, = self.axes[0, 0].plot([], [])
#         self.chizLine, = self.axes[0, 1].plot([], []) 
#         self.chi = self._SecondChi() # 摂動後のχ値（流域面積と領域サイズは不変なのでχも一連のプロットで同値）
#         self.timetext_chiz = self.axes[0, 1].text(self.chi.max(), 0, None, fontsize=10)
        self.IterNum = int(IterNum)
        self.Interval = self.totalYr/self.IterNum
        
        plt.rcParams["xtick.minor.visible"] = True #x軸補助目盛りの追加
        plt.rcParams["ytick.minor.visible"] = True #y軸補助目盛りの追加
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams["xtick.major.width"] = 1 #X軸の主目盛の太さ
        plt.rcParams["ytick.major.width"] = 1 #Y軸の主目盛の太さ
        plt.rcParams["xtick.minor.width"] = 1 #X軸の副目盛の太さ
        plt.rcParams["ytick.minor.width"] = 1 #Y軸の副目盛の太さ
        plt.rcParams["xtick.major.size"] = 10 #X軸の主目盛の長さ
        plt.rcParams["ytick.major.size"] = 10 #Y軸の主目盛の長さ
        plt.rcParams["xtick.minor.size"] = 5 #X軸の副目盛の長さ
        plt.rcParams["ytick.minor.size"] = 5 #Y軸の副目盛の長さ
        plt.rcParams["xtick.labelsize"] = 15.0 #X軸の目盛りラベルのフォントサイズ
        plt.rcParams["ytick.labelsize"] = 15.0 #Y軸の目盛ラベルのフォントサイズ
        plt.rcParams['xtick.top'] = False #x軸の上部目盛り
        plt.rcParams['ytick.right'] = False #y軸の右部目盛り
        plt.rcParams['axes.linewidth'] = 1# 軸の線幅edge linewidth。囲みの太さ 
        
    def initplot(self):
        
        self.u_ax.plot(self.InitTpInst.x, self.InitTpInst.U, color="dimgrey", ls="--")
        self.u_ax.plot(self.DisequilibriumTpInst.x, self.DisequilibriumTpInst.U, color="darkred", ls="--", label="uplift 2")
        self.axes[0, 0].plot(self.DisequilibriumTpInst.x, self.DisequilibriumTpInst.z, color="black", ls="-", label="Initial Topography")
        self.axes[0, 0].set_xlabel("x [m]", fontsize="14")
        self.axes[0, 0].set_ylabel("z [m]", fontsize="14")
        self.axes[0, 0].set_ylim(self.zmin, self.zmax+100)
        self.axes[0, 0].set_xlim(self.xmin, self.zmax+100)
        
        
        # 隆起速度の注釈とy軸設定
        utext_cd, uarrow_cd, umax = self._get_uplift_annotate_umax()
        print(f"utext_cd, uarrow_cd, umax: {utext_cd, uarrow_cd, umax}")
        self.u_ax.annotate("uplift 1", uarrow_cd[0], utext_cd[0], fontsize=12, color="dimgrey", arrowprops=dict(arrowstyle='->', fc="dimgrey"))
        self.u_ax.annotate("uplift 2", uarrow_cd[1], utext_cd[1], fontsize=12, color="darkred", arrowprops=dict(arrowstyle='->', fc="darkred"))
        self.u_ax.set_ylabel("U [mm/yr]", fontsize="14")
        self.u_ax.set_ylim(0, umax +0.2)
        
        # h1, l1 = self.axes[0, 0].get_legend_handles_labels()
        # h2, l2 = self.u_ax.get_legend_handles_labels()
        # self.axes[0, 0].legend(h1+h2, l1+l2, loc="lower right", fontsize=12, facecolor="whitesmoke", edgecolor="grey", shadow=False, framealpha=0.8, borderpad=1.2)

        # self.axes[0, 0].set_ylim(-1, self.DisequilibriumTpInst.z.max()+40)
        # self.axes[0, 0].set_xlim(-1, self.DisequilibriumTpInst.x.max()+40)
        
    def plot_func(self, frame_z):
        
        """
        frameにはz値とその時間を返すジェネレータ関数のインスタンス
    
        """
        if frame_z[2] == int(self.Interval):
            self.u_ax.cla()
        self.axes[0, 0].cla() # ax をクリア
        self.axes[0, 0].plot(frame_z[0], frame_z[1], color="black")
        self.axes[0, 0].text(self.Textcoordinate[0], self.Textcoordinate[1], f"{frame_z[2]/1e6:.2f} Myr", ha="left", fontsize=15, bbox=(dict(boxstyle="square, pad=1.2", fc="white", ec="black")))
        self.axes[0, 0].set_ylim(self.zmin, self.zmax+100)
        self.axes[0, 0].set_xlim(self.xmin, self.xmax+100)
        # h1, l1 = self.axes[0, 0].get_legend_handles_labels()
        # h2, l2 = self.u_ax.get_legend_handles_labels()
        # self.axes[0, 0].legend(h1+h2, l1+l2, loc="lower right", fontsize=12, facecolor="whitesmoke", edgecolor="grey", shadow=False, framealpha=0.8, borderpad=1.2)
        self.axes[0, 0].set_xlabel("x [m]", fontsize="14")
        self.axes[0, 0].set_ylabel("z [m]", fontsize="14")
        
    def main(self):
        ani = FuncAnimation(self.fig, self.plot_func, frames=self._xzarray_generator, init_func=self.initplot, interval=100) #, interval=50
        ani.save(self.outpath+".mp4")
        # plt.close()
        return HTML(ani.to_jshtml())
    
        
class Animation_Plot_LeftRiver(Animation_Plot):
    
    def __init__(self, InitTpInst, DisequilibriumTpInst, CalcdTpInst, IterNum):
        super().__init__(InitTpInst, DisequilibriumTpInst, CalcdTpInst, IterNum)
        self.xmin, self.zmin, self.xmax, self.zmax = self._get_xzlim()
        self.Textcoordinate = (int((self.xmax-self.xmin)/20), int((self.zmax-self.zmin)*49/50))
        self.outpath = os.path.join(self.CalcdTpInst.dirpath, "Anim_LeftRiver"+self.CalcdTpInst.TrunkName)

    def _get_xzlim(self):
        x, z = self.Left_River_xz(int(self.totalYr)-10)
        return x.min(), z.min(), x.max(), z.max()
        
    def _xzarray_generator(self):
        """
        HDFファイル内に含まれるすべての標高値データセットから特定のタイムステップでのｚarrayと
        そのインデックスを取り出すためのジェネレータ関数
        """                
        for yr in range(0, self.totalYr, int(self.Interval)):
            # print(f"_xzarray_generator yr: {yr}")
            x, z = self.Left_River_xz(yr)
            yield [x, z, yr]
            
    def _get_uplift_annotate_umax(self):
        
        """左側河川の3/4の位置のx座標に注釈を入れる為の座標値を計算するメソッド。
        u1, u2の3/4の位置のx座標で値が小さい方が下、大きい方が上に注釈が位置するようにしている。
        u1, u2のテキスト座標値, arrow座標値とumaxを返り値とする"""
        tq_x_id = int(self.InitTpInst.z.argmax() * 3 / 4) # 3/4の位置のx座標
        u1_tqx = self.InitTpInst.U[tq_x_id] # 3/4の位置のu1
        u2_tqx = self.DisequilibriumTpInst.U[tq_x_id] # 3/4の位置のu2
        # uの最大値
        if self.InitTpInst.U.max() > self.DisequilibriumTpInst.U.max():
            umax = self.InitTpInst.U.max()
        else:
            umax = self.DisequilibriumTpInst.U.max()
        
        if u1_tqx > u2_tqx:
            return ((self.InitTpInst.x[tq_x_id], u1_tqx+(0.5*(umax-u1_tqx))), (self.InitTpInst.x[tq_x_id], u2_tqx-(0.5*(u2_tqx)))), ((self.InitTpInst.x[tq_x_id], u1_tqx), (self.InitTpInst.x[tq_x_id], u2_tqx)), umax
        else:
            return ((self.InitTpInst.x[tq_x_id], u1_tqx-(0.5*(u1_tqx))), (self.InitTpInst.x[tq_x_id], u2_tqx+(0.5*(umax-u2_tqx)))), ((self.InitTpInst.x[tq_x_id], u1_tqx), (self.InitTpInst.x[tq_x_id], u2_tqx)), umax
        
class Animation_Plot_WholeArea(Animation_Plot):
    
    def __init__(self, InitTpInst, DisequilibriumTpInst, CalcdTpInst, IterNum):
        super().__init__(InitTpInst, DisequilibriumTpInst, CalcdTpInst, IterNum)
        self.fig, self.axes = plt.subplots(1, 1, figsize=(20, 6), sharey = "all", squeeze=False)
        self.u_ax = self.axes[0, 0].twinx()
        self.xmin, self.zmin, self.xmax, self.zmax = self._get_xzlim()
        self.Textcoordinate = (int((self.xmax-self.xmin)/20), int((self.zmax-self.zmin)*49/50))
        self.outpath = os.path.join(self.CalcdTpInst.dirpath, "Anim_WholeArea"+self.CalcdTpInst.TrunkName)

    def _get_xzlim(self):
        # print(f"int(self.totalYr)-10: {int(self.totalYr)-self.dt}")
        self.tp_x, z = self.whole_TopograpyX(), self.whole_TopograpyZ(int(self.totalYr-self.dt))
        return self.tp_x.min(), z.min(), self.tp_x.max(), z.max()
        
    def _xzarray_generator(self):
        """
        HDFファイル内に含まれるすべての標高値データセットから特定のタイムステップでのｚarrayと
        そのインデックスを取り出すためのジェネレータ関数
        """            
        for yr in range(0, self.totalYr, int(self.Interval)):
            # print(f"_xzarray_generator yr: {yr}")
            z = self.whole_TopograpyZ(yr)
            yield [self.tp_x, z, yr]
            
    def _get_uplift_annotate_umax(self):
        
        """全体の3/4の位置のx座標に注釈を入れる為の座標値を計算するメソッド。
        u1, u2の3/4の位置のx座標で値が小さい方が下、大きい方が上に注釈が位置するようにしている。
        u1, u2のテキスト座標値, arrow座標値とumaxを返り値とする"""
        tq_x_id = int(self.InitTpInst.x.shape[0] * 3 / 4) # 3/4の位置のx座標
        u1_tqx = self.InitTpInst.U[tq_x_id] # 3/4の位置のu1
        u2_tqx = self.DisequilibriumTpInst.U[tq_x_id] # 3/4の位置のu2
        # uの最大値
        if self.InitTpInst.U.max() > self.DisequilibriumTpInst.U.max():
            umax = self.InitTpInst.U.max()
        else:
            umax = self.DisequilibriumTpInst.U.max()
        
        if u1_tqx > u2_tqx:
            return ((self.InitTpInst.x[tq_x_id], u1_tqx+(0.7*(umax-u1_tqx))), (self.InitTpInst.x[tq_x_id], u2_tqx-(0.7*(u2_tqx)))), ((self.InitTpInst.x[tq_x_id], u1_tqx), (self.InitTpInst.x[tq_x_id], u2_tqx)), umax
        else:
            return ((self.InitTpInst.x[tq_x_id], u1_tqx-(0.7*(u1_tqx))), (self.InitTpInst.x[tq_x_id], u2_tqx+(0.7*(umax-u2_tqx)))), ((self.InitTpInst.x[tq_x_id], u1_tqx), (self.InitTpInst.x[tq_x_id], u2_tqx)), umax