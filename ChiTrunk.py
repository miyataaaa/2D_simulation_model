import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from Ver3_Trunk import ZHDFreader

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
        self.U = self.df.iloc[:, 3].to_numpy(dtype=np.float64)
        
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
        
    def _get_Param(self):
    
        self.Parampath = os.path.join(self.dirpath, self.TrunkName+"_param_dict.h5")
        self.paramDf = pd.read_hdf(self.Parampath, key="param_dict", index=False)
        self.dt = int(self.paramDf['dt'].values[0])
        self.nmax = int(self.paramDf['nmax'].values[0])
        self.n = float(self.paramDf['n'].values[0])
        self.m = float(self.paramDf['m'].values[0])
        self.kb = float(self.paramDf['kb'].values[0])
        self.DatasetNum = int(self.paramDf['DatasetNum'].values[0])
        
    def zarray(self, dataset_pointer):
        return self.reader.zarray(dataset_pointer)
    
    def xarray(self):
        return self.reader.xarray()
        
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
        self.dx = 10       

    def _InitChi(self):
    
        ContA = self.InitTpInst.ContA
        U = self.InitTpInst.U * 1000 # タイムスケールをm/yr -> mm/yrに変更
        tempChi = ((U/(ContA**self.m))**(1/self.n)) * self.dx       
        return tempChi.cumsum()
        
    def _SecondChi(self):
        
        ContA = self.DisequilibriumTpInst.ContA
        U = self.DisequilibriumTpInst.U * 1000 # タイムスケールをm/yr -> mm/yrに変更
        tempChi = ((U/(ContA**self.m))**(1/self.n)) * self.dx
        return tempChi.cumsum()
    
    def _InitChi_onlyCA(self):

        ContA = self.InitTpInst.ContA
        # U = self.InitTpInst.U * 1000 # タイムスケールをm/yr -> mm/yrに変更
        tempChi = ((1/(ContA**self.m))**(1/self.n)) * self.dx       
        return tempChi.cumsum()
        
    def _SecondChi_onlyCA(self):
        
        ContA = self.DisequilibriumTpInst.ContA
        # U = self.DisequilibriumTpInst.U * 1000 # タイムスケールをm/yr -> mm/yrに変更
        tempChi = ((1/(ContA**self.m))**(1/self.n)) * self.dx
        return tempChi.cumsum()
    
    def _CalcdTpInst_Z(self, dataset_pointer):
        return self.CalcdTpInst.zarray(dataset_pointer)
    
class Single_PlotMaker(Calculator):
    
    def __init__(self, InitTpInst, DisequilibriumTpInst, CalcdTpInst):
        super().__init__(InitTpInst, DisequilibriumTpInst, CalcdTpInst)
        self.outpahtOrigin = os.path.join(self.CalcdTpInst.dirpath, self.CalcdTpInst.TrunkName)
        
    def chiplot(self):
        
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
 
        initChi = self._InitChi_onlyCA()
        secondChi = self._SecondChi_onlyCA()
        
        fig = plt.figure(figsize=(20, 14))
        ax_initxz = fig.add_subplot(221)
        ax_initchi = fig.add_subplot(222, sharey=ax_initxz)
        ax_secondxz = fig.add_subplot(223)
        ax_secondchi = fig.add_subplot(224, sharey=ax_secondxz)
        
        # 定常初期縦断形とχプロット
        ax_initxz.plot(self.InitTpInst.x, self.InitTpInst.z, color="black")
        ax_initchi.plot(initChi, self.InitTpInst.z, color="black")
        
        # 摂動後の縦断形とχプロット
        interval = int(self.nmax/5)
        zarray_List = []
        for i in range(self.DatasetNum):
            zarray = self._CalcdTpInst_Z(i)
            for j in range(0, self.nmax, interval):
                zarray_List.append(zarray[j])
        
        IterNum = len(zarray_List)
        z_max = -20
        # 傾きの比較の為に、摂動後のχプロットに摂動前のχプロットを描画
        ax_secondchi.plot(initChi, self.InitTpInst.z, color="black", label="Before perturbation")
        for i in range(IterNum):
            if i % (IterNum/20) == 0:
                if zarray_List[i].max() > z_max:
                    z_max = zarray_List[i].max()
                ax_secondxz.plot(self.DisequilibriumTpInst.x, zarray_List[i], color=cm.gist_rainbow(i/IterNum), label=f"{(i+1)*interval*self.dt}yr")
                ax_secondchi.plot(secondChi, zarray_List[i], color=cm.gist_rainbow(i/IterNum), label=f"{(i+1)*interval*self.dt}yr")
            else:
                continue
#                 ax_secondxz.plot(self.DisequilibriumTpInst.x, zarray_List[i], color=cm.gist_rainbow(i/IterNum))
#                 ax_secondchi.plot(secondChi, zarray_List[i], color=cm.gist_rainbow(i/IterNum))
        ax_initchi.set_xlabel("Chi [m]")
        ax_initchi.set_ylabel("Z [m]")
        ax_initxz.set_xlabel("X [m]")
        ax_initxz.set_ylabel("Z [m]")
        
        ax_secondchi.set_xlabel("Chi [m]")
        ax_secondchi.set_ylabel("Z [m]")
        ax_secondchi.legend(loc="lower right", ncol=4, fontsize=7)
        chirange = secondChi.max()-secondChi.min()
        ax_secondchi.set_xlim(secondChi.min()-(chirange/20), secondChi.max()+(chirange/20))
        ax_secondchi.set_ylim(-10, z_max+50)
        
        ax_secondxz.set_xlabel("X [m]")
        ax_secondxz.set_ylabel("Z [m]")
        ax_secondxz.legend(loc="lower right", ncol=4, fontsize=7)
        
        plt.show()
        fig.savefig(self.outpahtOrigin + ".png", bbox_inches='tight')
        fig.clear()
        plt.close(fig)
        
class Animation_Plot(Calculator):
    
    """縦断形とχプロットを時系列変化にそってアニメーション表示する為のクラス"""
    
    def __init__(self, InitTpInst, DisequilibriumTpInst, CalcdTpInst):
        super().__init__(InitTpInst, DisequilibriumTpInst, CalcdTpInst)
        self.outpahtOrigin = os.path.join(self.CalcdTpInst.dirpath, "Anim"+self.CalcdTpInst.TrunkName)
        self.fig, self.axes = plt.subplots(1, 2, figsize=(20, 14), sharey = "all", squeeze=False)
        self.xzLine, = self.axes[0, 0].plot([], [])
        self.chizLine, = self.axes[0, 1].plot([], []) 
        self.chi = self._SecondChi() # 摂動後のχ値（流域面積と領域サイズは不変なのでχも一連のプロットで同値）
        self.timetext_xz = self.axes[0, 0].text(self.DisequilibriumTpInst.x.max(), 0, None, fontsize=10)
        self.timetext_chiz = self.axes[0, 1].text(self.chi.max(), 0, None, fontsize=10)
        self.timeinterval = int(self.nmax/5)
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
        
#     def initplot(self):
#         self.axes[0].set_ylim(-1, self.DisequilibriumTpInst.z.max()*40)
        
    def plot_func(self, frame_z):
        
        """
        frameにはz値とその時間を返すジェネレータ関数のインスタンス
    
        """
#         for ax in self.axes:
            
#             ax.cla() # ax をクリア
        print(frame_z[0].shape, frame_z[1])
        TotalNum = self.DatasetNum*self.timeinterval
        # データをセット
        # print(f"type xzLine: {type(self.xzLine)}")
        self.xzLine.set_data(self.DisequilibriumTpInst.x, frame_z[0])
        self.chizLine.set_data(self.chi, frame_z[0])
        # 色をセット
        self.xzLine.set_color(cm.jet(frame_z[1]/TotalNum))
        self.chizLine.set_color(cm.jet(frame_z[1]/TotalNum))
        # テキストをセット
        self.timetext_xz.set_text(str(frame_z[1]*self.dt)+"yr")
        self.timetext_chiz.set_text(str(frame_z[1]*self.dt)+"yr")
        
        return self.fig,
        
    def _zarray_generator(self):
        """
        HDFファイル内に含まれるすべての標高値データセットから特定のタイムステップでのｚarrayと
        そのインデックスを取り出すためのジェネレータ関数
        """
        for i in range(1):
            z = self._CalcdTpInst_Z(i)
            for j in range(0, self.nmax, self.timeinterval):
                yield [z[j], (i*self.nmax)+j]
                
    def main(self):
        ani = FuncAnimation(self.fig, self.plot_func, frames=self._zarray_generator(), interval=50)
        # plt.close()
        HTML(ani.to_jshtml())