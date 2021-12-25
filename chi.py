import stream_capture as sc
import file_operation as fl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
Fl = fl.File()
Kw = fl.Kwargs()

# %pdb
class chi:
    def __init__(self, **kwargs):
        self.A0 = kwargs['A0']
        self.U0 = kwargs['U0']
        self.dx = kwargs['dx']
        self.dt = kwargs['dt']
        self.nmax = kwargs['nmax']
        self.n = kwargs['n']
        self.m = kwargs['m']
        self.degree = kwargs['degree']
        self.kb = kwargs['kb']
        self.ka = kwargs['ka']
        self.a = kwargs['a']
        self.U = kwargs['U']
        self.FileWords = kwargs['FileWords']
        self.Fpath = kwargs['Fpath']
    
    def generate_x(self, z):
        
        """HDF5ファイルにはxを保存していないので、z, dxから復元する。"""
        
        dx = self.dx
        jmax = (len(z)-1) * dx
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        x = np.arange(0, jmax+dx, dx)
        
        return x
        
    def calc_chi(self, ContA):
        
        """χパラメータを計算する。流域面積はインデックス０が河口側になるように引数に渡す"""
        
        n = self.n
        m = self.m
        U = self.U
        dx = self.dx
        A0 = self.A0
        U0 = self.U0
        
        # 0で割る事はできないので分水界の流域面積を1e-10に設定する。
#         ContA[-1] = 1e-10
#         ContA = np.flipud(ContA)
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        integrated = ((((A0/ContA)**(m)) * (U/U0)) ** (1/n)) * dx
        Chi = integrated.cumsum()
        
        return Chi

class main:
    
    def __init__(self):
        pass
    
    def calc(self, **chi_param):
        
        filename = Fl.get_hdffile(**chi_param)[0][1]
        param_dict_h5 = Fl.param_dict_from_h5file(**chi_param)
        param_dict_h5.update(chi_param)
        param_dict_h5 = Kw.change_to_numeric(**param_dict_h5)
        dataset_names, group_names = Fl.datasetname_from_h5file(**chi_param)
        sorted_datasets, head_nums = Fl.sorting_dataset(dataset_names[0], group_names[0], **param_dict_h5)
        plotNum = chi_param['plotNum']
               
        chi_ins = chi(**param_dict_h5)
        Clac = sc.Calc(**param_dict_h5)
        
        Chi_left = []
        Chi_right = []
        zlr_list = []
        zrr_list = []
        calcd_yr = []
        sorted_datasets_len = len(sorted_datasets)
        print(sorted_datasets_len)
        for i in range(0, len(sorted_datasets), int(sorted_datasets_len/plotNum)):
            
            dataset = sorted_datasets[i]
            head_num = head_nums[i]
            calcd_yr.append(head_num)
            print("i_currently dataset name : {}, {}".format(i, dataset))
            z = Fl.z_from_h5file(fname=filename, dataset_name=dataset)[0] # 各データセットの最初の年代のみχを計算
#             z_list.append(z) 
            x = chi_ins.generate_x(z)
            zlr, _, zrr, ContA_left, ContA_right, _ = Clac.split_to_three_zone(x, z)
            zrr = np.flipud(zrr)
            zlr_list.append(zlr)
            zrr_list.append(zrr)
            ContA_right = np.flipud(ContA_right)
            Chi_left.append(chi_ins.calc_chi(ContA_left)) 
            Chi_right.append(chi_ins.calc_chi(ContA_right)) 
        
        del ContA_left, ContA_right 
        return Chi_left, Chi_right, zlr_list, zrr_list, calcd_yr
    
    def main(self, **chi_param):
        
        plot = plot_chi()
        Chi_left, Chi_right, zlr_list, zrr_list, calcd_yr = self.calc(**chi_param)
        fpath = Fl.fname_for_chiplot(**chi_param)
        plot.plot_erea(Chi_left, Chi_right, zlr_list, zrr_list, calcd_yr, fpath)
        
        del Chi_left, Chi_right, zlr_list, zrr_list, calcd_yr
    
class plot_chi:
    
    def __init__(self):
        pass
    
    def plot_erea(self, chi_left, chi_right, zlr_list, zrr_list, calcd_yr, fpath):
        
        zlmax = zlr_list[0].max()
        zrmax = zrr_list[0].max()
        if zlmax > zrmax:
            zmax = zlmax
        else:
            zmax = zrmax
        
        fig = plt.figure(figsize = (7, 7),dpi = 100)
#         plt.rcParams["font.size"] = 22
        fig.patch.set_facecolor('white') 
        
        for i in range(len(calcd_yr)):
            if i == 0:
                plt.plot(chi_right[i][:-1], zrr_list[i][:-1], color=cm.autumn(i/len(calcd_yr)), linestyle = "dashed", lw = 3, label="right_river"+"_"+str(calcd_yr[i])+"yr")
                plt.plot(chi_left[i][:-1], zlr_list[i][:-1], color=cm.winter(i/len(calcd_yr)), lw = 2, label="left_river"+"_"+str(calcd_yr[i])+"yr")
            else:
                plt.plot(chi_left[i][:-1], zlr_list[i][:-1], color=cm.winter(i/len(calcd_yr)), lw = 2, label=str(calcd_yr[i])+"yr")
                plt.plot(chi_right[i][:-1], zrr_list[i][:-1], color=cm.autumn(i/len(calcd_yr)), linestyle = "dashed",lw = 3) #label=str(calcd_yr[i])+"yr"
                
        
        plt.grid(color="black", linestyle="dashed", linewidth=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.xlabel("χ")
        plt.ylabel("z")
#         plt.xlim()
        plt.ylim(0, 650)
        fig.savefig(fpath + ".png", bbox_inches='tight')
        fig.clear()
        plt.close(fig)
        
    
        
        
if __name__ == "__main__":
    
    chi_param = {
    "A0" : 1, # 流域面積規格化定数
    "U0" : 3.1e-11, # 隆起速度規格化定数 1mm/yr = 3.1e-12
    "Fpath" : r"C:\Users\miyar\NUMERICAL FLUID COMPUTATION METHOD\result_img", # 結果を保存するファイルの最上位のディレクトリパス
    "z_H5file_name" : "Elevation_0_(initial topograpy, erea size) == ('variable_erea_size', 4200)", # 分析対象のH5file名。allの場合はFpathに含まれるすべての
    "plotNum" : 5, #χプロットのトータルプロット数
    }
    

    main.main(**chi_param)
    
