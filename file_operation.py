# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:49:09 2021

@author: miyar
"""
import sys
import h5py
import os
import pandas as pd
import numpy as np
from memory_profiler import profile
import re

class File:
    
    def __init__(self):
        pass
    
    def fname_for_plot(self, i, **kwargs):
        
        """領域全体の時間経過プロット、領域別プロット、侵食速度プロット用のパス＋ファイル名生成関数。引数のiは
        param_dictのiterNumを想定している。"""
        
        FileWords = kwargs['FileWords']
        Fpath = kwargs['Fpath']
        fname = str(i) + "_" + FileWords
        nameTime = Fpath + "\\" + "Time_" + fname #計算結果を時系列変化
        nameLocation = Fpath + "\\" + "Location_" + fname #計算結果を河川・斜面領域に分けてグラフ化する時のファイル名
        nameErate = Fpath + "\\" + "Erate_" + fname #計算結果を河川・斜面領域に分けてグラフ化する時のファイル名
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return nameTime, nameLocation, nameErate

    def fname_for_hdf(self, i, **kwargs):
        
        """標高値の時系列データを格納するHDF5ファイル用のファイル名を自動生成するための関数。引数のiはparam_setの要素数を
        想定している。（正確には、main関数内での繰り返す部分でのi）"""
        
        FileWords = kwargs['FileWords']
        Fpath = kwargs['Fpath']
        fname = "_" + str(i) + "_" + FileWords
        total_fname = Fpath + '\\' + "Elevation" + fname
        
        return total_fname
    
    def fname_for_analysis(self, **kwargs):
        
        """分水界移動速度などを解析した結果を解析用の新たなHDFファイルに保存するためのパスを生成する関数。"""
        
        FileWords = kwargs['FileWords']
        Fpath = kwargs['Fpath']
        fname = "Analysis" + "_" + FileWords + ".h5"
        total_fname = Fpath + '\\' + fname
        
        return total_fname
    
    def fname_for_chiplot(self, **kwargs):
        
        Fpath = kwargs['Fpath']
        z_H5file_name = kwargs['z_H5file_name']
        chiplot = "Chiplot" + "_" + z_H5file_name
        for curDir, dirs, files in os.walk(Fpath):
                for file in files:
                    if file == z_H5file_name + ".h5":
                        path_for_chi = os.path.join(curDir, chiplot)
        
        return path_for_chi
    
    def fname_for_Analysisplot(self, **kwargs):
        
        Fpath = kwargs['Fpath']
        z_H5file_name = kwargs['z_H5file_name']
        analysisplot = "Analysisplot" + "_" + z_H5file_name
        for curDir, dirs, files in os.walk(Fpath):
                for file in files:
                    if file == z_H5file_name + ".h5":
                        path_for_analysis = os.path.join(curDir, analysisplot)
        
        return path_for_analysis
    
    def groupname_for_hdf(self, **kwargs):
        
        """HDFファイル内でのGroup名を自動生成するための関数。トータル計算時間で区別する。主に標高値を保存するHDFファイルで使う。"""
        
        nmax = kwargs['nmax']
        iterNum = kwargs['iterNum']
        total_time = (nmax / 365) * iterNum
        Group_name = str(total_time) + "yr"
        
        return Group_name
    
    def datasetname_for_hdf(self, i, **kwargs):
        
        """HDFファイル内でのdatasetの名前を自動生成するための関数。引数kはparam_dict内のiterNum1,
        iはparam_setの要素数を想定している。主に標高値を保存するHDFファイルで使う。"""
        
        nmax = kwargs['nmax']
        dataset_name = str((nmax/365) * i) + "~" + str((nmax/365) * (i+1)) + "yr"
        
        
        return dataset_name
    
    def save_to_hdf(self, z, i, k, **kwargs):
        
        """
        標高値の計算結果を保存するのに使う。
        """
        
        fname = self.fname_for_hdf(i, **kwargs) + ".h5"
        group_name = self.groupname_for_hdf(**kwargs)
        dataset_name = self.datasetname_for_hdf(k, **kwargs)
#         param_dict = kwargs
        
        print(fname, "\n", group_name, "\n", dataset_name)
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
#         print(type(kwargs))
        
#         df_param_dict = pd.DataFrame(kwargs.values(), index=kwargs.keys(), columns=['value'], dtype=str)
#         print(df_param_dict)
#         df_param_dict.to_hdf(fname, key="param_dict")
        
        if k == 0:
            with h5py.File(fname, "a") as f:
                g = f.create_group(group_name)
                g.create_dataset(dataset_name, data=z, compression="gzip", compression_opts=4)
#                 g.create_dataset("param_dict", data=param_dict, compression="gzip", compression_opts=4)
        
        else:
            with h5py.File(fname, "a") as f:
                g = f[group_name]
                print(type(g))
                g.create_dataset(dataset_name, data=z, compression="gzip", compression_opts=4)
                
    def do_save(self, fname, group_names=[], values=[]):
        
        """
        ~引数~
        fname -> 拡張子付きの、絶対パス
        group_names -> hdfファイル内に生成するグループ名のリスト
        values -> group_namesに対応する値
        """
        
        is_file = os.path.isfile(fname)
        dataset_name = "value"
        exist_id = []
#         group_names_id = [i for i in range(len(group_names))]
        if is_file:
            print("is_file: True")
            _, exist_gname = self.datasetname_from_h5file_all(fname)
            for gname in exist_gname:
                for j in range(len(group_names)):
                    if gname == group_names[j]:
                        exist_id.append(j)
#                         group_names_id.ramove(j)
            
            # 存在しているgraupを上書きする（一旦消去）。
            if len(exist_id) > 0:
                with h5py.File(fname, "a") as f:
                    for i in range(len(exist_id)):
                        del f[group_names[exist_id[i]]]
            else:
                print("there are no same name group")
                with h5py.File(fname, "a") as f:
                    for group_name, value in zip(group_names, values):
                        print("now saving {}".format(group_name))
                        g = f.create_group(group_name)
                        g.create_dataset(dataset_name, data=value, compression="gzip", compression_opts=4)     
            
        else:
            with h5py.File(fname, "a") as f:
                for group_name, value in zip(group_names, values):
                    print("group_name: {}\n value: {}".format(group_name, value.shape))
                    g = f.create_group(group_name)
                    g.create_dataset(dataset_name, data=value, compression="gzip", compression_opts=4)
                    
                
    def save_analysis(self, group_names=[], values=[], **kwargs):
        
        """
        現在予定しているgroup_names
        ['Divide height', 'Divide position', 
        'Ex Divide position', 'Ex Divide height', 'Ex
        time step', 'Divide migration rate', 
        'Head Erate diff', 'Mean Erate diff']
        """
        fname = self.fname_for_analysis(**kwargs)
        self.do_save(fname, group_names=group_names, values=values)
        
#     def overwrite_save_analysis(self, fname, group_names=[], values=[]):
        
        

    def get_hdffile(self, **kwargs):
        
        """引数辞書(param_dict)に指定したFpathより下の階層にあるh5fileを絶対パスで取得する関数。もし、引数辞書のz_H5file_nameを
        allにした場合はFpathより下の階層のすべてのh5fileを取得する。返り値は同じ階層にあるz_h5fileとparam_dict.h5ファイルを要素
        とするタプルを要素に持つリスト。特定のファイル名を指定した場合は、そのファイルが含まれる階層にある標高値hdfファイルor解析値hdfファイルとparamhdfファイル
        の絶対パスを要素とするタプルを要素にもつリストを返り値として返す"""
        
        Fpath = kwargs['Fpath']
        H5file_name = kwargs['z_H5file_name']
#         print("Fpath: {}\nhdffile: {}".format(Fpath, H5file_name))
        z_H5files = [] # 絶対パスを含むz_H5fileのリスト
        param_dict_h5files = [] # 絶対パスを含むz_param_dictのリスト
        analysis_h5files = []
        hdf_files = [] # 同じ階層にあるすべてのhdfファイルのタプルを要素に持つリスト
        path_for_analysis = ""
        i = 0
        if H5file_name == "all":
#             print("all")
            # すべての標高値h5fileを取得する場合
            for curDir, dirs, files in os.walk(Fpath):
                for file in files:
                    if file.endswith(".h5"):
                        if file == "param_dict.h5":
                            path_for_param = os.path.join(curDir, file)
                            param_dict_h5files.append(path_for_param)
                        elif file.startswith("Elevation"):
                            path_for_z = os.path.join(curDir, file)
                            z_H5files.append(path_for_z)
                        elif file.startswith("Analysis"):
                            path_for_analysis = os.path.join(curDir, file)
                            analysis_h5files.append(path_for_analysis)
                        else:
                            pass

#             from IPython.core.debugger import Pdb; Pdb().set_trace()
            if len(path_for_z) == len(path_for_analysis):
                for i in range(len(param_dict_h5files)):
                    hdf_files.append((param_dict_h5files[i], z_H5files[i], analysis_h5files[i]))
            else:
                for i in range(len(param_dict_h5files)):
                    hdf_files.append((param_dict_h5files[i], z_H5files[i]))
#                 print(hdf_files[i][0])
#                 print(hdf_files[i][1], "\n")
            
        else:
#             print("now: {}".format(Fpath))
            # 特定のファイルを指定している場合
            i = 0
            while len(hdf_files) <= 0:
#                 print(Fpath)
                try:
                    if i == 1:
                        raise Exception("there is no file !!!")
                        
                    for curDir, dirs, files in os.walk(Fpath):
                        for file in files:
                            if file == H5file_name + ".h5":
                                if H5file_name.startswith('Elevation'):
                                    path_for_param = os.path.join(curDir, "param_dict.h5")
                                    path_for_z = os.path.join(curDir, file)
                                    path_for_analysis = "none"
                                    hdf_files.append((path_for_param, path_for_z, path_for_analysis))
                                if H5file_name.startswith('Analysis'):
                                    path_for_param = os.path.join(curDir, "param_dict.h5")
                                    path_for_z = "none"
                                    path_for_analysis = os.path.join(curDir, file)
                                    hdf_files.append((path_for_param, path_for_z, path_for_analysis))
                    #もし、今探している階層になければ1つ上の階層に移動
    #                 print(Fpath)
                    Fpath = os.path.dirname(Fpath)
                    i += 1
                except Exception as e:
                    print(e)
                    break
           
        return hdf_files
    
    def get_analysis_file(self, **kwargs):
        
        hdf_files = self.get_hdffile(**kwargs)
        fname = hdf_files[0][-1]
        
        return fname
        
    def param_dict_from_h5file(self, **kwargs):

        """特定の階層のHDFファイルから引数辞書(param_dict)を取得する関数"""

        fname_param = self.get_hdffile(**kwargs)[0][0]
        param_dict_df = pd.read_hdf(fname_param, 'param_dict')
        new_param_dict = param_dict_df.to_dict()['value']
        
        return new_param_dict
    
#     @profile
    def datasetname_from_h5file(self, **kwargs):
        
        """特定の階層の特定のHDFファイルからdataset名を取得する関数。
        param_dictのz_H5file_nameを使用。１つのファイルのみを解析したいときに使う。"""
        
        fname_z = self.get_hdffile(**kwargs)[0][1]
        
        with h5py.File(fname_z, "r") as f:
            group_names = []
            dataset_names = []
            for group in f:
                group_names.append(group)
                
            for i in range(len(group_names)):
                dataset_names.append([])
                for value in f[group_names[i]].values():
#                     print(value.name)
                    dataset_names[i].append(value.name)
        
        return dataset_names, group_names
    
    def datasetname_from_h5file_all(self, fname_z):
        
        """特定の階層の特定のHDFファイルからdataset名を取得する関数。
        ある階層より下にあるすべての標高値hdffile解析するときにget_hdffile関数と
        組み合わせて使用する。
        
        ~usage~
        引数に解析対象のhdffileのpathを持たせる。
        ~end~~
        """
  
        with h5py.File(fname_z, "r") as f:
            group_names = []
            dataset_names = []
            for group in f:
                group_names.append(group)
                
            for i in range(len(group_names)):
                dataset_names.append([])
                for value in f[group_names[i]].values():
#                     print(value.name)
                    dataset_names[i].append(value.name)
        
        return dataset_names, group_names
    
    def z_from_h5file(self, fname="", dataset_name=""):
        
        """特定のHDFファイルの特定のdatasetから標高値を取り出す関数"""

        with h5py.File(fname, "r") as f:
            z = np.array(f[dataset_name])

        return z
    
    def value_from_h5file(self, fname="", dataset_name=""):
        
        with h5py.File(fname, "r") as f:
            value = np.array(f[dataset_name])

        return value
    
    def z_from_datasets(self, fname="", datasets_list=[], start=0, stop=-1):
        
        """
        特定のHDFファイル内の複数のデータセットから標高値を取り出して、１つの配列にして
        返す。
        
        ~usage~
        1.fnameにファイル名を含むパスを指定
        2.datasetlistに、昇順にソートしたデータセット名を要素に持つリストを指定
        3.start, stopにソートしたデートセット内のどこからどこまでの標高値を抽出するか指定
        ~end~
        """
        
        with h5py.File(fname, "r") as f:
            for i in range(start, stop):
                if i == start:
                    z = np.array(f[datasets_list[i]])
                else:
                    z_i = np.array(f[datasets_list[i]])
                    z = np.concatenate([z, z_i])
                    
        return z
        
        
    def sorting_dataset(self, dataset_names, group_name, **kwargs):
        
        """データセットを時系列順にソートするための関数"""
        
        time_interval = int(kwargs['nmax']) / 365
        # gropu名を含んだパスから、dataset名のみを取り出す
        only_dataset = []
        for name in dataset_names:
            only_dataset.append(name[len(group_name)+2:])
        
        # dataset名（例：1000.0yr~2000yr）から、先頭の数字を取り出す（ここでは1000.0）
        head_nums = []
        for yr in only_dataset:
            m = re.match("\d+\.0", yr)
            head_num = float(m.group())
            head_nums.append(head_num)
        
        # 昇順でソート
        head_nums.sort()
        
        # ソートしたリストをgroup名を含むパス名に変換
        sorted_datasets = []
        for num in head_nums:
            sorted_dataset = "/" + group_name + "/" + str(num) + "~" + str(num+time_interval) + "yr"
            sorted_datasets.append(sorted_dataset)
            
        return sorted_datasets, head_nums
    

class Kwargs:
    
#     import os
    
    def __init__(self):
        pass
    
    def reservation_list(self, **kwargs):
        
        """パラメータセット内にリスト型式でパラメータがセットされていた場合に、パラメータ組み合わせをタプル化した物を要素とするリストを返す関数
        もし、3つ以上のパラメータがlistで渡されていた場合はエラーになるようにする。"""
        
        
        # 予約リストから変数名と値を別々に取り出して格納するためのリスト
        param_name = []
        param_value = []
        param_set = []
        reservation_paramNum = 0
        for key, value in kwargs.items():
            if isinstance(value, list):
#                 print(key, type(value), "\n")
                reservation_paramNum += 1
                param_name.append(key)
                param_value.append(value)
#         print(param_value, "\n", reservation_paramNum)
        
        if reservation_paramNum < 3:
            
            for i in param_value[0]:
                for j in param_value[1]:
                    param_set.append((i, j))
        
        else:
            print(param_name, param_value)
            print("パラメータの設定が不正です。変化させるパラメータは2つ以下にしてください。")
            sys.exit()
            
        return param_name, param_set
    
    def update_kwargs(self, i, param_name, param_set, **originalkwargs):
        
        """パラメータの組みわせごとに、シミュレーションパラメータを更新して新たなパラメータセットを返す関数。また、パラメータの組みわせごとにディレクトリも内部で生成する。
        main関数内の繰り返し部分で使う事が前提なので、引数iはイテレーション回数を示すもの。main関数内を参照してください。内部でFilePathも書き変えているので注意。"""
        
        Fpath = originalkwargs['Fpath']
        updated_kwargs = originalkwargs.copy()
        updated_kwargs[param_name[0]] = param_set[i][0]
        updated_kwargs[param_name[1]] = param_set[i][1]
        updated_kwargs['FileWords'] = "(" + param_name[0] + ", " + param_name[1] + ")" + " == " + str(param_set[i])
        new_parent_path = Fpath + "\\" + param_name[0] + "-" + param_name[1] + "_patterns" + "\\" + str(param_set[i][0]) + "_" + str(param_set[i][1])
        os.makedirs(new_parent_path, exist_ok=True)
        updated_kwargs['Fpath'] = new_parent_path

        return updated_kwargs
    
    def isfloat(self, string):
        
        """"文字列が小数点を含む数値かどうかを判定する関数"""
        
        try:
            float(string)
            return True
        except ValueError:
            return False
        
    def change_to_numeric(self, **kwargs):
        
        """文字型に変換してHDFファイルに保存したパラメータ辞書内の数値文字を数値型に変換する"""
        changed_kwargs = kwargs.copy()
        for key, value in kwargs.items():
            if isinstance(value, str):
                if self.isfloat(value):
                    changed_kwargs[key] = float(value)
        
        return changed_kwargs
    
    def change_Elevation_to_Analysis(self, **kwargs):
        
        Analysis_kwargs = kwargs.copy()
        file = kwargs['z_H5file_name']
        file = re.sub(r'Elevation_\d', 'Analysis', file)
        Analysis_kwargs['z_H5file_name'] = file
        
        return Analysis_kwargs
        
#=================================================================================================================
if __name__ == "__main__":
    
    param_dict = {
    "dx" : 10, #空間刻み幅 [m]
    "dt" : 3600*24, #1タイムステップ当たりの時間 [s]
    "nmax": 365*10, #時間積分の繰り返し数
    "iterNum": 3, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
    "n" : 1.95,  # ストリームパワーモデルにおける急峻度指数 n
    "m" : 1, #　ストリームパワーモデルにおける流域面積指数 m
    "kd" : 2e-12, #拡散係数 [m^2/s]
    "degree" : 50, # 安息角 [度数]
    "kb" : 2.6e-15, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5)
    "ka" : 8700, #逆ハックの法則係数
    "a" : 0.73, #逆ハックの法則における指数：流域面積と河川長の関係を決定する 
    "th_ContA" : 300000, #チャネル形成に必要な最低流域面積 (threshold Contributory Area)[m^2] QGISだと3000ピクセルに相当
    "U" : 1.58e-11, # 隆起速度 [m/s] 0.8mm/yr = 2.54e-11, 0.5mm/yr = 1.58e-11,  0.1mm/yr = 3.1e-12
    "initial topograpy" : "eacharea_degree", # wholearea_degree, eacharea_degree
    "initial degree" : 11.86,  # 初期地形での勾配
    "initial divide" : 0.45,  # 初期地形での分水界位置を領域全体に対する割合で指定。左側境界が０で右側境界が1とする。
    "initial Ldegree" : [21.8], # 初期地形左側領域での勾配
    "initial Rdegree" : [11.8], # 初期地形右側領域での勾配
    "FileWords" : "something", # ファイル名に入れる任意のワード
    "Fpath" : r"C:\Users\miyar\NUMERICAL FLUID COMPUTATION METHOD\result_img", # 結果を保存するファイルのディレクトリパス
    "z_H5file_name" : "Elevation_0_(initial Ldegree, initial Rdegree) == (15.8, 11.8)" # 分析対象のH5file名。allの場合はFpathに含まれるすべての
        
}
    
    Fl = File(**param_dict)
    dataset_names = Fl.datasetname_from_h5file()
    
    for i in range(5):
        z = Fl.z_from_h5file(dataset_names[i])
        print(type(z))
        print(z.shape)
        print(z.nbytes)
 
    
#     z = Fl.z_from_h5file(dataset_names[0])
#     z_1 = Fl.z_from_h5file
    
#     print(type(z))
#     print(z.shape)
#     print(z.nbytes)
 
    
             
                         
    
    
