# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 11:25:08 2021

@author: miyar
"""

import Ver2_main_not_exhandling as m

#import time
#import os
import multiprocessing 

def list_of_dict(dict1, dict2): #, dict3
    
    return [dict1, dict2]  #, dict3

def main_wrapper(kwargs):
    return m.main(**kwargs)

#def main(**kwargs):
    #print(kwargs['dx'] * 2)
    #print(kwargs)
    #print(os.getpid()
    #deg = kwargs['initial degree'] * 2
    #return deg
    
if __name__ == '__main__':
    
    p = multiprocessing.Pool(2)
    param_dict_1 = {
        "dx" : 10, #空間刻み幅 [m]
        "dt" : 3600*24, #1タイムステップ当たりの時間 [s]
        "nmax": 365*1000, #時間積分の繰り返し数
        "iterNum": 300, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
        "n" : 1,  # ストリームパワーモデルにおける急峻度指数 n
        "m" : 0.5, #　ストリームパワーモデルにおける流域面積指数 m
        "kd" : "-", #拡散係数 [m^2/s]
        "degree" : 26, # 安息角 [度数]
        "kb" : 3.3e-13, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5)
        "ka" : 2703, #逆ハックの法則係数
        "a" : 0.91, #逆ハックの法則における指数：流域面積と河川長の関係を決定する 
        "xth" : 600, #チャネル形成に必要な最低流域面積 (threshold Contributory Area)[m^2] QGISだと3000ピクセルに相当
        "U" : 1.58e-11, # 隆起速度 [m/s] 0.8mm/yr = 2.54e-11, 0.5mm/yr = 1.58e-11,  0.1mm/yr = 3.1e-12
        "initial topograpy" : ["natureQGIS"], # wholearea_degree, eacharea_degree, natureQGIS
        "initial degree" : "-",  # 初期地形での勾配
        "initial divide" : "-",  # 初期地形での分水界位置を領域全体に対する割合で指定。左側境界が０で右側境界が1とする。
        "initial Ldegree" : "-", # 初期地形左側領域での勾配
        "initial Rdegree" : "-", # 初期地形右側領域での勾配
        "diff baselevel" : "-", # 初期地形でのベースレベル差
        "FileWords" : "something", # ファイル名に入れる任意のワード
        "Fpath" : r"G:\miyata\Simulation Model\result", # 結果を保存するファイルのディレクトリパス
        "z_H5file_name" : "-", # 分析対象のH5file名。allの場合はFpathに含まれるすべての
        "only_one_param" : "-",
        "xzExfile" : r"G:\miyata\Simulation Model\result\NY1.xlsx",
        "nature" : ["NY1"]
    }
    
    param_dict_2 = {
        "dx" : 10, #空間刻み幅 [m]
        "dt" : 3600*24, #1タイムステップ当たりの時間 [s]
        "nmax": 365*1000, #時間積分の繰り返し数
        "iterNum": 300, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
        "n" : 1,  # ストリームパワーモデルにおける急峻度指数 n
        "m" : 0.5, #　ストリームパワーモデルにおける流域面積指数 m
        "kd" : "-", #拡散係数 [m^2/s]
        "degree" : 26, # 安息角 [度数]
        "kb" : 3.3e-13, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5)
        "ka" : 11624, #逆ハックの法則係数
        "a" : 0.73, #逆ハックの法則における指数：流域面積と河川長の関係を決定する 
        "xth" : 600, #チャネル形成に必要な最低流域面積 (threshold Contributory Area)[m^2] QGISだと3000ピクセルに相当
        "U" : 1.58e-11, # 隆起速度 [m/s] 0.8mm/yr = 2.54e-11, 0.5mm/yr = 1.58e-11,  0.1mm/yr = 3.1e-12
        "initial topograpy" : ["natureQGIS"], # wholearea_degree, eacharea_degree, natureQGIS
        "initial degree" : "-",  # 初期地形での勾配
        "initial divide" : "-",  # 初期地形での分水界位置を領域全体に対する割合で指定。左側境界が０で右側境界が1とする。
        "initial Ldegree" : "-", # 初期地形左側領域での勾配
        "initial Rdegree" : "-", # 初期地形右側領域での勾配
        "diff baselevel" : "-", # 初期地形でのベースレベル差
        "FileWords" : "something", # ファイル名に入れる任意のワード
        "Fpath" : r"G:\miyata\Simulation Model\result", # 結果を保存するファイルのディレクトリパス
        "z_H5file_name" : "-", # 分析対象のH5file名。allの場合はFpathに含まれるすべての
        "only_one_param" : "-",
        "xzExfile" : r"G:\miyata\Simulation Model\result\NY2.xlsx",
        "nature" : ["NY2"]
    }
    
    param_dict_3 = {
        "dx" : 10, #空間刻み幅 [m]
        "dt" : 3600*24, #1タイムステップ当たりの時間 [s]
        "nmax": 365*1000, #時間積分の繰り返し数
        "iterNum": 300, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
        "n" : 1,  # ストリームパワーモデルにおける急峻度指数 n
        "m" : 0.5, #　ストリームパワーモデルにおける流域面積指数 m
        "kd" : "-", #拡散係数 [m^2/s]
        "degree" : 26, # 安息角 [度数]
        "kb" : 3.3e-13, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5)
        "ka" : 101454, #逆ハックの法則係数
        "a" : 0.35, #逆ハックの法則における指数：流域面積と河川長の関係を決定する 
        "xth" : 600, #チャネル形成に必要な最低流域面積 (threshold Contributory Area)[m^2] QGISだと3000ピクセルに相当
        "U" : 1.58e-11, # 隆起速度 [m/s] 0.8mm/yr = 2.54e-11, 0.5mm/yr = 1.58e-11,  0.1mm/yr = 3.1e-12
        "initial topograpy" : ["natureQGIS"], # wholearea_degree, eacharea_degree, natureQGIS
        "initial degree" : "-",  # 初期地形での勾配
        "initial divide" : "-",  # 初期地形での分水界位置を領域全体に対する割合で指定。左側境界が０で右側境界が1とする。
        "initial Ldegree" : "-", # 初期地形左側領域での勾配
        "initial Rdegree" : "-", # 初期地形右側領域での勾配
        "diff baselevel" : "-", # 初期地形でのベースレベル差
        "FileWords" : "something", # ファイル名に入れる任意のワード
        "Fpath" : r"G:\miyata\Simulation Model\result", # 結果を保存するファイルのディレクトリパス
        "z_H5file_name" : "-", # 分析対象のH5file名。allの場合はFpathに含まれるすべての
        "only_one_param" : "-",
        "xzExfile" : r"G:\miyata\Simulation Model\result\NY3.xlsx",
        "nature" : ["NY3"]
    }
    
    param_dict_4 = {
        "dx" : 10, #空間刻み幅 [m]
        "dt" : 3600*24, #1タイムステップ当たりの時間 [s]
        "nmax": 365*10, #時間積分の繰り返し数
        "iterNum": 1, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
        "n" : 1.95,  # ストリームパワーモデルにおける急峻度指数 n
        "m" : 1, #　ストリームパワーモデルにおける流域面積指数 m
        "kd" : 2e-12, #拡散係数 [m^2/s]
        "degree" : 50, # 安息角 [度数]
        "kb" : 2.6e-15, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5)
        "ka" : 8700, #逆ハックの法則係数
        "a" : 0.73, #逆ハックの法則における指数：流域面積と河川長の関係を決定する 
        "th_ContA" : 300000, #チャネル形成に必要な最低流域面積 (threshold Contributory Area)[m^2] QGISだと3000ピクセルに相当
        "U" : 1.58e-11, # 隆起速度 [m/s] 0.8mm/yr = 2.54e-11, 0.5mm/yr = 1.58e-11,  0.1mm/yr = 3.1e-12
        "initial topograpy" : "diff_baselevel", # wholearea_degree, eacharea_degree
        "initial degree" : 11.8,  # 初期地形での勾配
        "initial divide" : [0.3785],  # 初期地形での分水界位置を領域全体に対する割合で指定。左側境界が０で右側境界が1とする。
        "initial Ldegree" : 15.8, # 初期地形左側領域での勾配
        "initial Rdegree" : 11.8, # 初期地形右側領域での勾配
        "diff baselevel" : [0], # 初期地形でのベースレベル差
        "FileWords" : "something", # ファイル名に入れる任意のワード
        "Fpath" : r"F:\miyata\Simulation Model\result", # 結果を保存するファイルのディレクトリパス
        "z_H5file_name" : "Elevation_0_(initial degree, initial divide) == (11.86, 0.45)", # 分析対象のH5file名。allの場合はFpathに含まれるすべての
        "only_one_param" : "-"
    }
    
    dict_list = list_of_dict(param_dict_1, param_dict_2) #, param_dict_3
    result = p.map(main_wrapper, dict_list)
    print(result)
    #main_wrapper(param_dict)
