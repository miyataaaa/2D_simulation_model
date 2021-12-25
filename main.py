#import os 

import stream_capture as sc
import file_operation as fl
import gc
from memory_profiler import profile
import numpy as np
import pandas as pd

# @profile
def main(**originalkwargs):
    
    Kw = fl.Kwargs()
    Fl = fl.File()
    iterNum = originalkwargs['iterNum']
    if 'uplift' in originalkwargs:
        uplift = originalkwargs['uplift']
    else:
        uplift = "uniform"
    
    param_name, param_set = Kw.reservation_list(**originalkwargs)
    
    for i in range(len(param_set)):
        
        
        updated_kwargs = Kw.update_kwargs(i, param_name, param_set, **originalkwargs)
        
        Fpath = updated_kwargs['Fpath']
        df_param_dict = pd.DataFrame(updated_kwargs.values(), index=updated_kwargs.keys(), columns=['value'], dtype=str)
        df_param_dict.to_hdf(Fpath + '\\param_dict.h5', key = "param_dict")

        print("\nnow" + str(param_set[i]))
        print(str(len(param_set) -i) + " more to go")

        
        Itp = sc.Initial_Topograpy(**updated_kwargs)
        Cl = sc.Calc(**updated_kwargs)
        Pds = sc.plot_divide_simulation(**updated_kwargs)

        zmax = 0
        emax = 0
        
        try:
            for k in range(iterNum):
                nameTime , nameLocation, nameErate = Fl.fname_for_plot(k, **updated_kwargs)



                if k == 0:
                    try:
                        # do_computing関数の引数が初期地形かどうかで場合分けしている
                        x, init_z = Itp.select_initial_topograpy(**updated_kwargs)
                        z, divide_id_array, z_lr, z_hl, z_rr, r_dsp_array, stream_cp, l_erosion_rate, r_erosion_rate = Cl.do_computing(x, init_z, k, uplift=uplift)
            #                     from IPython.core.debugger import Pdb; Pdb().set_trace()
                        zmax = init_z.max() + 100
                        if type(z_lr[0]) == type(0):
                            emax = r_erosion_rate[0].max()
                        elif type(z_rr[0]) == type(0):
                            emax = l_erosion_rate[0].max()
                        else:
                            lemax = l_erosion_rate[0].max() #l_erosion_rate[0].max()
                            remax = r_erosion_rate[0].max()
                            if lemax >= remax:
                                emax = lemax
                            else:
                                emax = remax
                        r_dsp = 0
                        h_dsp = 0
                        if len(r_dsp_array) != 0:
                            r_dsp = r_dsp_array[0]
                            h_dsp = r_dsp_array[-1]
                        Pds.plot_xz(x, z, stream_cp, nameTime, k, zmax)
                        Pds.plot_three_zone(x, z_lr, z_hl, z_rr, r_dsp, h_dsp, stream_cp, nameLocation, zmax)
                        Pds.plot_erosion_rate(x, l_erosion_rate, r_erosion_rate, r_dsp, h_dsp, stream_cp, nameErate, k, emax)

                    except:
                        print("\nerror happend, go to next param set")
                        break

                    finally:
                        if stream_cp != 0:
                            z = Cl.convert_List_to_Numpy(z)
                            Fl.save_to_hdf(z, i, k, **updated_kwargs)
                            break
                        else:
                            z = np.array(z)
                            Fl.save_to_hdf(z, i, k, **updated_kwargs)
                        z_next = z[-1]
                        del z, divide_id_array, z_lr, z_hl, z_rr, r_dsp_array, stream_cp, l_erosion_rate, r_erosion_rate
                        gc.collect()



                elif k != (iterNum-1):
                    try:
                        z, divide_id_array, z_lr, z_hl, z_rr, r_dsp_array, stream_cp, l_erosion_rate, r_erosion_rate = Cl.do_computing(x, z_next, k, uplift=uplift)
                        print("zlr: \n{}".format(z_lr))
                        r_dsp = 0
                        h_dsp = 0
                        if len(r_dsp_array) != 0:
                            r_dsp = r_dsp_array[0]
                            h_dsp = r_dsp_array[-1]
                        Pds.plot_xz(x, z, stream_cp, nameTime, k, zmax)
                        Pds.plot_three_zone(x, z_lr, z_hl, z_rr, r_dsp, h_dsp, stream_cp, nameLocation, zmax)
                        Pds.plot_erosion_rate(x, l_erosion_rate, r_erosion_rate, r_dsp, h_dsp, stream_cp, nameErate, k, emax)
                    except:
                        print("\nerror happend, go to next param set")
                        break

                    finally:
                        if stream_cp != 0:
                            z = Cl.convert_List_to_Numpy(z)
                            Fl.save_to_hdf(z, i, k, **updated_kwargs)
                            break
                        else:
                            z = np.array(z)
                            Fl.save_to_hdf(z, i, k, **updated_kwargs)
                        z_next = z[-1]
                        del z, divide_id_array, z_lr, z_hl, z_rr, r_dsp_array, stream_cp, l_erosion_rate, r_erosion_rate
                        gc.collect()


                else:
                    try:
                        z, divide_id_array, z_lr, z_hl, z_rr, r_dsp_array, stream_cp, l_erosion_rate, r_erosion_rate = Cl.do_computing(x, z_next, k, uplift=uplift)
                        r_dsp = 0
                        h_dsp = 0
                        if len(r_dsp_array) != 0:
                            r_dsp = r_dsp_array[0]
                            h_dsp = r_dsp_array[-1]
                        Pds.plot_xz(x, z, stream_cp, nameTime, k, zmax)
                        Pds.plot_three_zone(x, z_lr, z_hl, z_rr, r_dsp, h_dsp, stream_cp, nameLocation, zmax)
                        Pds.plot_erosion_rate(x, l_erosion_rate, r_erosion_rate, r_dsp, h_dsp, stream_cp, nameErate, k, emax)

                    except:
                        print("\nerror happend, go to next param set")
                        break

                    finally:
                        if stream_cp != 0:
                            z = Cl.convert_List_to_Numpy(z)
                            Fl.save_to_hdf(z, i, k, **updated_kwargs)
                            break
                        else:
                            z = np.array(z)
                            Fl.save_to_hdf(z, i, k, **updated_kwargs)
                        del z, divide_id_array, z_lr, z_hl, z_rr, r_dsp_array, stream_cp, l_erosion_rate, r_erosion_rate
                        gc.collect()

        except:
            
            continue

    print("\nfinish!!!!!!!!!!!!!!\n")   
    f = "finish!!" + "initial divide = " + str(originalkwargs['initial divide'])

    return f

    #     print(param_dict['Fpath'])

    #     print(param_dict)

    
#=================================================================================================================================================================================================================================================

if __name__ == "__main__":
    
    param_dict = {
    "dx" : 10, #空間刻み幅 [m]
    "dt" : 3600*24, #1タイムステップ当たりの時間 [s]
    "nmax": 365*1000, #時間積分の繰り返し数
    "iterNum": 400, #総計算時間を決める数。具体的にはnamx*iterNumだけ計算する。メモリの都合上、このように総計算時間を分割した。
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
    "initial divide" : [0.1],  # 初期地形での分水界位置を領域全体に対する割合で指定。左側境界が０で右側境界が1とする。
    "initial Ldegree" : 15.8, # 初期地形左側領域での勾配
    "initial Rdegree" : 11.8, # 初期地形右側領域での勾配
    "diff baselevel" : [100], # 初期地形でのベースレベル差
    "FileWords" : "something", # ファイル名に入れる任意のワード
    "Fpath" : r"F:\miyata\Simulation Model\result", # 結果を保存するファイルのディレクトリパス
    "z_H5file_name" : "Elevation_0_(initial degree, initial divide) == (11.86, 0.45)", # 分析対象のH5file名。allの場合はFpathに含まれるすべての
    "only_one_param" : "-"   
}
    
    main(**param_dict)
#     import stream_capture as sc
#     Itp = sc.Initial_Topograpy(**param_dict)
#     x, z = Itp.wholearea_degree()
    
#     print("x:\n".format(x))
#     print("z:\n".format(z))
