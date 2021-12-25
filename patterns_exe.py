import os 
import stream_capture as sc
import gc

param_dict = {
    "dx" : 10, #空間刻み幅 [m]
    "dt" : 3600*24, #1タイムステップ当たりの時間 [s]
    "nmax": 365*8000, #時間積分の繰り返し数
    "n" : 1.95,  # ストリームパワーモデルにおける急峻度指数 n
    "m" : 1, #　ストリームパワーモデルにおける流域面積指数 m
    "kd" : 2e-12, #拡散係数 [m^2/s]
    "degree" : 50, # 安息角 [度数]
    "kb" : 2.6e-15, # 侵食係数 [m^0.4/s] 2.6e-15(n2, m1, U0.5), 2.4e-12(n0.8, m0.3, U0.5), 2.6e-15(n1.95, m1, U0.5)
    "ka" : 8700, #逆ハックの法則係数
    "a" : 0.73, #逆ハックの法則における指数：流域面積と河川長の関係を決定する 
    "th_ContA" : 300000, #チャネル形成に必要な最低流域面積 (threshold Contributory Area)[m^2] QGISだと3000ピクセルに相当
    "U" : 1.58e-11, # 隆起速度 [m/s] 0.8mm/yr = 2.54e-11, 0.5mm/yr = 1.58e-11,  0.1mm/yr = 3.1e-12
    "initial degree" : 30,  # 初期地形での勾配
    "initial divide" : 0.25,  # 初期地形での分水界位置を領域全体に対する割合で指定。左側境界が０で右側境界が1とする。
    "initial Ldegree" : 50, # 初期地形左側領域での勾配
    "initial Rdegree" : 15, # 初期地形右側領域での勾配
    "FileWords" : "something", # ファイル名に入れる任意のワード
    "Fpath" : r"C:\Users\students\NUMERICAL_FLUID_COMPUTATION_METHOD\result_img" # 結果を保存するファイルのディレクトリパス
}

# 変化させるパラメータの予約リスト
reservation_list = {"initial degree": [30], 
                    "initial divide": [0.125, 0.25, 0.3, 0.5]}

# 予約リストから変数名と値を別々に取り出して格納するためのリスト
param_name = []
param_value = []
for key, param in reservation_list.items():
    param_name.append(key)
    param_value.append(param)

# パラメータ組合わせをタプル化したものを要素とするリスト。なので、パラメータ組み合わせ数の数だけ要素を持つ
param_set = []
for i in param_value[0]:
    for j in param_value[1]:
        param_set.append((i, j))
        
# param_dict[param_name[0]] = param_set[0][0]
# print(param_dict)

Fpath = param_dict['Fpath']


for i in range(len(param_set)):
    param_dict[param_name[0]] = param_set[i][0]
    param_dict[param_name[1]] = param_set[i][1]
    param_dict['FileWords'] = "(" + param_name[0] + ", " + param_name[1] + ")" + " == " + str(param_set[i])
    new_parent_path = Fpath + "\\" + param_name[0] + "-" + param_name[1] + "_patterns" + "\\" + str(param_set[i][0]) + "_" + str(param_set[i][1])
    os.makedirs(new_parent_path, exist_ok=True)
    param_dict['Fpath'] = new_parent_path
    
    print("\nnow" + str(param_set[i]))
    print(str(len(param_set) -i) + " more to go")

    Fl = sc.File(**param_dict)
    Itp = sc.Initial_Topograpy(**param_dict)
    Cl = sc.Calc(**param_dict)
    Pds = sc.plot_divide_simulation(**param_dict)

    if __name__ == "__main__":
        
        iterNum = 35 #nmax * 3の計算をオブジェクトをdelしながら計算する事を意味する数字
        zmax = 0
        for k in range(iterNum):
            nameTime , nameLocation, nameErate = Fl.fname_form_param(k)
            

            if k == 0:
                x, init_z = Itp.wholearea_degree()
                z, divide_id_array, z_lr, z_hl, z_rr, r_dsp_array, stream_cp, l_erosion_rate, r_erosion_rate = Cl.do_computing(x, init_z, k)
                zmax = init_z.max() + 100
                r_dsp = 0
                h_dsp = 0
                if len(r_dsp_array) != 0:
                    r_dsp = r_dsp_array[0]
                    h_dsp = r_dsp_array[-1]
                graph_time = Pds.plot_xz(x, z, stream_cp, nameTime, k, zmax)
                graph_location = Pds.plot_three_zone(x, z_lr, z_hl, z_rr, r_dsp, h_dsp, stream_cp, nameLocation, zmax)
                graph_erosion = Pds.plot_erosion_rate(x, l_erosion_rate, r_erosion_rate, r_dsp, h_dsp, stream_cp, nameErate, k)
                z_next = z[-1]
                if stream_cp != 0:
                    break

                del z, z_lr, z_hl, z_rr, l_erosion_rate, r_erosion_rate
                gc.collect()

            elif k != (iterNum-1):
                z, divide_id_array, z_lr, z_hl, z_rr, r_dsp_array, stream_cp, l_erosion_rate, r_erosion_rate = Cl.do_computing(x, z_next, k)
                r_dsp = 0
                h_dsp = 0
                if len(r_dsp_array) != 0:
                    r_dsp = r_dsp_array[0]
                    h_dsp = r_dsp_array[-1]
                graph_time = Pds.plot_xz(x, z, stream_cp, nameTime, k, zmax)
                graph_location = Pds.plot_three_zone(x, z_lr, z_hl, z_rr, r_dsp, h_dsp, stream_cp, nameLocation, zmax)
                graph_erosion = Pds.plot_erosion_rate(x, l_erosion_rate, r_erosion_rate, r_dsp, h_dsp, stream_cp, nameErate, k)

                z_next = z[-1]
                if stream_cp != 0:
                    break

                del z, z_lr, z_hl, z_rr, l_erosion_rate, r_erosion_rate
                gc.collect()


            else:
                z, divide_id_array, z_lr, z_hl, z_rr, r_dsp_array, stream_cp, l_erosion_rate, r_erosion_rate = Cl.do_computing(x, z_next, k)
                r_dsp = 0
                h_dsp = 0
                if len(r_dsp_array) != 0:
                    r_dsp = r_dsp_array[0]
                    h_dsp = r_dsp_array[-1]
                graph_time = Pds.plot_xz(x, z, stream_cp, nameTime, k, zmax)
                graph_location = Pds.plot_three_zone(x, z_lr, z_hl, z_rr, r_dsp, h_dsp, stream_cp, nameLocation, zmax)
                graph_erosion = Pds.plot_erosion_rate(x, l_erosion_rate, r_erosion_rate, r_dsp, h_dsp, stream_cp, nameErate, k)
                
                del z, z_lr, z_hl, z_rr, l_erosion_rate, r_erosion_rate
                gc.collect()

print("\nfinish!!!!!!!!!!!!!!\n")   
    
#     print(param_dict['Fpath'])

#     print(param_dict)
