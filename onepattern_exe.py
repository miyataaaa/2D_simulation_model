import stream_capture as sc
param_dict = {
    "dx" : 10, #空間刻み幅 [m]
    "dt" : 3600*24, #1タイムステップ当たりの時間 [s]
    "nmax": 365*500, #時間積分の繰り返し数
    "n" : 2, # ストリームパワーモデルにおける急峻度指数 n
    "m" : 1, #　ストリームパワーモデルにおける流域面積指数 m
    "kd" : 2e-12, #拡散係数 [m^2/s]
    "degree" : 40, # 安息角 [度数]
    "kb" : 5.2e-15, # 侵食係数 [m^0.4/s]
    "ka" : 8700, #逆ハックの法則係数
    "a" : 0.73, #逆ハックの法則における指数：流域面積と河川長の関係を決定する 
    "th_ContA" : 300000, #チャネル形成に必要な最低流域面積 (threshold Contributory Area)[m^2] QGISだと3000ピクセルに相当
    "U" : 1.58e-11, # 隆起速度 [m/s] 0.5mm/yr
    "FileWords" : "param_modified", # ファイル名に入れる任意のワード
    "Fpath" : r"C:\Users\miyar\NUMERICAL FLUID COMPUTATION METHOD\result_img" # 結果を保存するファイルのディレクトリパス
}

iterNum = 4 #nmax * 3の計算をオブジェクトをdelしながら計算する事を意味する数字

Fl = sc.File(**param_dict)
tp = sc.Topograpy(**param_dict)
Cl = sc.Calc(**param_dict)
pds = sc.plot_divide_simulation(**param_dict)

if __name__ == "__main__":
    
    for k in range(iterNum):
        nameTime , nameLocation, nameErate = Fl.fname_form_param(k)

        if k == 0:
            x, init_z = tp.initial_topograpy_2() 
            z, divide_id_array, z_lr, z_hl, z_rr, r_dsp_array, stream_cp, l_erosion_rate, r_erosion_rate = Cl.do_computing(x, init_z)
            r_dsp = 0
            h_dsp = 0
            if len(r_dsp_array) != 0:
                r_dsp = r_dsp_array[0]
                h_dsp = r_dsp_array[-1]
            graph_time = pds.plot_xz(x, z, stream_cp, nameTime, k)
            graph_location = pds.plot_three_zone(x, z_lr, z_hl, z_rr, r_dsp, h_dsp, stream_cp, nameLocation)
            graph = pds.plot_erosion_rate(x, l_erosion_rate, r_erosion_rate, r_dsp, h_dsp, stream_cp, nameErate, k)
            z_next = z[-1]
            if stream_cp != 0:
                break
                
            del z, z_lr, z_hl, z_rr, l_erosion_rate, r_erosion_rate

        elif k != (iterNum-1):
            z, divide_id_array, z_lr, z_hl, z_rr, r_dsp_array, stream_cp, l_erosion_rate, r_erosion_rate = Cl.do_computing(x, z_next)
            r_dsp = 0
            h_dsp = 0
            if len(r_dsp_array) != 0:
                r_dsp = r_dsp_array[0]
                h_dsp = r_dsp_array[-1]
            graph_time = pds.plot_xz(x, z, stream_cp, nameTime, k)
            graph_location = pds.plot_three_zone(x, z_lr, z_hl, z_rr, r_dsp, h_dsp, stream_cp, nameLocation)
            graph = pds.plot_erosion_rate(x, l_erosion_rate, r_erosion_rate, r_dsp, h_dsp, stream_cp, nameErate, k)

            z_next = z[-1]
            if stream_cp != 0:
                break

            del z, z_lr, z_hl, z_rr, l_erosion_rate, r_erosion_rate

        else:
            z, divide_id_array, z_lr, z_hl, z_rr, r_dsp_array, stream_cp, l_erosion_rate, r_erosion_rate = Cl.do_computing(x, z_next)
            r_dsp = 0
            h_dsp = 0
            if len(r_dsp_array) != 0:
                r_dsp = r_dsp_array[0]
                h_dsp = r_dsp_array[-1]
            graph_time = pds.plot_xz(x, z, stream_cp, nameTime, k)
            graph_location = pds.plot_three_zone(x, z_lr, z_hl, z_rr, r_dsp, h_dsp, stream_cp, nameLocation)
            graph = pds.plot_erosion_rate(x, l_erosion_rate, r_erosion_rate, r_dsp, h_dsp, stream_cp, nameErate, k)
