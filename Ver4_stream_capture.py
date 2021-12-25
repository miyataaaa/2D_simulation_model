#!/usr/bin/env python
# coding: utf-8

# In[ ]:
"""Ver3では境界条件に本流のシミュレーション結果を使用できる使用に変更。Ver2_mulitprocessing.pyと、Ver3_main_not_exhandling.pyと組み合わせて使用する。
Ver4では隆起速度をCalcクラスのメソッドではなく、新たに定義したUpliftMakerクラスから与える仕様に変更。
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from memory_profiler import profile
import file_operation as fl
Fl = fl.File()
#import joblib
#get_ipython().run_line_magic('pdb', 'on')

# %pdb


class Initial_Topograpy:
    
    """様々な初期地形を形成するための関数をまとめたクラス。"""
    
    def __init__(self, **kwargs):
        self.dx = kwargs['dx']
        self.dt = kwargs['dt']
        self.nmax = kwargs['nmax']
        self.degree = kwargs['degree']
        self.xth = kwargs['xth']
        self.initTp = kwargs['initial topograpy']
        self.Fpath = kwargs['Fpath']
        self.xzExfile = self._get_xzExfile(**kwargs)
        self.df = self._get_df()
        
    def _get_xzExfile(self, **kwargs):
        
        """xzExfileには、x, z, (u *なくても良い)が格納されたExcelファイルの絶対パスを指定"""
        
        if 'xzExfile' in kwargs:
            if type(kwargs['xzExfile']) == list:
                xzExfile = kwargs['xzExfile'][0]
            else:
                xzExfile = kwargs['xzExfile']
            
            if xzExfile == "-":
                # 辞書内にxzExfileは存在するが使っていない場合。
                self.xzExfile = None
            else:
                self.xzExfile = xzExfile
        else:
            # そもそも辞書内にxzExfileが存在しない場合
            self.xzExfile = None
            
    def _get_df(self):
        if self.xzExfile != None:
            self.df = pd.read_excel(self.xzExfile, 'for simulation', index_col=None)
        else:
            self.df = None
                  
    def natureQGIS_xzu(self, **kwargs):
        
        """
        QGISから抽出した標高値を初期地形とする関数。
        抽出した標高は、dx幅で補間されたデータ状態でExcelファイルに格納されており、
        sheet名がfor simulationである前提。
        kwargs['xzExfile']に絶対パスを指定。
        """
        assert self.df != None, 'there is No DataFrame'
        x = self.df.iloc[:, 0].to_numpy()
        z = self.df.iloc[:, 1].to_numpy()
        
        return x, z
        
    def select_initial_topograpy(self, **kwargs):
        
        if self.initTp == "natureQGIS_xzu":
            return self.natureQGIS_xzu(**kwargs)
        
# =============================================================================================================================
class Topograpy:
    
    """計算途中で形成された地形から分水界の位置、河川・斜面境界を計算する関数をまとめたクラス。侵食や隆起を計算するためのクラスではなく、計算に必要な地形パラメータをそろえるクラス。初期地形形成関数は別クラスとして分離した。
    ハックの法則の距離の取り方を流路長から水平方向のみに変更。分水界の左右の流域面積、勾配を求める関数もある"""
    
    def __init__(self, **kwargs):
        self.dx = kwargs['dx']
        self.ka = kwargs['ka']
        self.a = kwargs['a']
        self.xth = kwargs['xth']    

    def ContA_left(self, x_left):
        """
        2021/10/22変更
        
        与えられた河川領域の流域面積のみを計算する方法に変更。関数内で、別のx軸座標の取り方を規定して流域面積を逆ハックの法則で計算。
        谷頭が1mからスタートするかつ、与えられたxのndarrayと同じ要素数になるように新たなx座標を定義。
        
    　　左側河川は、流域面積を計算する時のx座標の取り方と、おおもとのx座標の正の向きが反転する事に注意。
        """
        
        dx = self.dx
        ka = self.ka
        a = self.a
        river_length = np.flipud(np.arange(1, (len(x_left)*dx)+1, dx))
        ContA_left = ka * (river_length ** a) 
        ContA_left = np.round(ContA_left, 4)
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return ContA_left 

    def ContA_right(self, x_right):
        
        """
        2021/10/22変更
        
        与えられた河川領域の流域面積のみを計算する方法に変更。関数内で、別のx軸座標の取り方を規定して流域面積を逆ハックの法則で計算。
        谷頭が1mからスタートするかつ、与えられたxのndarrayと同じ要素数になるように新たなx座標を定義。
        """
        dx = self.dx
        ka = self.ka
        a = self.a
        river_length = np.arange(1, (len(x_right)*dx)+1, dx)
        ContA_right = ka * (river_length ** a) 
        ContA_right = np.round(ContA_right, 4)
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return ContA_right

    def _river_hillslope_boundary(self, x, z):
        
        """
        hillslopeの最初と最後のインデックスを返す関数。
        
        Ver1.
        勾配と流域面積の関係で位置が変化する仕様ではなく、チャネル形成に必要な
        最低流域面積を決めておく仕様に変更した。
        
        2021/10/22変更
        Ver2.
        ヒルスロープを固定長(xth)にして、3つの領域に分ける仕様に変更
        """
        xth = self.xth
        dx = self.dx
        divide_id = z.argmax()
        half_of_Hill = int(xth/dx)
        
        boundary_left_id = divide_id - half_of_Hill
        boundary_right_id = divide_id + half_of_Hill    
#         print(boundary_left_id, boundary_right_id)
        
        if boundary_left_id < 0:
            boundary_left_id = 0
        if boundary_right_id < 0:
            boundary_right_id = len(x)-1
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return boundary_left_id, boundary_right_id

#===========================================================================================================================================================================================================================


class Calc(Topograpy):
    
    """分水界より左側の河川、斜面領域、右側の河川での標高変化を計算するクラス。"""
    
    def __init__(self, initTP, **kwargs):
        super().__init__(**kwargs)
        self.initTP = initTP
        self.dt = kwargs['dt']
        self.nmax = kwargs['nmax']
        self.n = kwargs['n']
        self.m = kwargs['m']
        self.kd = kwargs['kd']
        self.degree = kwargs['degree']
        self.kb = kwargs['kb']
        self.U = kwargs['U']
        
    def split_to_three_zone(self, x, z):
        
        """対象領域を分水界左右の河川、斜面領域に分けてそれぞれのｚ（標高）のndarrayと、
        左右の河川の河川部分だけの流域面積(ContA)のndarrayを返す関数。分水界の左側全体の流域面積を
        求めたいときはこのクラスの基底クラスのContA関数を使うこと。右側も同様。threshold_hillslope関数のために
        hillslope部分のxの値も同時に抽出して返す。
        
        2021/10/22変更
        _river_hillslope_boundary関数の変更に伴い、内部での河川丘陵地境界の使い方を変更。
        具体的に、1.流域面積は河川領域のみで計算。2.modified_r_boundaryとr_boundaryの統一。
        """
        
        l_boundary, r_boundary = self._river_hillslope_boundary(x, z)
        
        # 3つの領域が存在するかしないかで場合分け。関数の返り値の数は3つの領域が存在する場合に合わせるので空の配列を返す処理を使っている。
        
        if l_boundary == 0:
            z_left_river = np.array([]) # 存在しないので空のndarrayを返す。
            z_hillslope = z[0 : r_boundary+2] # 境界値に川の領域の最後の要素を使うので、左側河川の最後の要素＋hillslope+右側河川の最後の要素を持つ配列
            z_right_river = z[r_boundary+1:] #右側の河川の領域だけ
            x_hillslope = x[0 : r_boundary+2]
            ContA_left = np.array([]) # 存在しないので空のndarrayを返す
            ContA_right = self.ContA_right(x[r_boundary+1:])
#             from IPython.core.debugger import Pdb; Pdb().set_trace()            
        elif r_boundary == (len(z)-1):
            z_left_river = z[:l_boundary] # zはndarrayで要素はスカラーなのでコピーによって生成されるのはビューではなく独立したものになる。これは純粋に川の領域だけ
            z_hillslope = z[l_boundary-1 : ] # 境界値に川の領域の最後の要素を使うので、左側河川の最後の要素＋hillslope+右側河川の最後の要素を持つ配列
            z_right_river = np.array([]) #右側の河川の領域だけ。右側の河川はないのでからのndarrayを返す。
            x_hillslope = x[l_boundary-1 : ]
            ContA_left = self.ContA_left(x[:l_boundary])
    #         from IPython.core.debugger import Pdb; Pdb().set_trace()
            ContA_right = np.array([]) # 存在しないので空のndarrayを返す
        else:
            z_left_river = z[:l_boundary] # zはndarrayで要素はスカラーなのでコピーによって生成されるのはビューではなく独立したものになる。これは純粋に川の領域だけ
            z_hillslope = z[l_boundary-1 : r_boundary+2] # 境界値に川の領域の最後の要素を使うので、左側河川の最後の要素＋hillslope+右側河川の最後の要素を持つ配列
            z_right_river = z[r_boundary+1:] #右側の河川の領域だけ
            x_hillslope = x[l_boundary-1 : r_boundary+2]
            ContA_left = self.ContA_left(x[:l_boundary])
    #         from IPython.core.debugger import Pdb; Pdb().set_trace()
            ContA_right = self.ContA_right(x[r_boundary+1:])
        
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return z_left_river, z_hillslope, z_right_river, ContA_left, ContA_right, x_hillslope
    
    def river_left(self, z, ContA):
        
        """分水界の左側の河川の標高変化を計算する関数。とりあえず、隆起速度は０。
        引数のzとContAは、do_computing関数の中で定義されるzold = z.copy()を3領域に分割した物の左側河川の値。最後に３つ計算したら
        z = np.concatenate((zlr, zh, zrr))にすることで結合して更新していく。河口側境界の標高は様々な条件に合わせて可変にする。
        さらに、侵食速度のみを可視化するために侵食速度値も返り値として計算する"""
        
        dx = self.dx
        dt = self.dt
        kb = self.kb
        m = self.m
        n = self.n
        
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
#         zlr, _, _, ContA_left, _, _ = self.split_to_three_zone(x, z)
        zlr = z
        ContA_left = ContA
#         zlr_new = zlr[:] #　返り値用の配列。境界条件が０で固定ではなく、可変にするためにzlrのコピーを代入。インデックス参照の形でコピーを作る。

#         for j in range(1, len(zlr)):
#             zlr_new[j] =  zlr[j] - (self.dt * self.kb * (ContA_left[j]**self.m) * (((zlr[j]-zlr[j-1])/self.dx)**self.n))
        
        riverse_zlr = np.flipud(zlr)
        diff_zlr = np.abs(np.flipud(np.hstack((np.diff(riverse_zlr), np.zeros(1)))))
        erosion_rate = kb * (ContA_left ** m) * (((diff_zlr) / dx) ** n) #侵食速度
        zlr_new = zlr - (dt * erosion_rate)
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return zlr_new, erosion_rate
    
    def river_right(self, z, ContA):
        
        """分水界の右側の河川の標高変化を計算する関数。とりあえず、隆起速度は０。
        引数のzとContAは、do_computing関数の中で定義されるzold = z.copy()を3領域に分割した物の右側河川での値。最後に３つ計算したら
        z = np.concatenate((zlr, zh, zrr))にすることで結合して更新していく。後退差分で差分をとる＋インデックスの
        最大値が河口側境界になるので逆順ループにする。河口側境界の標高は様々な条件に合わせて可変にする。"""
        
        dx = self.dx
        dt = self.dt
        kb = self.kb
        m = self.m
        n = self.n
        
        zrr = z
        ContA_right = ContA
#         zrr_new = zrr[:]
#         for j in range(len(zrr)-2, -1, -1):
#             # インデックス大きいほうが河口側境界なので逆順ループ。これにより、後退差分が以下の形になる。
#             zrr_new[j] =  zrr[j] - (self.dt * self.kb * (ContA_right[j]**self.m) * (((zrr[j]-zrr[j+1])/self.dx)**self.n)) #* (zold[j]-zold[j-1]) 
        
        diff_zrr = np.abs(np.hstack((np.diff(zrr), np.zeros(1)))) # 河口側境界の勾配は０に設定するので境界は侵食されない。
        erosion_rate = kb * (ContA_right ** m) * (((diff_zrr) / dx) ** n)
        zrr_new = zrr - (dt * erosion_rate)
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return zrr_new, erosion_rate
    
    def diffusion_hillslope(self, z): # numpyベクトル計算に書き換えていないので注意！！！！！！！！！！！！！！（更新が止まってる）
    
        """hillslopeによる作用での標高変化を各タイムステップで計算するための関数。
        zhはインデックスがhillslope±１での値をもつ標高値。なのでfor文はzh[1:-1]で繰り返す"""
        
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        zh = z
        
        zh_new = np.zeros(len(zh)-2) # zhの最初と最後は河川部分なのでzh-2の要素数を持つ新たなndarrayを生成する
        
        for j in range(1, len(zh)-1):
            
            # 計算ではzhの最初と最後の値を境界条件として使いたい。繰り返しのjはzhでのhillslopeの部分が基準
            # rangeは終端を含まないので最後の要素(インデックス:len(zh)-1)の１つ前まで計算する
            dz = self.kd * self.dt * (zh[j+1] -(2.0 * zh[j]) + zh[j-1]) / (self.dx ** 2)
            # あらかじめ用意しておいたzh_newはzhより要素が２個少ない。zh_newの最初の要素がhillslopeの最初になるようにしている。
            zh_new[j-1] = zh[j] + dz

        return zh_new

    def threshold_hillslope(self, z, x, len_zlr, len_zrr):
        
        """安息角以上になると地滑りが発生する事を想定したhillslope部分の変化を記述する関数。関数の構成は左右の河川末端ノードからthresholdでの
        標高値を計算する→thresholdを超えるならthreshold標高値で置換。超えないならそのままの標高値。もし、どちらかの河川が消失した状況なら、引数ｚの末端ノード
        はhillslopeの値になるので操作４で場合分けする。場合分けのために左右の河川の要素数を引数に追加した"""
        
        len_zlr = len_zlr
        len_zrr = len_zrr
        
        #----------1.hillslope areaを抽出して、分水界を境に左右に分割する------------------------------------------------------------------
        zh = z # ここで受け取るhiilslopeは両端の末端ノードが河川の末端ノードでデータはndarray（左右どちらかの河川領域が消失している場合、末端ノードのどちらかはhillsope領域
        xh = x
        xh_nrm = np.subtract(xh, xh.min()) # nrm = normalization 
#        
        #----------2.分水界より左側で、安息角での標高値を計算-----------------------------------------------------------------------------------------------
        degree = self.degree
        tan = np.tan(np.radians(degree))
        zl_threshold = np.add(zh[0], np.multiply(xh_nrm, tan)) 
        
        #----------3.分水界より右側で、安息角での標高値を計算-----------------------------------------------------------------------------------------------
        riverse_xh = np.flipud(np.abs(np.subtract(xh, x.max()))) # 右側の河口側境界を原点にするための変換。3つの領域全体でのxの右端を0にするためにxの最大値で引く＆配列の左右反転
        riverse_xh_nrm = np.subtract(riverse_xh, riverse_xh.min()) # hillslope領域の右端を0にするための操作（反転してるので計算上では左端）
        zr_threshold = np.add(zh[-1], np.flipud(np.multiply(riverse_xh_nrm, tan)))
        
        #----------4.標高値を更新-----------------------------------------------------------------------------------------------
        # 更新前の標高値が閾値を超えてしまっているなら、閾値標高で置き換えて、そうでないならそのままの値にする操作
        zh_new = np.where(zl_threshold < zh, zl_threshold, zh) # まずは分水界の左から更新
        if len_zlr == 0:
            zh_new = np.where(zr_threshold < zh_new, zr_threshold, zh_new)[:-1] # 右端の河川領域を削った配列を返す。
        elif len_zrr == 0:
            zh_new = np.where(zr_threshold < zh_new, zr_threshold, zh_new)[1:] # 左端の河川領域を削った配列を返す。
        else:
            zh_new = np.where(zr_threshold < zh_new, zr_threshold, zh_new)[1:-1] # 両端の河川領域を削った配列を返す。
        
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return zh_new
    
     
    #@profile    
    def do_computing(self, x, init_z, iterNum, z_LeftBoundary, z_RightBoundary, uplift="FromDEM"):
        
        """計算を実行する関数。引数のinit_zは初期地形での標高ｚ(型はndarray)。
        この関数の返り値は各タイムステップの標高のndarrayを要素とする大きさnmaxで各要素が大きさjmaxの計算結果ｚ
        z_LeftBoundary, z_RightBoundaryは境界条件に使用する本流での標高値。
        uplift引数でどの隆起速度クラスをインスタンス化するか指定。
        """
        
        nmax = self.nmax
        T = (nmax/365) * iterNum  # 経過時間の表示を調整するための変数。累積時間を保持する事が目的。もっと簡略化？変形するなら (nmax*iterNum + i)/365
        z_array = [0]*nmax
        z_array[0] = init_z
        
        z_lriver_array = [0]*nmax
        z_hillslope_array = [0]*nmax
        z_rriver_array = [0]*nmax       
        
        l_erosion_rate = [0]*nmax
        r_erosion_rate = [0]*nmax
        
        r_dsp_array = []  #左側の河川が消失したタイムステップインデックスを格納するlist。
        stream_cp = 0 #stram captureが起きたタイムステップインデックスを格納する変数。

        # 条件に合わせて隆起速度クラスをインスタンス化
        if uplift == "FromDEM":
            U_maker = Uplift_FromDEM(self.initTP, self.dt)
        elif uplift == "uniform":
            U_maker = UniUplift(self.U, self.dt)
        else:
            # とりあえず一様隆起
            U_maker = UniUplift(self.U, self.dt)
    
        for i in range(1, nmax):
#             print(i)
            # タイムステップ1からで、zoldは1つ前のタイムステップの標高値なので3つの領域に分けるときはi-1
            zold = copy.deepcopy(z_array[i-1])
            z_top = zold.max()
            z_lboundary = zold[0]
            z_rboundary = zold[-1]
            
            if z_lboundary >= z_top:
                # 計算対象領域の左端が最も標高が高い　＝＝　左側の河川が消失し、左側の本流を途中で河川争奪した状況
                z_rriver_array[i-1] = zold
                print("\nDid stream capture occur? left river is victim\n")
                print("The time step when stream capture occur is i = {}\n".format(i - 1))
                stream_cp = i-1
                break
                
            elif z_rboundary >= z_top:
                # 計算対象領域の右端が最も標高が高い　＝＝　右側の河川が消失し、右側の本流を途中で河川争奪した状況
                z_lriver_array[i-1] = zold
                print("\nDid stream capture occur? right river is victim\n")
                print("The time step when stream capture occur is i = {}\n".format(i - 1))
                stream_cp = i-1
                break
            else:
                zlr, zh, zrr, ContA_left, ContA_right, xh = self.split_to_three_zone(x, zold)
                len_zlr = len(zlr)
                len_zrr = len(zrr)
    #             from IPython.core.debugger import Pdb; Pdb().set_trace()
                if len(zlr) == 0:
                    # 左側の河川が消失した場合
                
                    # 可視化の為の操作
                    z_hillslope_array[i-1] = zh[:-1]
                    z_rriver_array[i-1] = zrr
                    r_dsp_array.append(i-1)
                    
                    # 右側河川での侵食速度を計算
                    zrr_new, r_erosion_rate[i-1],  = self.river_right(zrr, ContA_right)

                    # 侵食量と隆起量を計算                 
                    z_array[i] = U_maker.Add_U(np.concatenate((self.threshold_hillslope(zh, xh, len_zlr, len_zrr), zrr_new)))
                    if i % (nmax / 10) == 0:
                        print("Curently time step is t={}yr".format(T + i/365))

    #                 from IPython.core.debugger import Pdb; Pdb().set_trace()
                elif len(zrr) == 0:
                    # 右側の河川が消失した場合
            
                    # 可視化の為の操作
                    z_hillslope_array[i-1] = zh[1:]
                    z_lriver_array[i-1] = zlr
                    r_dsp_array.append(i-1)
                    
                    # 左側河川での侵食を計算
                    zlr_new, l_erosion_rate[i-1] = self.river_left(zlr, ContA_left)

                    # 侵食量と隆起量を計算
                    z_array[i] = U_maker.Add_U(np.concatenate((zlr_new, self.threshold_hillslope(zh, xh, len_zlr, len_zrr))))
                    
                    if i % (nmax / 10) == 0:
                        print("Curently time step is t={}yr".format(T + i/365))

    #                 from IPython.core.debugger import Pdb; Pdb().set_trace()
                else:
                    # 3つの領域すべてが存在する場合
                
                    # 可視化の時に３つの領域で色分けするために計算後のz_arrayを３分割。split_to_hillslopeは、hillslopeの最初と最後の値がfluvial領域の値なのでスライスしたものを格納
                    z_lriver_array[i-1] = zlr
                    z_hillslope_array[i-1] = zh[1:-1]
                    z_rriver_array[i-1] = zrr
                    
                    # 左側河川での侵食速度を計算
                    zlr_new, l_erosion_rate[i-1] = self.river_left(zlr, ContA_left)
                    
                    # 右側河川での侵食速度を計算
                    zrr_new, r_erosion_rate[i-1] = self.river_right(zrr, ContA_right)

                    # 侵食量と隆起量を計算
                    z_array[i] = U_maker.Add_U(np.concatenate((zlr_new, self.threshold_hillslope(zh, xh, len_zlr, len_zrr), zrr_new)))   
                    # 最終タイムステップの時

                    if i % (nmax / 10) == 0:
                        print("Curently time step is t={}yr".format(T + i/365))
                        

        del z_only_erosion, zold
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        # 返り値はlistだが、各要素はndarray
        return z_array, z_lriver_array, z_hillslope_array, z_rriver_array, r_dsp_array, stream_cp, l_erosion_rate, r_erosion_rate


    def convert_List_to_Numpy(self, z):
        
        """計算された標高値のndarrayを要素に持つリストをndarrayに変換する関数。河川争奪が起きたときに、リスト内に、標高値ndarrayと０が
        混在していたので、0を排除して河川争奪が起きるまでの標高値ndarrayだけを要素に持つndarrayをhdfファイルに保存するために作成した関数"""
        
        new_zarray_list = [z[i] for i in range(len(z)) if type(z[i]) == type(z[0])] 
        converted_z = np.array(new_zarray_list)
        
        return converted_z
    
    #def chi(self, x, z)

class Uplift_FromDEM:
    
    def __init__(self, initTP, dt):
        self.uDEM_flag = uDEM_flag
        self.u = initTP.df.iloc[:, 3].to_numpy()
        self.dt = dt
        
    def Add_U(self, z, z_LeftBoundary, z_RightBoundary):
        u = self.u
        dt = self.dt
        return np.hstack((z_LeftBoundary, z[1:-1]+u*dt, z_RightBoundary))
        
class UniUplift:
    
    def __init__(self, u, dt):
        self.dt = dt
        self.u = u
        
    def Add_U(self, z, z_LeftBoundary, z_RightBoundary):
        u = self.u
        dt = self.dt
        return np.hstack((z_LeftBoundary, z[1:-1]+u*dt, z_RightBoundary))
            
#============================================================================================================================================================================================================================
class plot_divide_simulation(Calc):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    #@profile      
    def plot_xz(self, x, z, stream_cp, fname, iterNum, zmax):
        
        """引数のkは、全体で計算＄描画プロセスを何回繰り返すかを示すもの。計算＆描画プロセスを1回繰り返すとグラフに表示される時間が0からになってしまうので
        そこに連続性を持たせたくて、計算＆描画プロセス数を使って表示する時間を表示する。"""
        
        nmax = self.nmax
        #zmax = z[0].max() + 50 # グラフのｚ軸サイズの上限値
        #print("plot_xz zmax: {}".format(zmax))
        T = (nmax/365) * iterNum  # グラフ間で時間の表示を調整するための変数。累積時間を保持する事が目的。もっと簡略化？変形するなら (nmax*iterNum + i)/365
#         fig = plt.figure(figsize = (10, 5), dpi = 100)
        fig = plt.figure(figsize = (45, 10), dpi = 100)
        plt.rcParams["font.size"] = 22
        fig.patch.set_facecolor('white') 
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        
        if stream_cp == 0:
            for i in range(0, len(z), int(nmax/20)):
                plt.plot(x, z[i], color = cm.jet(i/nmax), label='t={:.3f} yr'.format(T + i/365)) #marker='o', ms=3, 
            plt.plot(x, z[-1], color = 'darkred', label='t={:.3f} yr'.format((nmax/365) * (iterNum+1)))
        else:
            for i in range(0, len(z[:stream_cp+1]), int(stream_cp/20)):
                plt.plot(x, z[i], color = cm.jet(i/(stream_cp+1)), label='t={:.3f} yr'.format(T + i/365))
            plt.plot(x, z[stream_cp], color = 'darkred', label='stream capture: t={:.3f} yr'.format((stream_cp/365) * (iterNum+1)))
          
        plt.grid(color="black", linestyle="dashed", linewidth=0.5)
        #plt.axis([600, 1300, 790, 1010])
#         plt.axis([970, 1030, 970, 1030])
#         plt.axis([600, 1400, 0, 1300])
#         plt.axis([600, 1400, 800, 1300])
#         plt.axis([600, 1400, 910, 1020])
#         plt.axis([600, 1400, 0, 70])
#         plt.axis([2400, 3000, 350, 380])
#         plt.axis([2200, 3200, 350, 400])
        plt.xlim()
        plt.ylim(0, zmax)
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
#         plt.show()
#         fig.savefig("img_i2_3000yr_kb2.8e-12_time.png")
        fig.savefig(fname + ".png", bbox_inches='tight')
        fig.clear()
        plt.close(fig)
    #@profile 
    def plot_three_zone(self, x, zlr, zhl, zrr, r_dsp, h_dsp, stream_cp, fname, zmax):
        
        nmax = self.nmax
        #zmax = zhl[0].max() + 50 # グラフのｚ軸サイズの上限値
        #print("plot_three_zone: {}".format(zmax))
#         fig = plt.figure(figsize = (10, 5), dpi = 100)
        fig = plt.figure(figsize = (45, 10), dpi = 100)
        plt.rcParams["font.size"] = 22
        fig.patch.set_facecolor('white') 
        
        #最初から左側の河川がない場合
        if type(zlr[0]) == type(0):
            # hillslopeの最初のインデックス
            boundary_l = 0
            # right_riverの最初のインデックス
            boundary_r = len(zhl[0])
            # 初期地形（ラベルを領域で分けたかったのでこうした。時系列での繰り返しをなくしたかった。）
            plt.plot(x[boundary_l:boundary_r], zhl[0], color = 'olive', label="hillslope")
            plt.plot(x[boundary_r:], zrr[0], color = 'mediumturquoise', label="river")
            plt.plot(x[boundary_r], zrr[0][0], marker = 'o', ms = 7, color = 'coral')
            
            zhl = self.convert_List_to_Numpy(zhl)
            len_zhl = zhl.shape[0]
            for i in range(0, len_zhl+1, int(nmax/20)):
                # hillslopeの左端のインデックス*領域の左端
                    boundary_l = 0
                    # right_riverの最初のインデックス
                    boundary_r = len(zhl[i])
                    plt.plot(x[boundary_l:boundary_r], zhl[i], color = 'darkkhaki')
                    plt.plot(x[boundary_r:], zrr[i], color = 'mediumturquoise')
                    plt.plot(x[boundary_r], zrr[i][0], marker = 'o', ms = 7, color = 'coral')
                    
            if stream_cp != 0:
                # 河川争奪して計算領域が1つの河川になった時のプロット
                plt.plot(x, zrr[stream_cp], color = 'dodgerblue', label = "stream capture occured")
            
        else:
            # hillslopeの最初のインデックス
            boundary_l = len(zlr[0])
            # right_riverの最初のインデックス
            boundary_r = len(zlr[0]) + len(zhl[0])
            # 初期地形（ラベルを領域で分けたかったのでこうした。時系列での繰り返しをなくしたかった。）
            plt.plot(x[:boundary_l], zlr[0], color = 'mediumturquoise', label="river") #marker='o', ms=3, label='t={} yr'.format(i/365)
            plt.plot(x[boundary_l:boundary_r], zhl[0], color = 'olive', label="hillslope")
            plt.plot(x[boundary_r:], zrr[0], color = 'mediumturquoise')
            plt.plot(x[boundary_l - 1], zlr[0][-1], marker = 'o', ms = 7, color = 'coral', label="hillslope_boundary")
            plt.plot(x[boundary_r], zrr[0][0], marker = 'o', ms = 7, color = 'coral')
      
            # プロットする領域ごとにplt.plotしているので、計算領域に3領域が存在する場合とそうでない場合にわけて条件分岐
            if r_dsp == 0 and h_dsp == 0:
                # r_dsp ~~ river dissapear
                for i in range(1, len(zlr), int(nmax/20)):
                    # hillslopeの最初のインデックス
                    boundary_l = len(zlr[i])
                    # right_riverの最初のインデックス
                    boundary_r = len(zlr[i]) + len(zhl[i])
    
                    plt.plot(x[:boundary_l], zlr[i], color = 'mediumturquoise') #marker='o', ms=3, label='t={} yr'.format(i/365)
                    plt.plot(x[boundary_l:boundary_r], zhl[i], color = 'darkkhaki')
                    plt.plot(x[boundary_r:], zrr[i], color = 'mediumturquoise')
                    plt.plot(x[boundary_l - 1], zlr[i][-1], marker = 'o', ms = 7, color = 'coral')
                    plt.plot(x[boundary_r], zrr[i][0], marker = 'o', ms = 7, color = 'coral')
                
            else:
                # 計算領域から河川が消えた場合。
                for i in range(1, len(zhl[:h_dsp+1]), int(nmax/20)):
                    if i < r_dsp:
                        # 河川が消えるまでのタイムステップでのプロット
                        
                        # hillslopeの左端のインデックス
                        boundary_l = len(zlr[i])
                        # right_riverの最初のインデックス
                        boundary_r = len(zlr[i]) + len(zhl[i])
    
    
                        plt.plot(x[:boundary_l], zlr[i], color = 'mediumturquoise') #marker='o', ms=3, label='t={} yr'.format(i/365)
                        plt.plot(x[boundary_l:boundary_r], zhl[i], color = 'darkkhaki')
                        plt.plot(x[boundary_r:], zrr[i], color = 'mediumturquoise')
        #                 plt.plot(x[boundary_l - 1], zlr[i][-1], marker = 'o', ms = 7, color = 'coral')
                        plt.plot(x[boundary_r], zrr[i][0], marker = 'o', ms = 7, color = 'coral')
                    else:
                        # 計算途中でどちらかの河川が消失したタイムステップ以降のプロット.2領域の状況でまずforloop。最後に河川争奪した時の様子を追加プロット
                        
                        if zlr[r_dsp] == 0:
                            # 左側の河川が消失した場合
                            # hillslopeの左端のインデックス*領域の左端
                            boundary_l = 0
                            # right_riverの最初のインデックス
                            boundary_r = len(zhl[i])
    
    
    #                         plt.plot(x[:boundary_l], zlr[i], color = 'mediumturquoise') #marker='o', ms=3, label='t={} yr'.format(i/365)
                            plt.plot(x[boundary_l:boundary_r], zhl[i], color = 'darkkhaki')
                            plt.plot(x[boundary_r:], zrr[i], color = 'mediumturquoise')
            #                 plt.plot(x[boundary_l - 1], zlr[i][-1], marker = 'o', ms = 7, color = 'coral')
                            plt.plot(x[boundary_r], zrr[i][0], marker = 'o', ms = 7, color = 'coral')
                        
                        else:
                            # 右側の河川が消失した場合
                            # hillslopeの左端のインデックス
                            boundary_l = len(zlr[i])
                            # right_riverの最初のインデックス*領域の右端
                            boundary_r = len(x)
    
    
                            plt.plot(x[:boundary_l], zlr[i], color = 'mediumturquoise') #marker='o', ms=3, label='t={} yr'.format(i/365)
                            plt.plot(x[boundary_l:boundary_r], zhl[i], color = 'darkkhaki')
    #                         plt.plot(x[boundary_r:], zrr[i], color = 'mediumturquoise')
                            plt.plot(x[boundary_l - 1], zlr[i][-1], marker = 'o', ms = 7, color = 'coral')
    #                         plt.plot(x[boundary_r], zrr[i][0], marker = 'o', ms = 7, color = 'coral')
            if stream_cp != 0:
                # 河川争奪して計算領域が1つの河川になった時のプロット
                plt.plot(x, zrr[stream_cp], color = 'dodgerblue', label = "stream capture occured")
                    

        plt.grid(color="black", linestyle="dashed", linewidth=0.5)
        plt.xlim()
        plt.ylim(0, zmax)
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=30)
#         plt.show()
#         fig.savefig("img_i2_3000yr_kb2.8e-12_location.png")
#         fig.savefig("img_g1_5000yr_kb2.8e-12_location.png")
        fig.savefig(fname + ".png", bbox_inches='tight')
        fig.clear()
        plt.close(fig)
    #@profile 
    def plot_erosion_rate(self, x, l_erosion_rate, r_erosion_rate, r_dsp, h_dsp, stream_cp, fname, iterNum, emax):
        
        """侵食速度の時系列変化を可視化するための関数。引数は水平距離x(ndarray)、左側河川侵食速度の時系列list、右側河川の侵食速度の時系列list、
        河川がない状態の時間を示すindexを要素としたlist, hillslopeがない状態の時間を示すindexを要素としたlist, 河川争奪が起きたタイムステップを示すindex値、
        パスを含んだ出力用のファイル名。侵食速度の時系列リストは長さがnmaxで、中身は[0:nmax-1]が各水平地点での侵食速度を格納したndarrayで[nmax-1]が０。"""

                
        U = self.U
        nmax = self.nmax      
        emax /= U
        T = (nmax/365) * iterNum  # グラフ間で時間の表示を調整するための変数。累積時間を保持する事が目的。もっと簡略化？変形するなら (nmax*iterNum + i)/365
#         fig = plt.figure(figsize = (10, 5), dpi = 100)
        fig = plt.figure(figsize = (45, 10), dpi = 100)
        plt.rcParams["font.size"] = 22
        fig.patch.set_facecolor('white') 
        
        if r_dsp == 0 and h_dsp == 0:
            # r_dsp ~~ river dissapear
            for i in range(1, len(l_erosion_rate), int(nmax/20)):
                # hillslopeの最初のインデックス
                boundary_l = len(l_erosion_rate[i-1])
                
                # right_riverの最初のインデックス
#                 """エラーを起こすように設定"""
#                 boundary_r = len(x) - len(r_erosion_rate[100])
                boundary_r = len(x) - len(r_erosion_rate[i-1]) # 全体の数から右側河川領域の要素数だけ引くとそれ以外の領域の要素数になる→インデックスは０からなのでこれで右側河川の最初のインデックスになる。

                plt.plot(x[:boundary_l], l_erosion_rate[i-1] / U, color = cm.jet(i/nmax), label='t={:.3f} yr'.format(T + i/365)) #marker='o', ms=3, label='t={} yr'.format(i/365)
                plt.plot(x[boundary_r:], r_erosion_rate[i-1] / U, color = cm.jet(i/nmax))
#                 plt.plot(x, r_erosion_rate[i], color = cm.jet(i/nmax))
                
        else:
            # 計算領域から河川が消えた場合。
            for i in range(1, len(l_erosion_rate[:h_dsp+1]), int(nmax/20)):
                if i < r_dsp:
                    # 河川が消えるまでのタイムステップでのプロット
                    
                    # hillslopeの最初のインデックス
                    boundary_l = len(l_erosion_rate[i-1])

                    # right_riverの最初のインデックス
                    boundary_r = len(x) - len(r_erosion_rate[i-1]) # 全体の数から右側河川領域の要素数だけ引くとそれ以外の領域の要素数になる→インデックスは０からなのでこれで右側河川の最初のインデックスになる。

                    plt.plot(x[:boundary_l], l_erosion_rate[i-1] / U, color = cm.jet(i/nmax), label='t={:.3f} yr'.format(T + i/365)) #marker='o', ms=3, label='t={} yr'.format(i/365)
                    plt.plot(x[boundary_r:], r_erosion_rate[i-1] / U, color = cm.jet(i/nmax))
                else:
                    # 計算途中でどちらかの河川が消失したタイムステップ以降のプロット.2領域の状況でまずforloop。最後に河川争奪した時の様子を追加プロット
                    
                    if l_erosion_rate[r_dsp] == 0:
                        # 左側の河川が消失した場合
                        # right_riverの最初のインデックス
                        boundary_r = len(x) - len(r_erosion_rate[i-1]) # 全体の数から右側河川領域の要素数だけ引くとそれ以外の領域の要素数になる→インデックスは０からなのでこれで右側河川の最初のインデックスになる。

                        plt.plot(x[boundary_r:], r_erosion_rate[i-1] / U, color = cm.jet(i/nmax), label='t={:.3f} yr'.format(T + i/365))
                    else:
                        # 右側の河川が消失した場合
                        # hillslopeの左端のインデックス
                        boundary_l = len(l_erosion_rate[i-1])
                        
                        plt.plot(x[:boundary_l], l_erosion_rate[i-1] / U, color = cm.jet(i/nmax), label='t={:.3f} yr'.format(T + i/365)) #marker='o', ms=3, label='t={} yr'.format(i/365)
                        

            # 河川争奪して計算領域が1つの河川になった時のプロット
#             plt.plot(x, r_erosion_rate[stream_cp], color = 'dodgerblue', label = "stream capture occured")
                    

        plt.grid(color="black", linestyle="dashed", linewidth=0.5)
        plt.xlim()
        plt.ylim(0, emax+5)
        plt.xlabel("x")
        plt.ylabel("erosion rate / uplift rate")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=30)
#         plt.show()
#         fig.savefig("img_i2_3000yr_kb2.8e-12_location.png")
#         fig.savefig("img_g1_5000yr_kb2.8e-12_location.png")
        fig.savefig(fname + ".png", bbox_inches='tight')
        fig.clear()
        plt.close(fig)
        
#======================================================================================================================================================================================================

