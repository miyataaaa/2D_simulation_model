#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from memory_profiler import profile
import file_operation as fl
Fl = fl.File()
#import joblib
get_ipython().run_line_magic('pdb', 'on')

# %pdb


class Initial_Topograpy:
    
    """様々な初期地形を形成するための関数をまとめたクラス。"""
    
    def __init__(self, **kwargs):
        self.dx = kwargs['dx']
        self.dt = kwargs['dt']
        self.nmax = kwargs['nmax']
        self.degree = kwargs['degree']
        self.th_ContA = kwargs['th_ContA']
        self.initTp = kwargs['initial topograpy']
        self.initDeg = kwargs['initial degree']
        self.initDiv = kwargs['initial divide']
        self.initLdeg = kwargs['initial Ldegree']
        self.initRdeg = kwargs['initial Rdegree']
        self.FileWords = kwargs['FileWords']
        self.Fpath = kwargs['Fpath']
    
    def select_initial_topograpy(self, **kwargs):
        
        """param_dict内のキーで初期地形関数を設定するためのwarapper関数"""
        
        initTp_function = self.initTp
        
        if initTp_function == "wholearea_degree":
            x, init_z = self.wholearea_degree()
        
        elif initTp_function == "eacharea_degree":
            x, init_z = self.eacharea_degree()
        
        elif initTp_function == "continue_any_file":
            x, init_z = self.continue_any_file(**kwargs)
        
        elif initTp_function == "create_terraces":
            x, init_z = self.create_terraces(**kwargs)
        
        elif initTp_function == "diff_baselevel":
            x, init_z = self.diff_baselevel(**kwargs)
            
        elif initTp_function == "variable_erea_size":
            x, init_z = self.variable_erea_size(**kwargs)
            
        else:
            x, init_z = self.initial_topograpy()
        
        return x, init_z

        
    def initial_topograpy(self):
        
        """初期地形を設定する関数のバージョン２。左右対称な初期地形で野根川支流でモデル対称として選定した場所と同等のサイズ感。jmaxは5400に設定してください"""
        
        jmax = 5400
        x = np.arange(0, jmax+self.dx, self.dx)
        z = np.array(x, dtype=float)
        for i in range(len(x)):
            if x[i] < 2700 :
                z[i] = 0.14 * x[i]
            else:
                z[i] = -0.14 * x[i] + 756
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return (x, z)
    
    
    def initial_asymmetry_topograpy(self):
        
        """左右非対称な初期地形を形成する関数。野根川支流をモデル化した。jmaxは4500に設定してください。"""
        
        jmax = 4500
        x = np.arange(0, jmax+self.dx, self.dx)
        z = np.array(x, dtype=float)
        for i in range(len(x)):
            if x[i] <= 1200 :
                z[i] = 0.21 * x[i] + 500
            else:
                z[i] = -0.228 * x[i] + 1026
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return (x, z)

    

    def initial_asymmetry_topograpy_3(self):
        """若干左右非対称な初期地形を形成する関数。左右の境界に標高差はない設定。jmaxは4200に設定してください。"""
        
        jmax = 4000
        x = np.arange(0, jmax+self.dx, self.dx)
        z = np.array(x, dtype=float)
        z = -0.125 * x + 500
        return (x, z)
    
    def wholearea_degree(self):
        
        """指定した位置に分水界を持ち、分水界の左右で初期勾配は同じ初期地形を形成する関数。分水界位置の指定は、
        全体領域に対する割合で行う。0が左側境界と一致。１が右側境界と一致"""
        
        initDeg = self.initDeg # 初期地形勾配
        initDiv = self.initDiv # 分水界水平位置を領域全体に対する割合で表現した時の値（左側境界が０）
        jmax = 4200 # 領域サイズ
        xd = jmax * initDiv #分水界の位置（左側境界からの距離）
        grad = np.tan(np.radians(initDeg))
        
        x = np.arange(0, jmax+self.dx, self.dx)
        z = np.array(x, dtype=float)
        
        z = np.where(x < xd, grad*((jmax-xd)+(x-xd)), grad*(jmax-x))
         # 標高値がマイナスになる場合は初期地形全体を平行移動させてから返す。
        zmin = z.min()
        if zmin != 0:
            z = z + abs(zmin) 
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return (x, z)

    def eacharea_degree(self):
        """指定した位置に分水界を持ち、分水界の左右で異なる勾配を指定できる初期地形形成関数。分水界位置の指定は、
        全体領域に対する割合で行う。0が左側境界と一致。１が右側境界と一致。"""
        Ldeg = self.initLdeg # 初期地形左側領域勾配
        Rdeg = self.initRdeg # 初期地形右側領域勾配
        initDiv = self.initDiv # 分水界水平位置を領域全体に対する割合で表現した時の値（左側境界が０）
        jmax = 4200 # 領域サイズ
        xd = jmax * initDiv #分水界の位置（左側境界からの距離）
        Lgrad = np.tan(np.radians(Ldeg))
        Rgrad = np.tan(np.radians(Rdeg))

        x = np.arange(0, jmax+self.dx, self.dx)
        z = np.array(x, dtype=float)

        z = np.where(x < xd, Rgrad*(jmax-xd)+Lgrad*(x-xd), Rgrad*(jmax-x))

        # 標高値がマイナスになる場合は初期地形全体を平行移動させてから返す。
        zmin = z.min()
        if zmin != 0:
            z = z + abs(zmin) 
        #         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return (x, z)
    
    def generate_x(self, z):
        
        """HDF5ファイルにはxを保存していないので、z, dxから復元する。"""
        
        dx = self.dx
        jmax = (len(z)-1) * dx
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        x = np.arange(0, jmax+dx, dx)
        
        return x
    
    def continue_any_file(self, **kwargs):
        
        """特定の条件で一度計算して得られた標高の時系列データの最終タイムステップでの値を初期地形として扱う関数。"""
        
        filename = Fl.get_hdffile(**kwargs)[0][1]
#         print(filename)
        dataset_names, group_names = Fl.datasetname_from_h5file(**kwargs)
#         print("dataset_names:\n {}, group_names:\n {}".format(dataset_names, group_names))
        sorted_datasets, head_nums = Fl.sorting_dataset(dataset_names[0], group_names[0], **kwargs)
#         print("\nsorted_datasets:\n {}".format(sorted_datasets))
        dataset = sorted_datasets[-1]
#         print("dataset:\n {}".format(dataset))
        z = Fl.z_from_h5file(fname=filename, dataset_name=dataset)[-1]
        x = self.generate_x(z)
        
        return (x, z)
    
    def create_terraces(self, **kwargs):
        
        """領域の左右を任意の高さだけ隆起させて、台地のような初期地形を形成する関数。一度計算したデータを初期地形とすることを
        前提としている。"""
        
        x, z = self.continue_any_file(**kwargs)
        ex_boundary_z = z[1:-1]
#         print(ex_boundary_z)
        z_terraces = ex_boundary_z + 300
        z_terraces = np.hstack((z[0], np.hstack((z_terraces, z[-1]))))
#         print(z_terraces)
        
        return (x, z_terraces)
    
    def diff_baselevel(self, **kwargs):
        
        """任意のベースレベル差と流域面積比を指定する初期地形形成関数。
        
        ~使用方法~
        ベースレベル差(diff baselevel)をパラメータ引数で指定する
        
        """
        
        zb = kwargs['diff baselevel'] # 左右の河川でのベースレベル差
        zd = 1000
        initDiv = self.initDiv # 分水界水平位置を領域全体に対する割合で表現した時の値（左側境界が０）
        jmax = 4200 # 領域サイズ
        xd = jmax * initDiv #分水界の位置（左側境界からの距離）
        dx = self.dx
        
        x = np.arange(0, jmax+dx, dx)
        z = np.where(x < xd, ((zd-zb)/xd)*x+zb , (zd/(jmax-xd))*(jmax-x))
        
        return x, z
        
    def variable_erea_size(self, **kwargs):
        
        """領域サイズを任意のサイズに指定できる初期地形形成関数。
        
        ~使用方法~
        領域サイズ(erea size)、ベースレベル差(diff baselevel)、初期勾配(initial degree)をパラメータ引数で指定する
        
        """
        
        dx = self.dx
        initDeg = self.initDeg #初期勾配（度数）
        jmax = kwargs['erea size'] #領域サイズ
        zb = kwargs['diff baselevel'] #ベースレベル差
        grad = np.tan(np.radians(initDeg))
        xd = (0.5*((jmax*grad) - zb)) / grad
        x = np.arange(0, jmax+dx, dx)
        z = np.where(x < xd, grad*x+zb, grad*(jmax-x))
        
        return x, z
        
# =============================================================================================================================
class Topograpy:
    
    """計算途中で形成された地形から分水界の位置、河川・斜面境界を計算する関数をまとめたクラス。侵食や隆起を計算するためのクラスではなく、計算に必要な地形パラメータをそろえるクラス。初期地形形成関数は別クラスとして分離した。
    ハックの法則の距離の取り方を流路長から水平方向のみに変更。分水界の左右の流域面積、勾配を求める関数もある"""
    
    def __init__(self, **kwargs):
        self.dx = kwargs['dx']
        self.ka = kwargs['ka']
        self.a = kwargs['a']
        self.th_ContA = kwargs['th_ContA']     
    
    def position_divide(self, x, z):
        #x, z = self.initial_topograpy()
        divide_id = z.argmax()
        x_left = x[:divide_id+1]
        z_left = z[:divide_id+1]
        x_right = x[divide_id:]
        z_right = z[divide_id:]
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        return divide_id, x_left, z_left, x_right, z_right

    def ContA_left(self, x, z):
        _, x_left, _, _ , _  = self.position_divide(x, z)
#         divide_id, x_left, _, _ , _  = self.position_divide(x, z)
        # 原点が河口の場合
        # 河口から分水界までのｘをスライス
#         river_length = np.array([(x_left[j+1]-x_left[j]) if j < divide_id  else 0 for j in range(0, len(x_left))])
#         inversion_river = np.flipud(river_length)

#         # 反転させたdistanceをインデックスが小さいほうから足し合わせ、それを反転させる
#         ContA_left = np.flipud(np.array([(self.ka * (np.sum(inversion_river[0:j+1]))**self.a) for j in range(len(x_left))]))
        
        dx = self.dx
        ka = self.ka
        a = self.a
        river_length = np.flipud(np.arange(0, len(x_left)*dx, dx))
        ContA_left = ka * (river_length ** a) 
        ContA_left = np.round(ContA_left, 4)
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return ContA_left 

    def ContA_right(self, x, z):
        
#         divide_id, _, _, x_right, _ = self.position_divide(x, z)
#         # xの最大値が分水界の右側の河川の河口の場合
#         # 河口から分水界までのｘをスライス
#         river_length = np.array([(x_right[j]-x_right[j-1]) if j >=1 
#                              else 0 for j in range(0, len(x_right))])
        
#         ContA_right = np.array([(self.ka *(np.sum(river_length[0:j+1]))**self.a) for j in range(len(x_right))])
        _, _, _, x_right, _ = self.position_divide(x, z)
        dx = self.dx
        ka = self.ka
        a = self.a
        river_length = np.arange(0, len(x_right)*dx, dx)
        ContA_right = ka * (river_length ** a) 
        ContA_right = np.round(ContA_right, 4)
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return ContA_right

    
    def grad_forward_difference(self, x, z):
        """前進差分で勾配を求める関数。分水界の右側の河川の勾配を計算したくて作成した。"""
        dx = self.dx
#         grad = np.array([abs(z[j+1]-z[j])/dx if j < len(x) -1 else 0 for j in range(len(z)) ])
        #grad = np.array([(z[j+1]-z[j])/(x[j+1]-x[j]) if j > 0  else 0 for j in range(len(x)) ])
        
        grad = np.abs(np.diff(z, append=1.14e-13)) / dx
        grad = np.round(grad, 4)
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return grad

    def grad_backward_difference(self, x, z):
        """後退差分で勾配を求める関数。分水界の左側の河川で勾配を計算したくて作成した。"""
        dx = self.dx
#         grad = np.array([abs(z[j]-z[j-1])/dx if j > 0  else 0 for j in range(len(z)) ])
#         print("\nforward_grad: {}".format(grad))
#         print("\nlen(forward_grad): \n{}".format(len(grad)))
        riverse_z = np.flipud(z)
        grad = np.abs(np.flipud(np.diff(riverse_z, append=0))) / dx
        grad = np.round(grad, 4)
        from IPython.core.debugger import Pdb; Pdb().set_trace()
        return grad

    def river_hillslope_boundary(self, x, z):
        
        """hillslopeの最初と最後のインデックスを返す関数。勾配と流域面積の関係で位置が変化する仕様ではなく、チャネル形成に必要な
        最低流域面積を決めておく仕様に変更した。"""
        th_ContA = self.th_ContA
#         _, x_left, z_left, _ , _  = self.position_divide(x, z)
#         _, _, _, x_right, z_right = self.position_divide(x, z)
        ContA_left = self.ContA_left(x, z)
        ContA_right = self.ContA_right(x, z)
        
        boundary_left = np.where(th_ContA >= ContA_left)
        boundary_right = np.where(th_ContA >= ContA_right)

        boundary_left_id = boundary_left[0][0]
        boundary_right_id = boundary_right[0][-1]
#         boundary_left = (ContA_left*(grad_left**nci)) >= Tci
#         boundary_right = (ContA_right*(grad_right**nci)) >= Tci
        
#         # 地点jよりも分水界側がすべてfalseであるidのndarrayの最初の要素を返す # 返り値の型はnumpy.int32
#         # 返り値のjは、hillslopeの最初と最後の地点を表すことになる。
#         boundary_left_id = np.array([j for j in range(len(boundary_left)) if np.sum(boundary_left[j:]) == 0])[0]
#         boundary_right_id = np.array([j-1 for j in range(1, len(boundary_right)) if np.sum(boundary_right[:j]) == 0])[-1]
        boundary_modified_right_id = (len(ContA_left)-1) + boundary_right_id
    
        
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return boundary_left_id, boundary_right_id, boundary_modified_right_id

#===========================================================================================================================================================================================================================


class Calc(Topograpy):
    
    """分水界より左側の河川、斜面領域、右側の河川での標高変化を計算するクラス。"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        hillslope部分のxの値も同時に抽出して返す。"""
        
        l_boundary, r_boundary, modifed_r_boundary = self.river_hillslope_boundary(x, z)
        
        # 3つの領域が存在するかしないかで場合分け。関数の返り値の数は3つの領域が存在する場合に合わせるので空の配列を返す処理を使っている。
        
        if l_boundary == 0:
            z_left_river = np.array([]) # 存在しないので空のndarrayを返す。
            z_hillslope = z[0 : modifed_r_boundary+2] # 境界値に川の領域の最後の要素を使うので、左側河川の最後の要素＋hillslope+右側河川の最後の要素を持つ配列
            z_right_river = z[modifed_r_boundary+1:] #右側の河川の領域だけ
            x_hillslope = x[0 : modifed_r_boundary+2]
            ContA_left_to_lboundary = np.array([]) # 存在しないので空のndarrayを返す
            ContA_right_from_rboundary = self.ContA_right(x, z)[r_boundary+1:]
#             from IPython.core.debugger import Pdb; Pdb().set_trace()            
        elif modifed_r_boundary == (len(z)-1):
            z_left_river = z[:l_boundary] # zはndarrayで要素はスカラーなのでコピーによって生成されるのはビューではなく独立したものになる。これは純粋に川の領域だけ
            z_hillslope = z[l_boundary-1 : ] # 境界値に川の領域の最後の要素を使うので、左側河川の最後の要素＋hillslope+右側河川の最後の要素を持つ配列
            z_right_river = np.array([]) #右側の河川の領域だけ。右側の河川はないのでからのndarrayを返す。
            x_hillslope = x[l_boundary-1 : ]
            ContA_left_to_lboundary = self.ContA_left(x, z)[:l_boundary]
    #         from IPython.core.debugger import Pdb; Pdb().set_trace()
            ContA_right_from_rboundary = np.array([]) # 存在しないので空のndarrayを返す
        else:
            z_left_river = z[:l_boundary] # zはndarrayで要素はスカラーなのでコピーによって生成されるのはビューではなく独立したものになる。これは純粋に川の領域だけ
            z_hillslope = z[l_boundary-1 : modifed_r_boundary+2] # 境界値に川の領域の最後の要素を使うので、左側河川の最後の要素＋hillslope+右側河川の最後の要素を持つ配列
            z_right_river = z[modifed_r_boundary+1:] #右側の河川の領域だけ
            x_hillslope = x[l_boundary-1 : modifed_r_boundary+2]
            ContA_left_to_lboundary = self.ContA_left(x, z)[:l_boundary]
    #         from IPython.core.debugger import Pdb; Pdb().set_trace()
            ContA_right_from_rboundary = self.ContA_right(x, z)[r_boundary+1:]
        
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return z_left_river, z_hillslope, z_right_river, ContA_left_to_lboundary, ContA_right_from_rboundary, x_hillslope
    
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
    
#     def threshold_z(self, x, degree):
        
#         """universal function化される事が前提の関数。thresholdでの標高値を返す"""
        
#         z = x * np.tan(np.radians(degree))

#         return z

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
    
    def oneside_faster_uplift(self, x, z):
        
        """分水界の左が設定した隆起速度の１０００倍の速度で隆起する。"""
        
        z_add_uplift = np.zeros(len(z))
        for j in range(1, len(z_add_uplift)-2):
            if j < 0.5 * len(z_add_uplift):
                z_add_uplift[j] = z[j] + (3 * self.U * self.dt)
            else:
                z_add_uplift[j] = z[j] + (self.U * self.dt)
                
        return z_add_uplift
    
    def tilting_uplift(self, x, z):
        
        """空間勾配を持った隆起速度。分水界左側の河口側境界が最も速く、右側の河口境界が最も遅い"""
        
        dt = self.dt
        
        ex_boundary_x = x[1:-1]
        ex_boundary_z = z[1:-1]
        intercept = 4.1e-11
        coef = -1.84e-15
        tilt_U = intercept + coef * ex_boundary_x
        z_add_uplift = ex_boundary_z + (tilt_U * dt)
        z_add_uplift = np.hstack((z[0], np.hstack((z_add_uplift, z[-1]))))
#         for j in range(1, len(z_add_uplift)-2):
#             z_add_uplift[j] = z[j] + (self.U * self.dt) + (0.03 * x[j])
        
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        return z_add_uplift
                
    def whole_uniform_uplift(self, x, z):
                                 
        """計算対象領域全体が同じ隆起速度で隆起する全体一様隆起を表現する関数"""
        
        U = self.U
        dt = self.dt
        
        ex_boundary_z = z[1:-1]
        z_add_uplift_ex = ex_boundary_z + (U * dt)
        #z_add_uplift = np.hstack((np.zeros(1), np.hstack((z_add_uplift, np.zeros(1)))))
        # 左右非対称クラスのための一時的操作
        z_add_uplift = np.hstack((z[0], np.hstack((z_add_uplift_ex, z[-1]))))
        
#         z_add_uplift = np.zeros(len(z))
#         for j in range(1, len(z_add_uplift)-2):
#             z_add_uplift[j] = z[j] + (self.U * self.dt)
        del ex_boundary_z, z_add_uplift_ex
                
        return z_add_uplift            
    #@profile    
    def do_computing(self, x, init_z, iterNum, uplift="uniform"):
        
        """計算を実行する関数。引数のinit_zは初期地形での標高ｚ(型はndarray)。
        この関数の返り値は各タイムステップの標高のndarrayを要素とする大きさnmaxで各要素が大きさjmaxの計算結果ｚ"""
        
        nmax = self.nmax
        T = (nmax/365) * iterNum  # 経過時間の表示を調整するための変数。累積時間を保持する事が目的。もっと簡略化？変形するなら (nmax*iterNum + i)/365
        z_array = [0]*nmax
        z_array[0] = init_z
        
        z_lriver_array = [0]*nmax
        z_hillslope_array = [0]*nmax
        z_rriver_array = [0]*nmax
        divide_id_array = [0]*nmax
        
        l_erosion_rate = [0]*nmax
        r_erosion_rate = [0]*nmax
        
        r_dsp_array = []  #左側の河川が消失したタイムステップインデックスを格納するlist。
        stream_cp = 0 #stram captureが起きたタイムステップインデックスを格納する変数。
#         ContA_left_array = [0]*nmax
#         ContA_right_array = [0]*nmax
#         grad_left_array = [0]*(nmax )
#         grad_right_array = [0]*(nmax )
        
#         divide_id_array[0] = init_z.argmax()
#         ContA_left_array[0] = self.ContA_left(x, init_z)
#         ContA_right_array[0] = self.ContA_right(x, init_z)
    
        for i in range(1, nmax):
            
            # タイムステップ1からで、zoldは1つ前のタイムステップの標高値なので3つの領域に分けるときはi-1
            zold = copy.deepcopy(z_array[i-1])
            divide_id_array[i-1] = zold.argmax()
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
                    r_erosion_rate[i-1] = self.river_right(zrr, ContA_right)[1]

                    # 侵食量と隆起量を計算
                    z_only_erosion = np.concatenate((self.threshold_hillslope(zh, xh, len_zlr, len_zrr), self.river_right(zrr, ContA_right)[0]))
                    if uplift == "uniform":
                        z_array[i] = self.whole_uniform_uplift(x, z_only_erosion)
                    elif uplift == "tilting":
                        z_array[i] = self.tilting_uplift(x, z_only_erosion)
                    else:
                        raise Exception("uncorrect uplift type")
        
                    if i % (nmax / 10) == 0:
                        print("Curently time step is t={}yr".format(T + i/365))

    #                 from IPython.core.debugger import Pdb; Pdb().set_trace()
                elif len(zrr) == 0:
                    # 右側の河川が消失した場合
            
                    # 可視化の為の操作
                    z_hillslope_array[i-1] = zh[1:]
                    z_lriver_array[i-1] = zlr
                    r_dsp_array.append(i-1)
                    
                    # 左側河川での侵食速度を計算
                    l_erosion_rate[i-1] = self.river_left(zlr, ContA_left)[1]

                    # 侵食量と隆起量を計算
                    z_only_erosion = np.concatenate((self.threshold_hillslope(zh, xh, len_zlr, len_zrr), self.river_left(zlr, ContA_left)[0]))
                    if uplift == "uniform":
                        z_array[i] = self.whole_uniform_uplift(x, z_only_erosion)
                    elif uplift == "tilting":
                        z_array[i] = self.tilting_uplift(x, z_only_erosion)
                    else:
                        raise Exception("uncorrect uplift type")
                        
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
                    l_erosion_rate[i-1] = self.river_left(zlr, ContA_left)[1]
                    
                    # 右側河川での侵食速度を計算
                    r_erosion_rate[i-1] = self.river_right(zrr, ContA_right)[1]

                    # 侵食量と隆起量を計算
                    z_only_erosion = np.concatenate((self.river_left(zlr, ContA_left)[0], self.threshold_hillslope(zh, xh, len_zlr, len_zrr), self.river_right(zrr, ContA_right)[0]))
                    if uplift == "uniform":
                        z_array[i] = self.whole_uniform_uplift(x, z_only_erosion)
                    elif uplift == "tilting":
                        z_array[i] = self.tilting_uplift(x, z_only_erosion)
                    else:
                        raise Exception("uncorrect uplift type")
                        
                    # 最終タイムステップの時
                    if i == (nmax-1):
                        divide_id_array[i] = self.position_divide(x, z_array[i])[0]
                        z_lriver_array[i], z_hillslope, z_rriver_array[i], _, _, _  = self.split_to_three_zone(x, z_array[i])
                        z_hillslope_array[i] = z_hillslope[1:-1]


                    if i % (nmax / 10) == 0:
                        print("Curently time step is t={}yr".format(T + i/365))
                        

        del z_only_erosion, zold
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        # 返り値はlistだが、各要素はndarray
        return z_array, divide_id_array, z_lriver_array, z_hillslope_array, z_rriver_array, r_dsp_array, stream_cp, l_erosion_rate, r_erosion_rate


    def convert_List_to_Numpy(self, z):
        
        """計算された標高値のndarrayを要素に持つリストをndarrayに変換する関数。河川争奪が起きたときに、リスト内に、標高値ndarrayと０が
        混在していたので、0を排除して河川争奪が起きるまでの標高値ndarrayだけを要素に持つndarrayをhdfファイルに保存するために作成した関数"""
        
        new_zarray_list = [z[i] for i in range(len(z)) if type(z[i]) == type(z[0])] 
        converted_z = np.array(new_zarray_list)
        
        return converted_z
    
    #def chi(self, x, z)
    
    
            
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

