import file_operation as fl
import pandas as pd
import numpy as np
import os
import copy
import matplotlib.pyplot as plt 
import matplotlib as mpl
from memory_profiler import profile
import sys
Fl = fl.File()
Kw = fl.Kwargs()

class analysis:
    
    def __init__(self):
        pass
    
    def compile_all_param(self, **kwargs):
        
        """
        
        指定した階層より下のパスに存在するparam_dict.h5ファイルの情報を１つのDataframeにして
        返す関数。
        
        ~usage~
        
        解析したいparam_dict.h5ファイルがある階層よりも上位の階層のパスをkey = "Fpath"
        のvalueとして持つ辞書型オブジェクトを引数に指定する。また、param_dict.h5のgroup名は'param_dict'
        である必要がある。２次元シミュレーションモデルを通じて生成されたデータならこれは気にする必要はない。
        
        ~end~
        
        """
        
        z_and_param = Fl.get_hdffile(**kwargs)
        dicts = []
        for i in range(len(z_and_param)):
            param_path = z_and_param[i][0]
            param = pd.read_hdf(param_path, 'param_dict')
            dicts.append(param)
            
        params = pd.concat(dicts, axis=1)

        # データフレームのindexを任意の文字列順に並べ替える。
        reindex = ['FileWords', 'initial topograpy', 'diff baselevel', "erea size", 'dx', 'dt', 'nmax', 
                   'iterNum', 'n', 'm', 'kd', 'degree', 'kb', 'ka', 'a', 'th_ContA', 'U', 'initial degree', 
                   'initial divide', 'initial Ldegree', 'initial Rdegree', 'Fpath', 'z_H5file_name', 'only_one_param']
        params = params.reindex(reindex, axis=0)
        
        # colums名が'value'になっているので、'FileWorgs'行の値で書き換える
        series = params.iloc[:1, :].loc['FileWords']
        new_colums = list(series.value)
        params = params.set_axis(new_colums, axis=1)
        
        # FileWordsは、colums名になったので削除。diff baselevleは重要なパラメータだが不正な値を含むので一旦削除する。
        params = params.drop(["FileWords", "diff baselevel"])
        params = params.T
        
        return params
    
    def add_diff_baselevel(self, **kwargs):
        
        z_and_param = Fl.get_hdffile(**kwargs)
        baselevels = []
        print("timestepNum: {}".format(len(z_and_param)))
        for i in range(len(z_and_param)):
            print("now: {}_{}".format(i+1, os.path.basename(z_and_param[i][1])))
            param_path = z_and_param[i][0]
            z_path = z_and_param[i][1]
        #     print(z_path)
            param = pd.read_hdf(param_path, 'param_dict')
            param = param.to_dict()['value']
            param.update(kwargs)
            dataset_names, group_names = Fl.datasetname_from_h5file_all(z_path)
        #     print(dataset_names, group_names)

            sorted_datasets, _ = Fl.sorting_dataset(dataset_names[0], group_names[0], **param)
            z = Fl.z_from_h5file(fname=z_path, dataset_name=sorted_datasets[0])
            baselevel = z[0][0] - z[0][-1]
            baselevels.append(baselevel)
            
        return baselevels
    
    def to_excel(self, param_pd, add_colums_data=[], add_colums_name="", **kwargs):
        
        p_path = kwargs['Fpath']
        f_path = os.path.join(p_path, 'complied_result.xlsx')
        print(f_path)
        if len(add_colums_data) == 0:
            param_pd.to_excel(f_path, na_rep="NaN")
        else:
            param_pd.insert(1, add_colums_name, add_colums_data)
#             from IPython.core.debugger import Pdb; Pdb().set_trace()
            param_pd.to_excel(f_path, na_rep="NaN")
            
    def z_and_param(self, i_dataset=0, **kwargs):
        
        """
        該当するファイル、データセットの標高値と時間幅dtを返す関数
        
        ~usage~
        １.結果を保存しているフォルダ群の一番上の階層のパスと、
           解析対象の標高値zファイルの拡張子抜きのファイル名を要素に持つ辞書(**kwargs)
        ２.何番目のデータセットかを示す整数値(i_dataset)
        ~end~
        
        """
        
        z_fname = Fl.get_hdffile(**kwargs)[0][1]
        updated_kwargs = Fl.param_dict_from_h5file(**kwargs)
        Fpath = updated_kwargs['Fpath']
        updated_kwargs.update(**kwargs)
        updated_kwargs['Fpath'] = Fpath
        updated_kwargs = Kw.change_to_numeric(**updated_kwargs)
        dataset_name, group_name = Fl.datasetname_from_h5file(**updated_kwargs)
        dataset_name, _ = Fl.sorting_dataset(dataset_name[0], group_name[0], **updated_kwargs)
        z = Fl.z_from_h5file(fname=z_fname, dataset_name=dataset_name[i_dataset])
        datasetNum = len(dataset_name)
        if i_dataset == 0:
            print("updated_dict: \n{}".format(updated_kwargs))
            print("dataset_names: \n{}".format(dataset_name))
            print("dataset_length: {}".format(datasetNum))
            
        return z, updated_kwargs, datasetNum
    
    def z_and_param_2(self, start=0, stop=-1, **kwargs):
        
        """
        該当するファイル、データセットの標高値と時間幅dtを返す関数
        
        ~usage~
        １.結果を保存しているフォルダ群の一番上の階層のパスと、
           解析対象の標高値zファイルの拡張子抜きのファイル名を要素に持つ辞書(**kwargs)
        ２.どのデータセットから(start)どのデータセットまで(stop)かを指定
        ~end~
        
        """
        
        z_fname = Fl.get_hdffile(**kwargs)[0][1]
        updated_kwargs = Fl.param_dict_from_h5file(**kwargs)
        Fpath = updated_kwargs['Fpath']
        updated_kwargs.update(**kwargs)
        updated_kwargs['Fpath'] = Fpath
        updated_kwargs = Kw.change_to_numeric(**updated_kwargs)
        dataset_name, group_name = Fl.datasetname_from_h5file(**updated_kwargs)
        dataset_name, _ = Fl.sorting_dataset(dataset_name[0], group_name[0], **updated_kwargs)
        z = Fl.z_from_datasets(fname=z_fname, datasets_list=dataset_name, start=start, stop=stop)
#         z = Fl.z_from_h5file(fname=z_fname, dataset_name=dataset_name[i_dataset])
        datasetNum = len(dataset_name)
#         if i_dataset == 0:
#             print("updated_dict: \n{}".format(updated_kwargs))
#             print("dataset_names: \n{}".format(dataset_name))
#             print("dataset_length: {}".format(datasetNum))
            
        return z, updated_kwargs, datasetNum
        
    def e_rate(self, z, dt, U):
        
        """
        
        1.データセット1つ分の標高値、型は2次元ndarray(z)
        2.時間幅(dt)
        3.隆起速度(U)
        
        ３つを与える事で、侵食速度(erosion rate)を計算する関数。
        単位はメートル、秒で統一。
        
        """
        # (timestpe i) - (timestep i+1)
        z_diff = z[:-1] - z[1:] 
#         print(z_diff)
        z_diff += U*dt
#         print(z_diff)
        e_rate = z_diff / dt
        e_rate /= U
        
        return e_rate
    
    def zd_id(self, z):
        
        """
        1つのデータセット分の標高値データから、各タイムステップでの分水界位置を求める関数。
        """
        len_erate = z.shape[0]
        zd_list = []
        zd_id = []
        for i in range(len_erate):
            zd = z[i].max()
            z_row = z[i]
            d_id = np.where(z_row == zd)
        #     print(d_id)
            zd_list.append(zd)
            zd_id.append(d_id[0][0])
        zd_ndarray = np.array(zd_list)
        zd_id = np.array(zd_id)
#         id_diff = np.diff(zd_id)
#         zd_sp = id_diff*dx/dt
        
        return zd_id, zd_ndarray
        
    def compile_zd_id(self, start=0, stop=-1, **kwargs):
        
        
        """
        複数のデータセットでの分水界標高と位置を示すidをそれぞれ１つのndarrayにして返す関数
        
        ~usage~
        １.結果を保存しているフォルダ群の一番上の階層のパスと、
           解析対象の標高値zファイルの拡張子抜きのファイル名を要素に持つ辞書(**kwargs)
        ２.どのデータセットから(start)どのデータセットまで(stop)かを指定。
           もしstart=0, stop=-1なら、すべてのデータセットで計算を実行。
        ~end~
        
        """
        z, param, datasetNum = self.z_and_param(i_dataset=start, **kwargs)
        zd_id, zd = self.zd_id(z)
        if start == 0 and stop == -1:
            iterNum = datasetNum
            print("iterNum: {}".format(iterNum))
            for i in range(1, iterNum):
                if i % (iterNum/20) == 0:
                    print("now: i={}".format(i))
                z_next, _, _ = self.z_and_param(i_dataset=i, **kwargs)
                zd_id_next, zd_next = self.zd_id(z_next)
                zd_id = np.concatenate([zd_id, zd_id_next], 0)
                zd = np.concatenate([zd, zd_next], 0)
        else: 
            iterNum = stop - start
            print("iterNum: {}".format(iterNum))
            for i in range(start+1, stop+1):
                if i % (iterNum/20) == 0:
                    print("now: i={}".format(i))
                z_next, _, _ = self.z_and_param(i_dataset=i, **kwargs)
                zd_id_next, zd_next = self.zd_id(z_next)
                zd_id = np.concatenate([zd_id, zd_id_next], 0)
                zd = np.concatenate([zd, zd_next], 0)
        
            
        return zd_id, zd, param
    
    def divide_height(self, **kwargs):
        
        fname = Fl.get_analysis_file(**kwargs)
        print("isfile: {}".format(os.path.isfile(fname))) 
        dataset_names, group_names = Fl.datasetname_from_h5file_all(fname)
        dataset_zd = ""
        for name in dataset_names:
            if name[0] == '/Divide height/value':
                dataset_zd = name[0]
        if dataset_zd == "":
            print("there is no dataset")
            sys.exit()
        dataset_zd = dataset_names[0][0]
        zd_ndarray = Fl.value_from_h5file(fname=fname, dataset_name=dataset_zd)
        
        return zd_ndarray
    
    def divide_position(self, **kwargs):
        
        fname = Fl.get_analysis_file(**kwargs)
        print("fname: {}".format(fname))
        print("isfile: {}".format(os.path.isfile(fname))) 
        dataset_names, group_names = Fl.datasetname_from_h5file_all(fname)
        print("dataset_names: {}".format(dataset_names))
        dataset_id = ""
        for name in dataset_names:
            if name[0] == '/Divide position/value':
                dataset_id = name[0]
        if dataset_id == "":
            print("there is no dataset")
            sys.exit()
        param = Fl.param_dict_from_h5file(**kwargs)
        dx = int(param['dx'])
        zd_id = Fl.value_from_h5file(fname=fname, dataset_name=dataset_id)
        zd_id = zd_id * dx
        
        print("dataset_names: {}".format(dataset_names))
        print("dataset_id: {}".format(dataset_id))
        
        return zd_id
    
    def extract_divide_position_height(self, zd_id, zd_ndarray):
        
        """
        全タイムステップでの分水界の位置と標高値から、ある条件の元でデータを選定し抽出する関数。
        全タイムステップでの値を計算や描画に使うとデータサイズが非常に大きく時間がかかるのでデータを削減する。
        分水界位置の近似曲線を求めてそこから分水界移動速度を求める上でも重要な工程。
        
        ~引数~
        zd_id -> 全タイムステップでの分水界位置を示すndarray
        zd_ndarray -> 全タイムステップでの分水界標高値を示すndarray
        
        ~抽出過程、条件の説明~
        縦軸：分水界位置、横軸；時間でプロットしてみたらわかるが、シミュレーションの特性によって
        分水界位置が時間の経過と共に階段状に変化する。（例:x=10がタイムステップ0~10まで続いてx=9がタイムステップ11~13まで続くみたいな。
        但し、分水界標高は一定ではない事に注意）そこで、同じ分水界位置が一定時間続く特性を利用して同じ分水界位置を示す時間幅から、
        (1).最初 (2).4分の1 (3).4分の3　(4).最後 -> この4つのタイムステップでの標高値と分水界位置を抜き出す。
        最後に抜き出したデータをそれぞれ1次元ndarrayにまとめて返す。
        こうする事で、分水界位置の変化の軌跡を保ちながらデータが全タイムステップ数から、ユニークな分水界位置の数×４に削減される。
        
        ~返り値~
        id_uqs ->　抽出したタイムステップでの分水界位置を格納したndarray
        zd_uqs -> 抽出したタイムステップでの分水界標高を格納したndarray
        uq_ts -> 抽出したタイムステップを格納したndarray
        *タイムステップの単位はシミュレーションで設定したdtになる（特に変えていなけれな1日単位のはず）
        """
        
        id_uniques = np.flipud(np.unique(zd_id))
        print("id_uniques: {}".format(id_uniques))
        id_uqs = id_uniques[0:1]
        zd_uqs = zd_ndarray[0:1]
        uq_ts = np.zeros(1)
        for id_uq in id_uniques:
            uq_t = np.where(zd_id==id_uq)
            print("uq_t: {}".format(uq_t))
            uq_t = np.where(zd_id==id_uq)[0]
            print("uq_t[0]: {}".format(uq_t))
            quarter = int(len(uq_t)/4)
            three_qua = quarter*3
            zd_uq_id = int(uq_t[0])
            zd_uq_1 = np.array([zd_ndarray[int(uq_t[0])]])
            zd_uq_2 = np.array([zd_ndarray[int(uq_t[0])+quarter]])
            zd_uq_3 = np.array([zd_ndarray[int(uq_t[0])+three_qua]])
            zd_uq_4 = np.array([zd_ndarray[int(uq_t[-1])]])
            uq_ts = np.concatenate((uq_ts, uq_t[0:1], uq_t[quarter:quarter+1], uq_t[three_qua:three_qua+1], uq_t[-1:]))
            zd_uqs = np.concatenate((zd_uqs, zd_uq_1, zd_uq_2, zd_uq_3, zd_uq_4))
            id_uq_array = np.array([id_uq])
            id_uqs = np.concatenate((id_uqs, id_uq_array, id_uq_array, id_uq_array, id_uq_array))
            
        print("zd_id_unique: {}".format(id_uniques))
        print("uq_ts: {}".format(uq_ts))
        print("id_uqs: {}".format(id_uqs))
        print("zd_uqs: {}".format(zd_uqs))
        print("uq_ts.shape: {}".format(uq_ts.shape))
        print("id_uqs.shape: {}".format(id_uqs.shape))
        print("zd_uqs.shape: {}".format(zd_uqs.shape))
        
        return id_uqs, zd_uqs, uq_ts
#     @profile
    def extract_divide_position_height_2(self, zd_id, zd_ndarray):
        
        """
        ********改善点**************************************************************
        extract_divide_position_height関数だと、分水界が行ったり来たりするような動きを見せたときに、
        zd_idが同じタイムステップが複数回存在する事が原因で期待する処理をしないので、np.where関数ではなく、np.diff関数で
        タイムステップ i とタイムステップ i+1で分水界位置が違う場所を求めた
        ***************************************************************************
        
        全タイムステップでの分水界の位置と標高値から、ある条件の元でデータを選定し抽出する関数。
        全タイムステップでの値を計算や描画に使うとデータサイズが非常に大きく時間がかかるのでデータを削減する。
        分水界位置の近似曲線を求めてそこから分水界移動速度を求める上でも重要な工程。
        
        ~引数~
        zd_id -> 全タイムステップでの分水界位置を示すndarray
        zd_ndarray -> 全タイムステップでの分水界標高値を示すndarray
        
        ~抽出過程、条件の説明~
        縦軸：分水界位置、横軸；時間でプロットしてみたらわかるが、シミュレーションの特性によって
        分水界位置が時間の経過と共に階段状に変化する。（例:x=10がタイムステップ0~10まで続いてx=9がタイムステップ11~13まで続くみたいな。
        但し、分水界標高は一定ではない事に注意）そこで、同じ分水界位置が一定時間続く特性を利用して同じ分水界位置を示す時間幅から、
        (1).最初 (2).4分の1 (3).4分の3　(4).最後 -> この4つのタイムステップでの標高値と分水界位置を抜き出す。
        最後に抜き出したデータをそれぞれ1次元ndarrayにまとめて返す。
        こうする事で、分水界位置の変化の軌跡を保ちながらデータが全タイムステップ数から、ユニークな分水界位置の数×４に削減される。
        
        ~返り値~
        id_uqs ->　抽出したタイムステップでの分水界位置を格納したndarray
        zd_uqs -> 抽出したタイムステップでの分水界標高を格納したndarray
        uq_ts -> 抽出したタイムステップを格納したndarray
        *タイムステップの単位はシミュレーションで設定したdtになる（特に変えていなけれな1日単位のはず）
        """
        print("zd_id: {}".format(zd_id))
        
#         diff_id = np.array([i for i in range(len(zd_id)-1) if zd_id[i]-zd_id[i+1]!=0])
            
        diff_id = np.diff(zd_id)
        # 分水界位置が、タイムステップi, i+1で違う時間を求める。これを各分水界位置での代表時間とする。
        time_rep = np.where(diff_id != 0)[0]
        print("time_rep: {}".format(time_rep))
        id_uqs = zd_id[0:1]
        zd_uqs = zd_ndarray[0:1]
        uq_ts = np.zeros(1)
        for i in range(len(time_rep)):
            if i==0:
                t_interval = time_rep[i]
            else:
                t_interval = time_rep[i]-time_rep[i-1]
            # 代表時間から、1/4時間前、2/4時間前、3/4時間前の時間を求める。これが、標高値や分水界位置の元配列から抜き出す時のインデックス
            t = time_rep[i]
            quarter = t - int(t_interval/4)
            two_qua = t - int(t_interval/4)*2
            three_qua = t - int(t_interval/4)*3
            # 標高値の元配列からスライスする
            zd_uq_1 = np.array([zd_ndarray[three_qua]])
            zd_uq_2 = np.array([zd_ndarray[two_qua]])
            zd_uq_3 = np.array([zd_ndarray[quarter]])
            zd_uq_4 = np.array([zd_ndarray[t]])
            # 分水界位置の元配列からスライスする。
            id_uq_1 = np.array([zd_id[three_qua]])
            id_uq_2 = np.array([zd_id[two_qua]])
            id_uq_3 = np.array([zd_id[quarter]])
            id_uq_4 = np.array([zd_id[t]])
            
            # スライスした物をそれぞれ結合する
            uq_ts = np.concatenate((uq_ts, np.array([three_qua]), np.array([two_qua]), np.array([quarter]), np.array([t])))
            zd_uqs = np.concatenate((zd_uqs, zd_uq_1, zd_uq_2, zd_uq_3, zd_uq_4))
            id_uqs = np.concatenate((id_uqs, id_uq_1, id_uq_2, id_uq_3, id_uq_4))
            
        print("uq_ts: {}".format(uq_ts))
        print("id_uqs: {}".format(id_uqs))
        print("zd_uqs: {}".format(zd_uqs))
        print("uq_ts.shape: {}".format(uq_ts.shape))
        print("id_uqs.shape: {}".format(id_uqs.shape))
        print("zd_uqs.shape: {}".format(zd_uqs.shape))
        
        return id_uqs, zd_uqs, uq_ts

    
    def polynomial_and_derivative(self, x, y, d):
        
        """
        numpy.polyfit関数で得られる係数値とその導関数を計算して返す関数。
        返り値は、近似多項式の係数の1次元多項式オブジェクトと、
        導関数の係数の1次元多項式オブジェクトを要素に持つタプル
        """
        
        pl_y = np.poly1d(np.polyfit(x, y, d))
#         dy = pl_y.deriv()
        
        return pl_y
    
    def migration_rate(self, poly_xd, t):
        
        x = poly_xd(t)
        diff_x = np.gradient(x)[1:]
        diff_t = np.gradient(t)[1:]
#         diff_x = np.diff(x)[2:]
#         diff_t = np.diff(t)[2:]
        migration_rate = diff_x / diff_t
        print("migration_rate: \n{}".format(migration_rate))
        return migration_rate
    
    def extract_z(self, uq_ts, nmax, **kwargs):
        
        """
        extract_divide_position_height_2関数で抽出したタイムステップでの標高値配列を元データから抽出して、
        １つの3次元配列にして返す関数。
        
        ~引数~
        1.抽出したタイムステップを示すndarray配列uq_ts(unique timestep)
        2.1つのデータセット内での行数を示すnmax
        
        """
        # 示すタイムステップがどのデータセットかを求める。
        i_dataset = uq_ts // nmax
        i_dataset_uq = np.unique(i_dataset)
        print("i_dataset: {}".format(i_dataset))
        print("i_dataset_uq: {}".format(i_dataset_uq))
        # 求めたデータセット内の何行目かを求める。これは配列になる。
        i_row = uq_ts % nmax
        print("i_row: {}".format(i_row))
        extract_z = []
        for i in range(len(i_dataset_uq)):
            z, _, _ = self.z_and_param(i_dataset=int(i_dataset_uq[i]), **kwargs)
#             z_i = z[int(i_row[i])]
            extract_z.append(z)
        
        extract_z = np.array(extract_z)
        
        return extract_z, i_dataset, i_dataset_uq, i_row
    
    def compile_erate(self, extract_z, i_dataset, i_dataset_uq, i_row, dt, U):
        
        """
        ~工程~
        1.抽出した特定のデータセットの標高値ndarrayからerate関数を使って侵食速度を復元する。(compiled_erate)
        2.復元した侵食速度ndarrayから、該当するタイムステップでの列を抽出する。(compiled_erate_ex)
        
        ~引数の説明~
        extract_z -> i_dataset_uqで示されるデータセットでの標高値ndarray構成される。なので3次元配列となっている。
        i_dataset -> extract_divide_position_height_2関数で抽出したのと
        　　　　　　　同じタイムステップがいくつめのdatasetかを示す数値格納したndarray
        i_dataset_uq -> i_datasetを構成する数値でユニークなものを示す配列
        i_row -> extract_divide_position_height_2関数で抽出したのと
        　　　　　同じタイムステップがいくつめのdatasetの何行目かを示す数値格納したndarray
        dt, U -> シミュレーションで用いたタイムステップ幅と隆起速度。（整数値型と浮動小数点型に変換してから渡す）
        
        ~返り値~
        複数の復元した特定のデータセット内のある行での侵食速度を1つの2次元ndarrayにまとめたもの。
        侵食速度はerate関数を使うので、正確には侵食速度÷隆起速度をした無次元侵食速度。
        """
        
        # 工程１
        compiled_erate = []
        print("iterNum for compile erate: {}".format(extract_z.shape[0]))
        for i in range(extract_z.shape[0]):
            print("now i={}".format(i))
            erate = self.e_rate(extract_z[i], dt, U)
            compiled_erate.append(erate)
        
        # 工程２
        compiled_erate = np.array(compiled_erate)
        compiled_erate_ex = []
        for i in range(len(i_dataset_uq)):
            same_dataset_id = np.where(i_dataset==i_dataset_uq[i])
#             print("same_dataset_id: {}".format(same_dataset_id))
#             print("len_same_dataset_id: {}".format(len(same_dataset_id[0])))
            for j in range(len(same_dataset_id[0])):
#                 print("j={}_i_row[same_dataset_id[0][j]]:{}".format(j, i_row[same_dataset_id[0][j]]))
                row = int(i_row[same_dataset_id[0][j]]) - 1
                compiled_erate_ex.append(compiled_erate[i][row])       
        compiled_erate_ex = np.array(compiled_erate_ex)   
        
        return compiled_erate_ex
    
    def compile_erate_2(self, dt, U, uq_ts, nmax, **kwargs):
        
        """
        !!!! compile_erate関数ではメモリが足りなかったので、extraxt_z関数とcompile_erate関数をマージした!!!!!
        -> 改善点としては、extraxt_z関数で標高値を抜き出して1つの3次元配列にしていたが、
        　それがメモリを食っている可能性大だったのでそれを作らないようにした。
        
        ~工程~
        1.抽出した特定のデータセットの標高値ndarrayからerate関数を使って侵食速度を復元する。(compiled_erate)
        2.復元した侵食速度ndarrayから、該当するタイムステップの行を抽出する。(compiled_erate_ex)
        
        ~引数の説明~
        extract_z -> i_dataset_uqで示されるデータセットでの標高値ndarray構成される。なので3次元配列となっている。
        i_dataset -> extract_divide_position_height_2関数で抽出したのと
        　　　　　　　同じタイムステップがいくつめのdatasetかを示す数値格納したndarray
        i_dataset_uq -> i_datasetを構成する数値でユニークなものを示す配列
        i_row -> extract_divide_position_height_2関数で抽出したのと
        　　　　　同じタイムステップがいくつめのdatasetの何行目かを示す数値格納したndarray
        dt, U -> シミュレーションで用いたタイムステップ幅と隆起速度。（整数値型と浮動小数点型に変換してから渡す）
        
        ~返り値~
        複数の復元した特定のデータセット内のある行での侵食速度を1つの2次元ndarrayにまとめたもの。
        侵食速度はerate関数を使うので、正確には侵食速度÷隆起速度をした無次元侵食速度。
        """
        # 示すタイムステップがどのデータセットかを求める。
        i_dataset = uq_ts // nmax
        i_dataset_uq = np.unique(i_dataset)
        print("i_dataset: {}".format(i_dataset))
        print("i_dataset_uq: {}".format(i_dataset_uq))
        # 求めたデータセット内の何行目かを求める。これは配列になる。
        i_row = uq_ts % nmax
        print("i_row: {}".format(i_row))
        z_sample, _, _ = self.z_and_param(i_dataset=int(i_dataset_uq[0]), **kwargs)
        compiled_erate_ex = np.zeros(z_sample.shape[1])
        print("compiled_erate_ex.shape: {}".format(compiled_erate_ex.shape))
        print("compiled_erate_ex: \n{}".format(compiled_erate_ex))
        del z_sample
        print("\niterNum: {}".format(i_dataset.shape))
        iterNum = 1
        for i in range(len(i_dataset_uq)):
            z, _, _ = self.z_and_param(i_dataset=int(i_dataset_uq[i]), **kwargs)
            erate = self.e_rate(z, dt, U)
            same_dataset_id = np.where(i_dataset==i_dataset_uq[i])
#             print("same_dataset_id: {}".format(same_dataset_id))
#             print("len_same_dataset_id: {}".format(len(same_dataset_id[0])))
            for j in range(len(same_dataset_id[0])):
                print("now={} i_row[same_dataset_id[0][j]]:{}".format(iterNum, i_row[same_dataset_id[0][j]]))
                row = int(i_row[same_dataset_id[0][j]]) - 1
#                 compiled_erate_ex.append(erate[row]) 
                compiled_erate_ex = np.vstack([compiled_erate_ex, erate[row]])
                iterNum += 1
       
        compiled_erate_ex = compiled_erate_ex[1:]
#         compiled_erate_ex = np.array(compiled_erate_ex)   
        print("compiled_erate_ex.shape: {}".format(compiled_erate_ex.shape))
        return compiled_erate_ex
    
    def split_erate(self, erate, id_uqs, x, ka, a, th_ContA, dx):
        
        ContA_x = (th_ContA/ka)**(1/a)
        ContA_x = int(ContA_x // dx) + 2
        iterNum = erate.shape[0]
        l_river_erate = []
        r_river_erate = []
        for i in range(iterNum):
            zd_id = int(id_uqs[i] / dx)
            l_bd = zd_id - ContA_x + 1
            r_bd = zd_id + ContA_x 
            l_river_erate.append(erate[i][:l_bd])
            r_river_erate.append(erate[i][r_bd:])
        
        return l_river_erate, r_river_erate
    
    def calc_left_right_diff(self, l_river_erate, r_river_erate):
        
        Head_erosion_lrDiff = []
        Mean_erosion_lrDiff = []
        for left, right in zip(l_river_erate, r_river_erate):
            if len(left) == 0:
                head_lr_diff = right[0]
                mean_lr_diff = right.mean()
                Head_erosion_lrDiff.append(head_lr_diff)
                Mean_erosion_lrDiff.append(mean_lr_diff)
            elif len(right) == 0:
                head_lr_diff = left[-1]
                mean_lr_diff = left.mean()
                Head_erosion_lrDiff.append(head_lr_diff)
                Mean_erosion_lrDiff.append(mean_lr_diff)
            else:
                head_lr_diff = right[0] - left[-1]
                mean_lr_diff = right.mean() - left.mean()
                Head_erosion_lrDiff.append(head_lr_diff)
                Mean_erosion_lrDiff.append(mean_lr_diff)
            
        Head_erosion_lrDiff = np.array(Head_erosion_lrDiff)
        Mean_erosion_lrDiff = np.array(Mean_erosion_lrDiff)
        
        return Head_erosion_lrDiff, Mean_erosion_lrDiff
    
#     @profile
    def value_for_plot(self, **kwargs_El):
        
        kwargs_Al = Kw.change_Elevation_to_Analysis(**kwargs_El)
        param = Fl.param_dict_from_h5file(**kwargs_Al)
        iterNum = int(param['iterNum'])
        nmax = int(param['nmax'])
        dt = int(param['dt'])
        U = float(param['U'])
        zd_id = self.divide_position(**kwargs_Al)
        zd_h = self.divide_height(**kwargs_Al)
        print("\nnow extract divide-(position, heght) and timesteps\n")
        id_uqs, zd_uqs, uq_ts = self.extract_divide_position_height_2(zd_id, zd_h)
        print("\nnow compile erosion rate\n")
        compiled_erate_ex = self.compile_erate_2(dt, U, uq_ts, nmax, **kwargs_El)

        print("\nnow calc migration rate\n")
        uq_ts = uq_ts / 365
        d = 10
        pl_id = self.polynomial_and_derivative(uq_ts, id_uqs, d)
        mr = self.migration_rate(pl_id, uq_ts)
        
        print("\nnow calc diff between left and right erosion rate\n")
        dx = int(param['dx'])
        jmax = (len(compiled_erate_ex[0])-1) * dx
        x = np.arange(0, jmax+dx, dx)
        a = float(param['a'])
        ka = float(param['ka'])
        th_ContA = float(param['th_ContA'])
        l_river_erate, r_river_erate = self.split_erate(compiled_erate_ex, id_uqs, x, ka, a, th_ContA, dx)
        Head_erosion_lrDiff, Mean_erosion_lrDiff = self.calc_left_right_diff(l_river_erate, r_river_erate)
        
        print("\n\nfinish")
        return id_uqs, zd_uqs, uq_ts, pl_id, d, mr, Head_erosion_lrDiff, Mean_erosion_lrDiff
    
#     @profile
    def plot(self, xd, zd, t, poly_xd, d, migration_rate, Head_erosion_lrDiff, Mean_erosion_lrDiff, fname):
        
#         t = t / 365
#         print("t :\n{}\nxd :\n{}\n zd :\n{}\n poly_xd :\n{}\n d :\n{}\nmigration_rate :\n{}\nHead_erosion_lrDiff :\n{}\nMean_erosion_lrDiff :\n{}\n"
#               .format(t, xd, zd, poly_xd, d, migration_rate, Head_erosion_lrDiff, Mean_erosion_lrDiff))
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['axes.titlesize'] = 17
        fig = plt.figure(figsize=(20, 10)) #figsize=(70, 10)
        ax_zd = fig.add_subplot(3, 2, 1)
        ax_id = fig.add_subplot(3, 2, 3)
        ax_zx = fig.add_subplot(3, 2, 5)
        ax_mr = fig.add_subplot(3, 2, 2)
        ax_he = fig.add_subplot(3, 2, 4)
        ax_me = fig.add_subplot(3, 2, 6)
        
#         ax_zd.plot(t, zd, color="slateblue", label="raw data", ls="None", marker='o')
#         ax_id.plot(t, xd, color="darkorchid", label="raw data", ls="None", marker='o')
# #         ax_id.plot(t, poly_xd(t),
# #                    ls="dashed", marker="o", ms=3, color="grey", label="polynomialfit d="+str(d))
#         ax_id.plot(t, poly_xd(t),
#                    ls="dashed", color="plum", label="polynomialfit d="+str(d))
#         ax_zx.plot(xd, zd, color="deeppink", label="raw data", ls="None", marker='o')
#         ax_mr.plot(t[1:], -1000*migration_rate, color="crimson", label="from approximation", ls="None", marker='o')
#         ax_he.plot(t, Head_erosion_lrDiff, color="darkorange", ls="None", marker='o')
#         ax_me.plot(t, Mean_erosion_lrDiff, color="gold", ls="None", marker='o')
        ax_zd.plot(t, zd, color="slateblue", label="raw data")
        ax_id.plot(t, xd, color="darkorchid", label="raw data")
        ax_id.plot(t, poly_xd(t),
                   ls="dashed", color="plum", label="polynomialfit d="+str(d))
        ax_zx.plot(xd, zd, color="deeppink", label="raw data")
        ax_mr.plot(t[1:], -1000*migration_rate, color="crimson", label="from approximation")
        ax_he.plot(t, Head_erosion_lrDiff, color="darkorange")
        ax_me.plot(t, Mean_erosion_lrDiff, color="gold")

        ax_zd.set_title("Divide height")
        ax_zd.set_xlabel("time [yr]")
        ax_zd.set_ylabel("z [m]")
        ax_zd.legend()

        ax_id.set_title("Divide position")
        ax_id.set_xlabel("time [yr]")
        ax_id.set_ylabel("x [m]")
        ax_id.legend()

        ax_zx.set_title("Divide position-height")
        ax_zx.set_xlabel("x [m]")
        ax_zx.set_ylabel("z [m]")
        ax_zx.set_xlim(0, 4250)
        ax_zx.set_ylim(0, zd.max()+50)
        ax_zx.legend()
        
        ax_mr.set_title("Divide migration rate")
        ax_mr.set_xlabel("time [yr]")
        ax_mr.set_ylabel("migration rate [mm/yr]")
        ax_mr.legend()
        
        ax_he.set_title("Difference between left and right head erosion rate")
        ax_he.set_xlabel("time [yr]")
        ax_he.set_ylabel("left - right")
#         ax_he.legend()
        
        ax_me.set_title("Difference between left and right mean erosion rate")
        ax_me.set_xlabel("time [yr]")
        ax_me.set_ylabel("left - right")
#         ax_me.legend()

        fig.subplots_adjust(hspace=0.5, wspace=0.2)

        plt.show()

        plt.style.use('seaborn-white')
        
        isfile = os.path.isfile(fname+".png")
        if isfile:
            os.remove(fname+".png")
        fig.savefig(fname + ".png", bbox_inches='tight')
        plt.clf()
        plt.close()

    def compile_zdid_and_saving(self, **kwargs):
        
        """
        compile_zd_id関数、save_analysis関数のラッパ
        
        ~usage~
        １.結果を保存しているフォルダ群の一番上の階層のパスと、
           解析対象の標高値zファイルの拡張子抜きのファイル名を要素に持つ辞書(**kwargs)
        ~end~
        
        """
        zd_id, zd, param = self.compile_zd_id(start=0, stop=-1, **kwargs)
        group_names = ['Divide position', 'Divide height']
        values = [zd_id, zd]
        Fl.save_analysis(group_names=group_names, values=values, **param)
        
    def calc_and_plot(self, **kwargs_El):
        
        """
        value_for_plot関数、plot関数、save_analysis関数のラッパ関数。
        
        具体的には、compile_zdid_and_saving関数で分水界位置・標高を計算して、保存したHFDファイルから値を取り出し、
        そこから、データ数を削減するために特定のタイムステップでの値から分水界の移動速度や左右河川の侵食速度差等を
        計算して、comile_zdid_and_saving関数で作成したHDFファイル内の新たなgroupとして保存する。
        また、同時にグラフも作成。
        
        ~引数~
        １.結果を保存しているフォルダ群の一番上の階層のパスと、
           解析対象の標高値zファイルの拡張子抜きのファイル名を要素に持つ辞書(**kwargs)
           **辞書に持たせるhdfファイル名は、analysis.h5ではなく、標高値の方である事に注意。
        """
        id_uqs, zd_uqs, uq_ts, pl_id, d, mr, Head_erosion_lrDiff, Mean_erosion_lrDiff = self.value_for_plot(**kwargs_El)
        fname_plot = Fl.fname_for_Analysisplot(**kwargs_El)
        self.plot(id_uqs, zd_uqs, uq_ts, pl_id, d, mr, Head_erosion_lrDiff, Mean_erosion_lrDiff, fname_plot)
        kwargs_Al = Kw.change_Elevation_to_Analysis(**kwargs_El)
        fname_Al = Fl.get_hdffile(**kwargs_Al)[0][2]
        print("\nfname_Al: {}".format(fname_Al))
        group_names = ['Ex Divide position', 'Ex Divide height', 'Ex time step', 
               'Divide migration rate', 'Head Erate diff', 'Mean Erate diff']
        values=[id_uqs, zd_uqs, uq_ts, mr, Head_erosion_lrDiff, Mean_erosion_lrDiff ]
        Fl.do_save(fname=fname_Al, group_names=group_names, values=values)
        
    def do_analysis(self, function="", files=[]):
        
        kwargs = {"Fpath" : r"C:\Users\miyar\NUMERICAL FLUID COMPUTATION METHOD\result_img", # 結果を保存するファイルのディレクトリパス
                  "z_H5file_name" : ""}
        
        for file in files:
            kwargs["z_H5file_name"] = file
            print("now: {}".format(file))
            
            try:
                if function == "compile_zdid_and_saving":
                    self.compile_zdid_and_saving(**kwargs)
                if function == "calc_and_plot":
                    self.calc_and_plot(**kwargs)
            except:
                print("\nno file, name is = {}\n".format(file))
                print("maybe you not do compile_zdid_and_saving or get a file's name wrong?")
                print("please try function only that you selected")
                continue

if __name__ == "__main__":
    

    files = [
             "Elevation_0_(initial topograpy, diff baselevel) == ('diff_baselevel', 500)"
             ]

    
    Al = analysis()
    Al.do_analysis(function="calc_and_plot", files=files)
    
#"Elevation_0_(initial topograpy, erea size) == ('variable_erea_size', 4200)",
