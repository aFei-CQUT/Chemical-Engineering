import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import zipfile
import os

class HeatTransferDataProcesser:
    """
    传热分析类，用于加载数据、处理数据、生成拟合图和压缩结果。
    """
    def __init__(self, excel_file):
        """
        初始化函数，加载Excel文件并读取数据集。
        
        参数:
        excel_file (str): Excel文件路径
        """
        self.excel_file = excel_file  # Excel文件路径
        self.datasets = self.load_data()  # 加载数据集
        self.results = []  # 存储处理结果

    def load_data(self):
        """
        加载Excel文件中的数据，并将其转换为numpy数组。
        
        返回:
        datasets (list): 包含每个工作表数据的列表
        """
        excel_file = pd.ExcelFile(self.excel_file)  # 读取Excel文件
        sheet_names = excel_file.sheet_names[:2]  # 获取前两个工作表的名称
        datasets = []
        for sheet_name in sheet_names:
            sheet = pd.read_excel(self.excel_file, sheet_name=sheet_name, header=None)  # 读取工作表数据
            data = np.array(sheet.iloc[1:7, 1:4].values, dtype=float)  # 提取并转换数据
            datasets.append({
                'Δp_kb': data[:, 0],  # 孔板压差数据
                't_in': data[:, 1],   # 入口温度数据
                't_out': data[:, 2]   # 出口温度数据
            })
        return datasets

    def heat_transfer_data_processer_data_preprocess(self, Δp_kb, t_in, t_out):
        """
        传热数据预处理，计算相关参数并生成原始数据和计算结果表格。
        
        参数:
        Δp_kb (numpy.ndarray): 孔板压差数据
        t_in (numpy.ndarray): 入口温度数据
        t_out (numpy.ndarray): 出口温度数据
        
        返回:
        ans_original_data (pd.DataFrame): 原始数据表格
        ans_calculated_data (pd.DataFrame): 计算结果表格
        data_for_fit (numpy.ndarray): 用于拟合的数据
        """
        # 计算过程
        t_w = 98.4  # 壁面温度
        d_o = 0.022  # 外径
        d_i = d_o - 2 * 0.001  # 内径
        l = 1.2  # 长度
        n = 1  # 管数
        C_0 = 0.65  # 流量系数
        S_i = np.pi * l * n * d_i  # 内表面积
        S_o = np.pi * l * n * d_o  # 外表面积
        A_i = (np.pi * d_i ** 2) / 4  # 内截面积
        A_0 = 2.27 * 10 ** -4  # 孔板面积
        t_avg = 0.5 * (t_in + t_out)  # 平均温度
        Cp = 1005  # 比热容
        ρ = (t_avg ** 2) / 10 ** 5 - (4.5 * t_avg) / 10 ** 3 + 1.2916  # 密度
        λ = -(2 * t_avg ** 2) / 10 ** 8 + (8 * t_avg) / 10 ** 5 + 0.0244  # 导热系数
        μ = (-(2 * t_avg ** 2) / 10 ** 6 + (5 * t_avg) / 10 ** 3 + 1.7169) / 10 ** 5  # 动力粘度
        Pr = (Cp * μ) / λ  # 普朗特数
        PrZeroFour = Pr ** 0.4  # 普朗特数的0.4次方
        V_t = 3600 * A_0 * C_0 * np.sqrt((2 * 10 ** 3 * Δp_kb) / ρ)  # 体积流量
        V_xiu = ((t_avg + 273.15) * V_t) / (t_in + 273.15)  # 修正体积流量
        u_m = V_xiu / (3600 * A_i)  # 平均流速
        W_c = (ρ * V_xiu) / 3600  # 质量流量
        Q = Cp * W_c * (t_out - t_in)  # 热流量
        α_i = Q / (t_avg * S_i)  # 内表面换热系数
        Nu_i = (d_i * α_i) / λ  # 努塞尔数
        Re_i = (ρ * d_i * u_m) / μ  # 雷诺数
        NuOverPrZeroFour = Nu_i / Pr ** 0.4  # 努塞尔数与普朗特数的0.4次方之比
        Δt1 = t_w - t_in  # 温差1
        Δt2 = t_w - t_out  # 温差2
        Δt_m = (Δt2 - Δt1) / (np.log(Δt2) - np.log(Δt1))  # 平均温差
        K_o = Q / (Δt_m * S_o)  # 总传热系数

        # 创建原始数据表格
        ans_original_data = pd.DataFrame({
            "序号": np.arange(1, len(Δp_kb) + 1),
            "Δp孔板/kPa": Δp_kb,
            "t入/°C": t_in,
            "t出/°C": t_out
        })

        # 创建计算结果表格
        ans_calculated_data = pd.DataFrame({
            "序号": np.arange(1, len(Δp_kb) + 1),
            "Δp": Δp_kb,
            "t入": t_in,
            "t出": t_out,
            "t平": t_avg,
            "ρ": ρ,
            "λ": λ,
            "μ": μ,
            "Pr": Pr,
            "Pr^0.4": PrZeroFour,
            "V_t": V_t,
            "V修": V_xiu,
            "u_m": u_m,
            "W_c": W_c,
            "Q": Q,
            "α_i": α_i,
            "Nu_i": Nu_i,
            "Re_i": Re_i,
            "Nu/Pr^0.4": NuOverPrZeroFour,
            "Δt1": Δt1,
            "Δt2": Δt2,
            "Δt_m": Δt_m,
            "K_o": K_o
        }).T

        # 准备用于拟合的数据
        data_for_fit = np.array([Re_i, NuOverPrZeroFour]).T

        return ans_original_data, ans_calculated_data, data_for_fit

    @staticmethod
    def fit_func(x, a, b):
        """
        拟合函数，用于曲线拟合。
        
        参数:
        x (numpy.ndarray): 自变量
        a (float): 拟合参数
        b (float): 拟合参数
        
        返回:
        numpy.ndarray: 拟合结果
        """
        return a + b * x

    def process_data(self):
        """
        处理数据集，进行数据预处理和曲线拟合。
        """
        for idx, dataset in enumerate(self.datasets):
            Δp_kb = dataset['Δp_kb']
            t_in = dataset['t_in']
            t_out = dataset['t_out']
    
            ans_original_data, ans_calculated_data, data_for_fit = self.heat_transfer_data_processer_data_preprocess(Δp_kb, t_in, t_out)
            
            # 检查并过滤掉非正值
            valid_indices = (data_for_fit[:, 0] > 0) & (data_for_fit[:, 1] > 0)
            valid_data = data_for_fit[valid_indices]
            
            if len(valid_data) > 0:
                ans_params, _ = curve_fit(self.fit_func, np.log10(valid_data[:, 0]), np.log10(valid_data[:, 1]))
            else:
                print(f"警告：数据集 {idx+1} 没有有效的正值用于拟合。")
                ans_params = None
    
            self.results.append({
                'original_data': ans_original_data,
                'calculated_data': ans_calculated_data,
                'data_for_fit': valid_data,
                'params': ans_params
            })
            
    def print_results(self):
        """
        打印输出results字典的calculated_data键值。
        """
        # 设置显示选项
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        
        for idx, result in enumerate(self.results):
            print(f"数据集 {idx+1} 的计算结果:")
            print(result['calculated_data'])    

    def plot_fit(self, data_for_fit, filename, title):
        """
        绘制拟合图并保存为文件。
        
        参数:
        data_for_fit (numpy.ndarray): 用于拟合的数据
        filename (str): 保存文件名
        title (str): 图表标题
        """
        if len(data_for_fit) == 0:
            print(f"警告：没有有效数据用于绘制 {filename}")
            return

        ans_params, _ = curve_fit(self.fit_func, np.log10(data_for_fit[:, 0]), np.log10(data_for_fit[:, 1]))

        plt.figure(figsize=(8, 6), dpi=125)
        plt.scatter(data_for_fit[:, 0], data_for_fit[:, 1], color='r', label=r'$\mathrm{Data}')
        plt.plot(data_for_fit[:, 0], 10 ** self.fit_func(np.log10(data_for_fit[:, 0]), *ans_params), color='k', label=r'$\mathrm{Fit}')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\mathrm{Re}, fontsize=14, fontweight='bold')
        plt.ylabel(r'$\mathrm{Nu/Pr^{0.4}}, fontsize=14, fontweight='bold')
        plt.title(title, fontsize=10, fontweight='bold')
        plt.legend()

        plt.grid(True, which='both', linestyle='-', linewidth=1)
        plt.minorticks_on()

        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)

        plt.savefig(filename, dpi=300)
        
        plt.show()
        
        plt.close()
        
    def generate_plots(self):
        """
        生成拟合图并保存为文件。
        """
        plt.rcParams['font.family'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False
    
        # 绘制第一个数据集的图
        if self.results[0]['params'] is not None:
            self.plot_fit(self.results[0]['data_for_fit'], './拟合图结果/1.png', '无强化套管')
        else:
            print("警告：无法生成无强化套管的图，因为没有有效的拟合参数。")
    
        # 绘制第二个数据集的图
        if self.results[1]['params'] is not None:
            self.plot_fit(self.results[1]['data_for_fit'], './拟合图结果/2.png', '有强化套管')
        else:
            print("警告：无法生成有强化套管的图，因为没有有效的拟合参数。")
    
        # 绘制比较图
        plt.figure(figsize=(8, 6), dpi=125)
        
        if self.results[0]['params'] is not None:
            plt.scatter(self.results[0]['data_for_fit'][:, 0], self.results[0]['data_for_fit'][:, 1], color='r', label='无强化套管')
            plt.plot(self.results[0]['data_for_fit'][:, 0], 10 ** self.fit_func(np.log10(self.results[0]['data_for_fit'][:, 0]), *self.results[0]['params']), color='r', label='无强化套管拟合')
            
        if self.results[1]['params'] is not None:
            plt.scatter(self.results[1]['data_for_fit'][:, 0], self.results[1]['data_for_fit'][:, 1], color='b', label='有强化套管')
            plt.plot(self.results[1]['data_for_fit'][:, 0], 10 ** self.fit_func(np.log10(self.results[1]['data_for_fit'][:, 0]), *self.results[1]['params']), color='b', label='有强化套管拟合')
            
        if self.results[0]['params'] is None and self.results[1]['params'] is None:
            print("警告：无法生成比较图，因为两个数据集都没有有效的拟合参数。")
        else:
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\mathrm{Re}, fontsize=14, fontweight='bold')
            plt.ylabel(r'$\mathrm{Nu/Pr^{0.4}}, fontsize=14, fontweight='bold')
            plt.title('无强化套管 vs.有强化套管', fontsize=10, fontweight='bold')
            plt.legend()
    
            plt.grid(True, which='both', linestyle='-', linewidth=1)
            plt.minorticks_on()
    
            plt.gca().spines['top'].set_linewidth(2)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['right'].set_linewidth(2)
    
            plt.savefig('./拟合图结果/3.png', dpi=300)
            
            plt.show()
        plt.close('all')
        
    @staticmethod
    def compress_results():
        """
        压缩结果文件夹并保存为zip文件。
        """
        dir_to_zip = r'./拟合图结果'
        dir_to_save = r'./拟合图结果.zip'

        with zipfile.ZipFile(dir_to_save, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(dir_to_zip):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, dir_to_zip)
                    zipf.write(file_path, arc_name)

        print(f'压缩完成，文件保存为: {dir_to_save}')

# 使用示例
if __name__ == "__main__":
    file_path = '传热原始数据记录表(非).xlsx'
    # 实例化要分析的数据对象
    heat_transfer_data_processer = HeatTransferDataProcesser(file_path)
    # 处理数据
    heat_transfer_data_processer.process_data()
    # 结果可在heat_transfer_data_processer的results属性查看
    # results属性内calculated_data即为计算后结果
    # 打印结果，但太长了打印需要换行
    # heat_transfer_data_processer.print_results()
    # 结果绘图
    heat_transfer_data_processer.generate_plots()
    # 压缩结果便于分享
    heat_transfer_data_processer.compress_results()
