import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
import pickle
import matplotlib.image as mpimg

class DryingDataProcessor:
    def __init__(self, file_path):
        """
        初始化类 DryingDataProcessor 的实例。

        参数:
        file_path (str): Excel 文件的路径。
        """
        self.file_path = file_path  # Excel 文件路径
        self.results = {}  # 存储处理结果
        self.setup_plot()  # 设置绘图参数

    def setup_plot(self):
        """
        设置绘图的全局参数。
        """
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 125

    def load_data(self):
        """
        加载 Excel 文件中的数据，并将其分配给相应的变量。
        """
        excel_file = pd.ExcelFile(self.file_path)
        self.sheet_names = excel_file.sheet_names  # 获取工作表名称列表

        # 读取第一个工作表的数据
        df1 = pd.read_excel(self.file_path, header=None, sheet_name=self.sheet_names[0])
        data1 = df1.iloc[:, 1].values  # 提取数据部分

        # 提取各列数据
        self.m_1 = data1[0] * 1e-3  # 初始质量 (kg)
        self.m_2 = data1[1] * 1e-3  # 最终质量 (kg)
        self.W2 = data1[2] * 1e-3  # 水分含量 (kg)
        self.G_prime = data1[3] * 1e-3  # 干燥速率 (kg/h)
        self.ΔP = data1[4]  # 压差 (Pa)

        # 读取第二个工作表的数据
        df2 = pd.read_excel(self.file_path, header=None, sheet_name=self.sheet_names[1])
        data2 = df2.iloc[1:, 1:].values  # 提取数据部分

        self.τ = data2[:, 0] / 60  # 时间 (h)
        self.W1 = data2[:, 1] * 1e-3  # 水分含量 (kg)
        self.t = data2[:, 2]  # 温度 (℃)
        self.tw = data2[:, 3]  # 湿球温度 (℃)

        self.r_tw = 2490  # 水的汽化潜热 (kJ/kg)
        self.S = 2.64 * 1e-2  # 干燥面积 (m^2)

    def preprocess_data(self):
        """
        进行数据预处理和计算。
        """
        self.τ_bar = (self.τ[:-1] + self.τ[1:]) / 2  # 平均时间 (h)
        self.G = (self.W1 - self.W2)  # 干燥速率 (kg)
        self.X = (self.G - self.G_prime) / self.G_prime  # 干基水分含量

        self.ans1 = np.array([self.G * 1000, self.X]).T  # 干燥速率和干基水分含量

        self.X_bar = (self.X[:-1] + self.X[1:]) / 2  # 平均干基水分含量
        self.U = -(self.G_prime / self.S) * (np.diff(self.X) / np.diff(self.τ))  # 干燥速率 (kg/m^2/h)

        self.U_c = np.mean(self.U[15:])  # 平均干燥速率

        self.ans2 = np.array([self.X_bar, self.U]).T  # 平均干基水分含量和干燥速率

        self.results.update({
            'ans1': self.ans1.tolist(),
            'ans2': self.ans2.tolist(),
            'τ_bar': self.τ_bar.tolist(),
            'X_bar': self.X_bar.tolist(),
            'U': self.U.tolist(),
            'U_c': float(self.U_c)
        })

    def plot_drying_curve(self):
        """
        绘制干燥曲线 (X vs τ)。
        """
        plt.figure(figsize=(8, 6), dpi=125)
        plt.scatter(self.τ_bar, self.X_bar, marker='o', color='r', label='平均拟合')
        plt.plot(self.τ_bar, self.X_bar, linestyle='-', color='k', label='平均拟合')
        plt.title("干燥曲线", fontsize=12)
        plt.xlabel(r"$\tau/h$", fontsize=12)
        plt.ylabel(r"$X/(kg·kg^{-1}$干基)", fontsize=12)
        plt.grid(True, which='both')
        plt.minorticks_on()

        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['top'].set_linewidth(2)

        plt.xlim(left=0)
        plt.ylim(bottom=0)

        plt.legend(fontsize=14)
        
        if not os.path.exists('./拟合图结果'):
            os.makedirs('./拟合图结果')
        
        plt.savefig('./拟合图结果/1.png', dpi=300)
        plt.show()

    def plot_drying_rate_curve(self):
        """
        绘制干燥速率曲线 (U vs X)。
        """
        plt.figure(figsize=(8, 6), dpi=125)
        plt.scatter(self.X_bar, self.U, marker='o', color='r')
        plt.title(r"干燥速率曲线", fontsize=12)
        plt.xlabel(r"$X/(kg·kg^{-1}$干基)", fontsize=12)
        plt.ylabel(r"$U/(kg·m^{-2}·h^{-1})$", fontsize=12)
        plt.grid(True, which='both')
        plt.minorticks_on()

        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['top'].set_linewidth(2)

        plt.xlim(left=0)
        plt.ylim(bottom=0)

        plt.savefig('./拟合图结果/2.png', dpi=300)
        plt.show()

    def further_calculations(self):
        """
        进行进一步的计算。
        """
        C_0 = 0.65  # 常数
        A_0 = (np.pi * 0.040 ** 2) / 4  # 面积 (m^2)
        ρ_空气 = 1.29  # 空气密度 (kg/m^3)
        t0 = 25  # 初始温度 (℃)

        self.α = (self.U_c * self.r_tw) / (self.t - self.tw)  # 传热系数
        self.V_t0 = C_0 * A_0 * np.sqrt(2 * self.ΔP / ρ_空气)  # 初始体积流量 (m^3/s)
        self.V_t = self.V_t0 * (273 + self.t) / (273 + t0)  # 体积流量 (m^3/s)

        self.results.update({
            'α': self.α.tolist(),
            'V_t0': float(self.V_t0),
            'V_t': self.V_t.tolist()
        })

    def integrate_images(self):
        """
        将绘制的图像整合到同一页中。
        """
        images = []
        for i in range(1, 3):
            img = mpimg.imread(f'./拟合图结果/{i}.png')
            images.append(img)

        fig, axes = plt.subplots(2, 1, figsize=(12, 12), dpi=125)
        for ax, img in zip(axes.flatten(), images):
            ax.imshow(img)
            ax.axis('off')

        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.savefig(r'./拟合图结果/拟合图整合图.png', bbox_inches='tight', dpi=300)
        plt.show()

    def compress_results(self):
        """
        压缩绘制的图像结果。
        """
        dir_to_zip = r'./拟合图结果'
        dir_to_save = r'./拟合图结果.zip'

        with zipfile.ZipFile(dir_to_save, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(dir_to_zip):
                for file in files:
                    file_dir = os.path.join(root, file)
                    arc_name = os.path.relpath(file_dir, dir_to_zip)
                    zipf.write(file_dir, arc_name)

        print(f'压缩完成，文件保存为: {dir_to_save}')

    def serialize_results(self):
        """
        序列化计算结果并保存到文件。
        """
        with open('results.pkl', 'wb') as f:
            pickle.dump(self.results, f)

    def format_results(self):
        """
        格式化输出计算结果。
        """
        for key, value in self.results.items():
            print(f"{key}: {value}")

    def run_drying_data_processor(self):
        """
        运行整个分析流程。
        """
        self.load_data()
        self.preprocess_data()
        self.plot_drying_curve()
        self.plot_drying_rate_curve()
        self.further_calculations()
        self.integrate_images()
        self.compress_results()
        self.serialize_results()

# 使用示例
if __name__ == '__main__':
    file_path = './干燥原始数据记录表(非).xlsx'
    drying_data_processor = DryingDataProcessor(file_path)
    drying_data_processor.run_drying_data_processor()

# 获取结果，依旧可以从对象的变量区寻找
# ans1 = drying_data_processor.ans1
# ans2 = drying_data_processor.ans2
# results = drying_data_processor.results
# drying_data_processor.format_results()
