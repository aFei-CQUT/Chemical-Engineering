import pandas as pd
import numpy as np
import sympy
import matplotlib.pyplot as plt
import zipfile
import os

class DistillationDataProcessor:
    """
    精馏数据处理器类

    用于处理和分析精馏实验数据,计算理论板数,绘制McCabe-Thiele图等。

    属性:
        df (DataFrame): 存储原始实验数据的DataFrame
        R (float): 回流比
        αm (float): 相对挥发度
        F (float): 进料量
        tS (float): 泡点温度
        tF (float): 进料温度
    """

    def __init__(self, file_path, R, αm, F, tS, tF):
        """
        初始化DistillationDataProcessor对象

        参数:
            file_path (str): Excel文件路径
            R (float): 回流比
            αm (float): 相对挥发度
            F (float): 进料量
            tS (float): 泡点温度
            tF (float): 进料温度
        """
        self.df = pd.read_excel(file_path, header=0)
        self.R = R
        self.αm = αm
        self.F = F
        self.tS = tS
        self.tF = tF
        self.set_constants()
        self.calculate_feed_parameters()
        self.calculate_compositions()
        self.solve_material_balance()
        self.calculate_stages()

    def set_constants(self):
        """设置常量"""
        self.ρA, self.ρB = 0.789, 1.0  # 乙醇和水的密度
        self.cA, self.cB = 2.4*1e+3, 4.189*1e+3  # 乙醇和水的比热容
        self.rA, self.rB = 850, 2260  # 摩尔汽化热
        self.MA, self.MB = 46, 18  # 乙醇和水的摩尔质量
        self.xA, self.xB = 0.1, 0.9  # 进料时乙醇和水的摩尔分数

    def calculate_feed_parameters(self):
        """计算进料参数"""
        self.cpm = self.xA * self.cA * self.MA + self.xB * self.cB * self.MB  # 平均比热容
        self.rm = self.xA * self.rA * self.MA + self.xB * self.rB * self.MB  # 平均摩尔汽化热
        self.q = (self.cpm * (self.tS - self.tF) + self.rm) / self.rm if self.tS else 1.5  # 进料热状态参数

    def calculate_x_ethanol(self, s):
        """
        计算乙醇的摩尔分数

        参数:
            s (float): 乙醇在混合液中的体积分数

        返回:
            float: 乙醇的摩尔分数
        """
        return (s * self.ρA / self.MA) / ((s * self.ρA / self.MA) + ((100 - s) * self.ρB / self.MB))

    def calculate_compositions(self):
        """计算各种组成"""
        self.xD_inf = self.calculate_x_ethanol(self.df.loc[0, '20°C酒精度(查表)/°'])  # 回流比 R = ∞ 时，馏出液中乙醇的摩尔分数
        self.xW_inf = self.calculate_x_ethanol(self.df.loc[1, '20°C酒精度(查表)/°'])  # 回流比 R = ∞ 时，釜残液中乙醇的摩尔分数
        self.xD = self.calculate_x_ethanol(self.df.loc[2, '20°C酒精度(查表)/°'])  # 回流比 R = 4 时，馏出液中乙醇的摩尔分数
        self.xW = self.calculate_x_ethanol(self.df.loc[3, '20°C酒精度(查表)/°'])  # 回流比 R = 4 时，釜残液中乙醇的摩尔分数
        self.xF = self.calculate_x_ethanol(self.df.loc[4, '20°C酒精度(查表)/°'])  # 进料液中乙醇的摩尔分数

    def solve_material_balance(self):
        """求解物料平衡方程"""
        if self.R >= 10000:
            xD = self.xD_inf
            xW = self.xW_inf
        else:
            xD = self.xD
            xW = self.xW
        
        A = sympy.Matrix([[1, 1], [xD, xW]])
        b = sympy.Matrix([self.F, self.xF * self.F])
        self.D, self.W = A.solve(b)  # D为馏出液量,W为釜残液量
        self.L = self.R * self.D  # 回流量

    def y_e(self, x):
        """
        平衡线方程

        参数:
            x (float): 液相组成

        返回:
            float: 气相组成
        """
        return self.αm * x / (1 + (self.αm - 1) * x)

    def x_e(self, y):
        """
        反平衡线方程

        参数:
            y (float): 气相组成

        返回:
            float: 液相组成
        """
        return y / (self.αm - (self.αm - 1) * y)

    def y_np1(self, x):
        """
        精馏段操作线方程

        参数:
            x (float): 液相组成

        返回:
            float: 气相组成
        """
        return self.R / (self.R + 1) * x + self.xD / (self.R + 1)

    def y_mp1(self, x):
        """
        提馏段操作线方程

        参数:
            x (float): 液相组成

        返回:
            float: 气相组成
        """
        return (self.L + self.q * self.F) / (self.L + self.q * self.F - self.W) * x - self.W / (self.L + self.q * self.F - self.W) * self.xW

    def y_q(self, x):
        """
        q线方程

        参数:
            x (float): 液相组成

        返回:
            float: 气相组成
        """
        if self.q == 1:
            return x
        else:
            return self.q / (self.q - 1) * x - 1 / (self.q - 1) * self.xF

    def calculate_stages(self):
        """计算理论板数"""
        if self.R >= 10000:
            xD = self.xD_inf
            xW = self.xW_inf
        else:
            xD = self.xD
            xW = self.xW

        self.xQ = ((self.R + 1) * self.xF + (self.q - 1) * xD) / (self.R + self.q)
        self.yQ = (self.xF * self.R + self.q * xD) / (self.R + self.q)

        yn = np.array([xD])
        xn = np.array([])
        max_iterations = 20
        iteration = 0

        while self.x_e(yn[-1]) > xW and iteration < max_iterations:
            iteration += 1
            xn = np.append(xn, self.x_e(yn[-1]))
            
            if xn[-1] > self.xQ:
                yn = np.append(yn, self.y_np1(xn[-1]))
            else:
                yn = np.append(yn, self.y_mp1(xn[-1]))

        xn = np.append(xn, self.x_e(yn[-1]))
        self.NT = len(xn)
        self.xn, self.yn = xn, yn

    def plot_results(self, filename):
        """绘制McCabe-Thiele图"""
        x_array = np.linspace(0, 1, 50)
        y_array = x_array
        y_e_array = self.y_e(x_array)
        y_np1_array = self.y_np1(x_array)
        y_mp1_array = self.y_mp1(x_array)

        xNT = np.array([self.xD])
        yNT = np.array([self.xD])
        for n, i in enumerate(self.xn):
            xNT = np.append(xNT, i)
            yNT = np.append(yNT, self.yn[n])
            xNT = np.append(xNT, i)
            if i >= self.xQ:
                yNT = np.append(yNT, self.y_np1(i))
            else:
                yNT = np.append(yNT, self.y_mp1(i))

        plt.figure(figsize=(8,6), dpi=125)
        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        plt.plot(x_array, y_array, ls="-", label="对角线")
        plt.plot(x_array, y_e_array, label="平衡线")
        plt.plot(x_array, y_np1_array, label="精馏操作线")
        plt.plot(x_array, y_mp1_array, label="提馏操作线")
        plt.plot(self.xn, self.yn, label="塔板操作平衡点", ls=":", marker="+", markersize=10)
        plt.plot(xNT, yNT, label="图解法—理论塔板", ls=":")

        plt.plot(self.xD, self.xD, marker=".", markersize=10)
        plt.plot(self.xW, self.xW, marker=".", markersize=10)
        plt.plot(self.xQ, self.yQ, marker=".", markersize=10)

        plt.annotate("$W$ 点", xy=(self.xW, self.xW), xytext=(self.xW + 0.05, self.xW), arrowprops=dict(arrowstyle="->"))
        plt.annotate("$D$ 点", xy=(self.xD, self.xD), xytext=(self.xD, self.xD - 0.05), arrowprops=dict(arrowstyle="->"))
        plt.annotate("$Q$ 点", xy=(self.xQ, self.yQ), xytext=(self.xQ, self.yQ - 0.05), arrowprops=dict(arrowstyle="->"))

        plt.legend()
        plt.gca().spines["top"].set_linewidth(2)
        plt.gca().spines["bottom"].set_linewidth(2)
        plt.gca().spines["left"].set_linewidth(2)
        plt.gca().spines["right"].set_linewidth(2)
        plt.gca().grid()

        plt.text(x=0.6, y=0.4, s=f"所需理论板数：{self.NT - 1}")

        plt.title("图解理论板数")
        plt.ylabel("$y$")
        plt.xlabel("$x$")
        
        plt.show()
        plt.savefig(f'./拟合图结果/{filename}.png', dpi=300)
        plt.close()  # 关闭图形,避免显示

    def save_results(self, filename):
        """输出结果并保存到文本文件"""
        results = []
        results.append("--- Q点数据 ---")
        results.append(f"xQ: {self.xQ:.4f}")
        results.append(f"yQ: {self.yQ:.4f}")
    
        results.append("\n--- xn和yn数据 ---")
        for i, (x, y) in enumerate(zip(self.xn, self.yn)):
            results.append(f"Stage {i}: xn = {x:.4f}, yn = {y:.4f}")
    
        results.append(f"\n理论板层数为: {self.NT - 1}")
    
        with open(f'./拟合图结果/{filename}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(results))

def process_and_save(file_path, R, αm, F, tS, tF, filename):
    distillation_data_processor = DistillationDataProcessor(file_path, R, αm, F, tS, tF)
    distillation_data_processor.plot_results(filename)
    distillation_data_processor.save_results(filename)

# 使用示例
if __name__ == "__main__":
    file_path = r'./精馏原始记录表(非).xlsx'
    
    # 确保输出目录存在
    os.makedirs('./拟合图结果', exist_ok=True)

    # R = 4 时
    process_and_save(file_path, R=4, αm=2.0, F=80, tS=30, tF=26, filename="R_4_结果")
    
    # R --> ∞时,设置 R = 10000
    process_and_save(file_path, R=10000, αm=2.0, F=80, tS=30, tF=26, filename="R_无穷大_结果")

    # 压缩文件夹
    dir_to_zip = r'./拟合图结果'
    dir_to_save = r'./拟合图结果.zip'

    with zipfile.ZipFile(dir_to_save, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dir_to_zip):
            for file in files:
                file_dir = os.path.join(root, file)
                arc_name = os.path.relpath(file_dir, dir_to_zip)
                zipf.write(file_dir, arc_name)

    print(f'\n压缩完成,文件保存为: {dir_to_save}')
