import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import zipfile
import os

class FluidFlowDataProcessor:
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.sheet_names = pd.ExcelFile(file_dir).sheet_names  # 获取工作表名称

    def process(self):
        """
        进行流体流动分析，包括计算雷诺数和摩擦系数，并进行双对数拟合。

        Returns:
            np.ndarray: 包含流速、雷诺数和摩擦系数的数组。
        """
        # 已知参数
        d = 0.008  # 管径(m)
        l = 1.70  # 管长(m)
        ρ = 996.5  # 水的密度(kg/m^3)
        g = 9.81  # 重力加速度(m/s^2)
        μ = 8.55e-4  # 粘性系数(Pa·s)

        # 从第一个工作表中读取数据 
        df = pd.read_excel(self.file_dir, sheet_name=self.sheet_names[0], header=None)

        # 提取特定数据
        Q = df.iloc[2:, 1].values / 3600 / 1000  # 流量数据(m^3/s)
        ΔPf = pd.concat([1000 * df.iloc[2:11, 2], ρ * g * df.iloc[11:, 3] / 1000], ignore_index=True).values  # 压力降数据(Pa)

        # 计算流速(m/s)
        u = Q / (np.pi / 4 * d**2)

        # 计算雷诺数(1)
        Re = np.asarray((d * u * ρ) / μ, dtype=float)

        # 计算摩擦系数(1)
        λ = np.asarray((2 * d) / (ρ * l) * ΔPf / u**2, dtype=float)

        # 泰勒拟合阶数
        degree = 9

        # lg(Re)-lg(λ)双对数拟合
        coefficients = np.polyfit(np.log10(Re), np.log10(λ), degree)
        p = np.poly1d(coefficients)

        # 生成更多的点用于插值
        log_Re_interp = np.linspace(np.min(np.log10(Re)), np.max(np.log10(Re)), 1000)
        log_lambda_interp = np.polyval(coefficients, log_Re_interp)

        # 设置字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 绘制散点图和拟合曲线(双对数坐标轴) - 无插值
        plt.figure(figsize=(8, 6), dpi=125)
        plt.scatter(np.log10(Re), np.log10(λ), color='b', label='数据点')
        plt.plot(np.log10(Re), p(np.log10(Re)), color='red', label='拟合曲线')
        plt.xlabel('lg(Re)')
        plt.ylabel('lg(λ)')
        plt.title('雷诺数与阻力系数双对数拟合(无插值)')
        plt.grid(True)
        plt.legend(loc='best')
        plt.savefig('./拟合图结果/雷诺数与阻力系数双对数拟合(无插值).png', dpi=300)

        # 绘制散点图和拟合曲线(双对数坐标轴) - 有插值
        plt.figure(figsize=(8, 6), dpi=125)
        plt.scatter(np.log10(Re), np.log10(λ), color='b', label='数据点')
        plt.plot(log_Re_interp, log_lambda_interp, color='r', label='插值曲线')
        plt.xlabel('lg(Re)')
        plt.ylabel('lg(λ)')
        plt.title('雷诺数与阻力系数双对数拟合(有插值)')
        plt.grid(True)
        plt.legend(loc='best')
        plt.savefig('./拟合图结果/雷诺数与阻力系数双对数拟合(有插值).png', dpi=300)

        plt.show()

        return np.column_stack((u, Re, λ)), df


class CentrifugalPumpCharacteristicsProcessor:
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.sheet_names = pd.ExcelFile(file_dir).sheet_names  # 获取工作表名称

    @staticmethod
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    def process(self):
        """
        分析离心泵的特性曲线，包括扬程、功率和效率的计算与二次拟合。

        Returns:
            np.ndarray: 包含扬程、电机功率和效率的数组。
        """
        # 从第二个工作表中读取数据
        df = pd.read_excel(self.file_dir, sheet_name=self.sheet_names[1], header=None)

        # 提取特定数据
        Q = df.iloc[1:, 1].values  # 流量数据()
        p_in = df.iloc[1:, 2].values  # 入口压力数据()
        p_out = df.iloc[1:, 3].values  # 出口压力数据()
        N_elc = df.iloc[1:, 4].values  # 电机功率数据()

        # 已知参数
        ρ = 995.7  # 水的密度(kg/m^3)
        g = 9.81  # 重力加速度(m/s^2)
        u_out = 0  # 出口流速(m/s)
        u_in = 0  # 入口流速(m/s)
        Δz = 0.23  # 高度差(m)
        η_elc = 0.6  # 电机效率(100%)

        # 计算扬程 H
        H = Δz + ((p_out - p_in) * 1e6) / (ρ * g) + (u_out**2 - u_in**2) / 2

        # 计算电机有效功率 N_elc_e
        N_elc_e = N_elc * η_elc * 1000  # 单位：瓦特

        # 计算输送流体的有效功率 N_e
        N_e = H * Q / (3600 * 1000) * ρ * g * 1000  # 单位：瓦特

        # 泵效率 η
        η = N_e / N_elc_e

        # 对泵的扬程进行二次拟合
        params_H, _ = curve_fit(self.quadratic, Q, H)
        H_fit = self.quadratic(Q, *params_H)

        # 对泵的有效功率进行二次拟合
        params_N_elc_e, _ = curve_fit(self.quadratic, Q, N_elc_e)
        N_fit = self.quadratic(Q, *params_N_elc_e)

        # 对泵的效率进行二次拟合
        params_η, _ = curve_fit(self.quadratic, Q, η)
        η_fit = self.quadratic(Q, *params_η)

        # 创建图形
        fig, ax1 = plt.subplots(figsize=(7.85, 6), dpi=125)

        # 绘制扬程-流量散点图和拟合曲线
        ax1.scatter(Q, H, color='blue', label='扬程 $H$ 数据')
        ax1.plot(Q, H_fit, 'b-', label='扬程 $H$ 拟合')
        ax1.set_xlabel('$Q/(m^3/h)$')
        ax1.set_ylabel('$H/m$', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # 创建第二个y轴用于绘制功率和效率
        ax2 = ax1.twinx()
        ax2.scatter(Q, N_elc_e, color='red', label='功率 $N$ 数据')
        ax2.plot(Q, N_fit, 'r--', label='功率 $N$ 拟合')
        ax2.set_ylabel('$N/kW$', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # 创建第三个y轴用于绘制效率
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # 偏移第三个y轴
        ax3.scatter(Q, η, color='green', label='效率 $\eta$ 数据')
        ax3.plot(Q, η_fit, 'g-.', label='效率 $\eta$ 拟合')
        ax3.set_ylabel('$\eta/\%$', color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        # 添加图例
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3)

        # 设置字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 显示图形
        plt.title('离心泵特性曲线及二次拟合')
        plt.tight_layout(rect=[0.05, 0.03, 0.95, 0.93])  # 调整布局
        plt.savefig(r'./拟合图结果/离心泵特性曲线及二次拟合.png', dpi=300)
        plt.show()

        return np.column_stack((H, N_elc_e, η)), df


def compress_results():
    """
    压缩拟合图结果文件夹。
    """
    # 待压缩的文件路径
    dir_to_zip = r'./拟合图结果'

    # 压缩后的保存路径
    dir_to_save = r'./拟合图结果.zip'

    # 创建ZipFile对象
    with zipfile.ZipFile(dir_to_save, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历目录
        for root, dirs, files in os.walk(dir_to_zip):
            for file in files:
                # 创建相对文件路径并将其写入zip文件
                file_dir = os.path.join(root, file)
                arc_name = os.path.relpath(file_dir, dir_to_zip)
                zipf.write(file_dir, arc_name)

    print(f'压缩完成，文件保存为: {dir_to_save}')


if __name__ == "__main__":
    file_dir = r'./流体原始数据记录表(非).xlsx'
    
    # 确保输出目录存在
    os.makedirs('./拟合图结果', exist_ok=True)
    
    # 流体流动分析
    fluid_flow_data_processor = FluidFlowDataProcessor(file_dir)
    ans1, df1 = fluid_flow_data_processor.process()
    
    # 离心泵特性分析
    centrifugal_pump_characteristics_processor = CentrifugalPumpCharacteristicsProcessor(file_dir)
    ans2, df2 = centrifugal_pump_characteristics_processor.process()
    
    # 压缩拟合图结果
    compress_results()
