import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
import zipfile
import os

class ExtractionDataProcesser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.results = {}
        self.setup_plot()
        
    def setup_plot(self):
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 125

    def load_data(self):
        excel_file = pd.ExcelFile(self.file_path)
        self.sheet_names = excel_file.sheet_names
        
        df1 = pd.read_excel(self.file_path, sheet_name=self.sheet_names[0], header=None)
        self.data1 = df1.iloc[1:, 1:].values

        self.n = self.data1[0]
        self.Vs_S = self.data1[1]  # L/h
        self.Vs_B = self.data1[2]  # L/h
        self.t = self.data1[3]
        self.c_NaOH = 0.01
        self.V_to_be_titrated = self.data1[[4, 6, 8], :]
        self.V_NaOH_used = self.data1[[5, 7, 9], :]

        # 分子量和密度
        self.M_A, self.M_B, self.M_S = 78, 122, 18  # kg/kmol
        self.ρ_A, self.ρ_B, self.ρ_S = 876.7, 800, 1000  # kg/m^3

    def preprocess_data(self):
        self.Vs_B_rect = self.Vs_S * np.sqrt(self.ρ_S * (7900 - self.ρ_B) / (self.ρ_B * (7900 - self.ρ_S)))  # L/h
        
        self.ans1 = (self.c_NaOH * self.V_NaOH_used * 1e-6 * self.M_B) / (self.ρ_B * self.V_to_be_titrated * 1e-6)
        self.X_Rb, self.X_Rt, self.Y_Eb = self.ans1[0], self.ans1[1], self.ans1[2]

        self.B = self.ρ_B * self.Vs_B * 1e-3  # L转换为m^3
        self.S = self.ρ_S * self.Vs_S * 1e-3  # L转换为m^3
        self.B_rect = self.ρ_B * self.Vs_B_rect * 1e-3  # L转换为m^3

        self.ans2 = np.array([self.B, self.S, self.B_rect])

        self.results.update({
            'ans1': self.ans1.tolist(),
            'ans2': self.ans2.tolist(),
            'X_Rb': self.X_Rb.tolist(),
            'X_Rt': self.X_Rt.tolist(),
            'Y_Eb': self.Y_Eb.tolist(),
        })

    def load_distribution_curve_data(self):
        df3 = pd.read_excel(self.file_path, sheet_name=self.sheet_names[2], header=None)
        data3 = df3.iloc[2:, :].values
        self.X3_data = data3[:, 0].astype(float)
        self.Y3_data = data3[:, 1].astype(float)

    def fit_distribution_curve(self):
        order = 3
        self.coefficients = np.polyfit(self.X3_data, self.Y3_data, order)
        self.X3_to_fit = np.linspace(min(self.X3_data), max(self.X3_data), 100)
        self.Y_fitted = np.polyval(self.coefficients, self.X3_to_fit)

        self.results.update({
            'X3_data': self.X3_data.tolist(),
            'Y3_data': self.Y3_data.tolist(),
            'coefficients': self.coefficients.tolist(),
            'X3_to_fit': self.X3_to_fit.tolist(),
            'Y_fitted': self.Y_fitted.tolist()
        })

    def calculate_operating_lines(self):
        self.k1 = (0 - self.Y_Eb[0]) / (self.X_Rt[0] - self.X_Rb[0])
        self.b1 = self.Y_Eb[0] - self.k1 * self.X_Rb[0]

        self.k2 = (0 - self.Y_Eb[1]) / (self.X_Rt[1] - self.X_Rb[1])
        self.b2 = self.Y_Eb[1] - self.k2 * self.X_Rb[1]

        self.results.update({
            'k1': float(self.k1),
            'b1': float(self.b1),
            'k2': float(self.k2),
            'b2': float(self.b2)
        })

    def plot_distribution_and_operating_lines(self):
        plt.figure(figsize=(8, 6))

        plt.scatter(self.X3_data, self.Y3_data, color='purple', marker='^', label='分配曲线数据点')
        plt.plot(self.X3_to_fit, self.Y_fitted, color='k', label='分配曲线')

        X_operating1 = np.linspace(0, self.X_Rb[0], 500)
        Y_operating1 = self.k1 * X_operating1 + self.b1
        plt.scatter([self.X_Rb[0], self.X_Rt[0]], [self.Y_Eb[0], 0], color='green', marker='o', label='操作线1定点')
        plt.plot(X_operating1, Y_operating1, linestyle='--', color='green', label='操作线1方程')

        X_operating2 = np.linspace(0, self.X_Rb[1], 500)
        Y_operating2 = self.k2 * X_operating2 + self.b2
        plt.scatter([self.X_Rb[1], self.X_Rt[1]], [self.Y_Eb[1], 0], color='orange', marker='o', label='操作线2定点')
        plt.plot(X_operating2, Y_operating2, linestyle='--', color='orange', label='操作线2方程')

        plt.title('分配曲线与操作线')
        plt.xlabel('X 数据')
        plt.ylabel('Y 数据')
        plt.legend()
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.grid(True)
        
        # 图表设置
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.grid(True, which='both')
        plt.minorticks_on()
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['top'].set_linewidth(2)
        
        # 确保保存路径存在
        if not os.path.exists('./拟合图结果'):
            os.makedirs('./拟合图结果')
        
        plt.show()
        
        plt.savefig('./拟合图结果/1', dpi=300)
        
        plt.close()

    def perform_graphical_integration(self):
        self.data5_for_graph_integral = []
        k = np.array([self.k1, self.k2])
        b = np.array([self.b1, self.b2])

        for i in range(len(self.Y_Eb)):
            Y5_Eb_data = np.linspace(0, self.Y_Eb[i], 20)
            X_Rb_data = (Y5_Eb_data - b[i]) / k[i]
            Y5star_data = np.polyval(self.coefficients, X_Rb_data)
            one_over_Y5star_minus_Y5 = 1 / (Y5star_data - Y5_Eb_data)

            self.data5_for_graph_integral.extend([Y5_Eb_data, X_Rb_data, Y5star_data, one_over_Y5star_minus_Y5])

            interp_func = interp1d(Y5_Eb_data, one_over_Y5star_minus_Y5, kind='cubic')
            Y5_Eb_data_smooth = np.linspace(Y5_Eb_data.min(), Y5_Eb_data.max(), 40)
            one_over_Y5star_minus_Y5_smooth = interp_func(Y5_Eb_data_smooth)

            plt.figure(figsize=(8, 6))
            plt.scatter(Y5_Eb_data_smooth, one_over_Y5star_minus_Y5_smooth, color='r', label=f'$\\frac{{1}}{{Y_{{5}}^*-Y_{{5}}}}$ 数据组 {i+1}')
            plt.plot(Y5_Eb_data_smooth, one_over_Y5star_minus_Y5_smooth, linestyle='-', color='k')
            plt.fill_between(Y5_Eb_data_smooth, one_over_Y5star_minus_Y5_smooth, alpha=0.5, color='gray')

            plt.xlabel('$Y_5$')
            plt.ylabel('$\\frac{1}{Y^{*}_5-Y_5}$')
            plt.legend()
            plt.title(f'$\\frac{{1}}{{Y^*_5-Y_5}} - Y_5$       数据组 {i+1}')

            integral_value = trapezoid(one_over_Y5star_minus_Y5_smooth, Y5_Eb_data_smooth)
            plt.text(0.5, -0.15, f'数值积分结果: {integral_value:.5f}', transform=plt.gca().transAxes, horizontalalignment='center')

            # 图表设置
            plt.xlim(left=Y5_Eb_data_smooth.min())
            plt.ylim(bottom=0)
            plt.grid(True, which='both')
            plt.minorticks_on()
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['right'].set_linewidth(2)
            plt.gca().spines['top'].set_linewidth(2)

            plt.savefig(f'./拟合图结果/{i+2}', dpi=300)
            plt.show()
            plt.close()

        self.ans3 = np.array(self.data5_for_graph_integral)
        self.results['ans3'] = self.ans3.tolist()

    def compress_results(self):
        dir_to_zip = r'./拟合图结果'
        dir_to_save = r'./拟合图结果.zip'

        with zipfile.ZipFile(dir_to_save, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(dir_to_zip):
                for file in files:
                    file_dir = os.path.join(root, file)
                    arc_name = os.path.relpath(file_dir, dir_to_zip)
                    zipf.write(file_dir, arc_name)

        print(f'压缩完成，文件保存为: {dir_to_save}')

    def run_extraction_data_processer(self):
        self.load_data()
        self.preprocess_data()
        self.load_distribution_curve_data()
        self.fit_distribution_curve()
        self.calculate_operating_lines()
        self.plot_distribution_and_operating_lines()
        self.perform_graphical_integration()
        self.compress_results()

# 使用示例
if __name__ == "__main__":
    file_path = './萃取原始数据记录表(非).xlsx'
    extraction_data_processer = ExtractionDataProcesser(file_path)
    extraction_data_processer.run_extraction_data_processer()
    
    # 所需结果
    # results = extraction_data_processer.results
    # 可在变量分析窗口的extraction_data_processer里面查看
    # 它的属性ans1，ans2，ans3和results即为所需
    # ans1 = extraction_data_processer.ans1
    # ans2 = extraction_data_processer.ans2
    # ans3 = extraction_data_processer.ans3
