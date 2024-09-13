import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings
import zipfile
import os

warnings.filterwarnings("ignore")

class PackedTowerAnalyzer:
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.sheet_names = pd.ExcelFile(file_dir).sheet_names
        self.results = []
        
        # 添加新的属性
        self.V_空 = None
        self.t_空 = None
        self.p_空气压力 = None
        self.Δp_全塔_mmH2O = None
        self.D = 0.1
        self.Z = 0.75
        self.A = None
        self.V_空_修 = None
        self.u = None
        self.ρ_水 = 9.8
        self.g = 9.8
        self.Δp_over_Z = None
        self.corr = None
        self.popt = None
        self.fit_label = None
        self.ans1 = None
        self.ans2 = None

    @staticmethod
    def linear_fit(x, a, b):
        return a * x + b

    @staticmethod
    def taylor_fit(x, *coefficients):
        n = len(coefficients)
        y_fit = np.zeros_like(x)
        for i in range(n):
            y_fit += coefficients[i] * (x ** i)
        return y_fit

    def analyze_fluid_dynamics(self, sheet_name, threshold=0.95):
        df = pd.read_excel(self.file_dir, header=None, sheet_name=sheet_name)
        data = df.iloc[2:, 1:].apply(pd.to_numeric, errors='coerce').values
        
        self.V_空 = data[:, 0]
        self.t_空 = data[:, 1]
        self.p_空气压力 = data[:, 2]
        self.Δp_全塔_mmH2O = data[:, 4]
        
        self.A = np.pi * (self.D / 2)**2
        
        self.V_空_修 = self.V_空 * np.sqrt((1.013e5 / (self.p_空气压力 * 1e3 + 1.013e5)) * ((self.t_空 + 273.15) / (25 + 273.15)))
        self.u = self.V_空_修 / self.A / 3600
        
        self.Δp_over_Z = self.ρ_水 * self.g * self.Δp_全塔_mmH2O * 1e-3 / self.Z
        
        self.corr, _ = pearsonr(self.u, self.Δp_over_Z)
        
        if abs(self.corr) >= threshold:
            self.popt, _ = curve_fit(self.linear_fit, self.u, self.Δp_over_Z)
            self.fit_label = f'线性拟合: $\\Delta p/Z = {self.popt[0]:.0f}u + {self.popt[1]:.0f}$'
        else:
            degree = 4
            initial_guess = np.zeros(degree + 1)
            self.popt, _ = curve_fit(self.taylor_fit, self.u, self.Δp_over_Z, p0=initial_guess)
            self.fit_label = '泰勒级数拟合: $\\Delta p/Z = '
            for i, coeff in enumerate(self.popt):
                self.fit_label += f'{coeff:.0f}u^{i} + '
            self.fit_label = self.fit_label[:-3] + '$'
        
        return {
            'u': self.u,
            'delta_p_over_z': self.Δp_over_Z,
            'corr': self.corr,
            'popt': self.popt,
            'fit_label': self.fit_label
        }

    def plot_fluid_dynamics(self):
        for index, sheet_name in enumerate(self.sheet_names[0:2], start=1):
            result = self.analyze_fluid_dynamics(sheet_name)
            self.results.append(result)

        self.ans1, self.ans2 = self.results[0], self.results[1]

        plt.rcParams['font.family'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False

        self._plot_single_result(self.ans1, '干填料', 'red')
        self._plot_single_result(self.ans2, '湿填料', 'blue')
        self._plot_comparison(self.ans1, self.ans2)

    def _plot_single_result(self, result, title, color):
        plt.figure(figsize=(8, 6))
        plt.scatter(result['u'], result['delta_p_over_z'], color=color, label=title)
        
        x_fit = np.linspace(np.min(result['u']), np.max(result['u']), 1000)
        if len(result['popt']) == 2:
            y_fit = self.linear_fit(x_fit, *result['popt'])
        else:
            y_fit = self.taylor_fit(x_fit, *result['popt'])
        
        plt.plot(x_fit, y_fit, 'k-', label=result['fit_label'])
        
        plt.xlabel('空塔气速 u (m/s)')
        plt.ylabel('单位高度填料层压降 $\\Delta p/Z$ (kPa/m)')
        plt.title(f'{title} u - $\\Delta p/Z$')
        plt.legend()
        plt.grid(True)
        plt.minorticks_on()
        plt.xlim(left=0, right=1.3)
        plt.ylim(bottom=0, top=40)
        self._set_spine_width(plt.gca())
        plt.savefig(f'./拟合图结果/u - Δp_over_Z ({title}).png', dpi=300)
        plt.show()

    def _plot_comparison(self, ans1, ans2):
        plt.figure(figsize=(8, 6))
        
        plt.scatter(ans1['u'], ans1['delta_p_over_z'], color='red', label='干填料')
        x_fit1 = np.linspace(np.min(ans1['u']), np.max(ans1['u']), 1000)
        y_fit1 = self.linear_fit(x_fit1, *ans1['popt'])
        plt.plot(x_fit1, y_fit1, 'k-', label=ans1['fit_label'])
        
        plt.scatter(ans2['u'], ans2['delta_p_over_z'], color='blue', label='湿填料')
        x_fit2 = np.linspace(np.min(ans2['u']), np.max(ans2['u']), 1000)
        y_fit2 = self.taylor_fit(x_fit2, *ans2['popt'])
        plt.plot(x_fit2, y_fit2, 'k-', label=ans2['fit_label'])
        
        plt.xlabel('空塔气速 u (m/s)')
        plt.ylabel('单位高度填料层压降 $\\Delta p/Z$ (kPa/m)')
        plt.title('u - $\\Delta p/Z$ 干填料 vs. 湿填料')
        plt.legend()
        plt.grid(True)
        plt.minorticks_on()
        plt.xlim(left=0, right=1.3)
        plt.ylim(bottom=0, top=40)
        self._set_spine_width(plt.gca())
        plt.savefig(r'./拟合图结果/u - Δp_over_Z (干填料 vs. 湿填料).png', dpi=300)
        plt.show()

    @staticmethod
    def _set_spine_width(ax):
        for spine in ax.spines.values():
            spine.set_linewidth(2)

class OxygenDesorptionAnalyzer:
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.sheet_names = pd.ExcelFile(file_dir).sheet_names
        
        # 添加新的属性
        self.ρ_水 = 1e3
        self.ρ_空 = 1.29
        self.g = 9.8
        self.M_O2 = 32
        self.M_H2O = 18
        self.M_空 = 29
        self.D = 0.1
        self.Z = 0.75
        self.A = None
        self.p_0 = 1.013e5
        self.V_水 = None
        self.V_空 = None
        self.ΔP_U管压差 = None
        self.c_富氧水 = None
        self.c_贫氧水 = None
        self.t_水 = None
        self.L = None
        self.V = None
        self.x_1 = None
        self.x_2 = None
        self.G_A = None
        self.Ω = None
        self.V_p = None
        self.p = None
        self.m = None
        self.y1 = 21
        self.y2 = 21
        self.x_1_star = None
        self.x_2_star = None
        self.Δx_m = None
        self.K_x_times_a = None
        self.a_fit = None
        self.b_fit = None

    @staticmethod
    def E(t):
        return (-8.5694e-5 * t ** 2 + 0.07714 * t + 2.56) * 1e9

    @staticmethod
    def correlation_func(vars, a, b):
        A, L, V = vars
        return A * L**a * V**b

    def analyze_oxygen_desorption(self, sheet_name):
        initial_guess = [1.0, 1.0]

        df = pd.read_excel(self.file_dir, header=None, sheet_name=sheet_name)
        data = df.iloc[2:, 1:].apply(pd.to_numeric, errors='coerce').values

        self.A = np.full((3,), np.pi * (self.D / 2) ** 2)
        self.V_水 = data[:, 1]
        self.V_空 = data[:, 2]
        self.ΔP_U管压差 = self.ρ_水 * self.g * data[:, 3] * 1e-3
        self.c_富氧水 = data[:, 5]
        self.c_贫氧水 = data[:, 6]
        self.t_水 = data[:, 7]

        self.L = self.ρ_水/self.M_H2O * self.V_水
        self.V = self.ρ_空/self.M_空 * self.V_空
        self.x_1 = (self.c_富氧水 * 1 / self.M_O2) / ((self.c_富氧水 * 1 / self.M_O2) * 1e-3 + 1e6 / 18)
        self.x_2 = (self.c_贫氧水 * 1 / self.M_O2) / ((self.c_贫氧水 * 1 / self.M_O2) * 1e-3 + 1e6 / 18)

        self.G_A = self.L / (self.x_1 - self.x_2)
        self.Ω = self.A
        self.V_p = self.Z * self.Ω
        self.p = self.p_0 + 1 / 2 * self.ΔP_U管压差
        self.m = self.E(self.t_水) / self.p

        self.x_1_star = self.y1 / self.m
        self.x_2_star = self.y2 / self.m
        self.Δx_m = ((self.x_1 - self.x_1_star) - (self.x_2 - self.x_2_star))/np.log((self.x_1 - self.x_1_star)/(self.x_2 - self.x_2_star))
        self.K_x_times_a = self.G_A / (self.V_p * self.Δx_m)

        params, _ = curve_fit(self.correlation_func, (self.A, self.L, self.V), self.K_x_times_a, p0=initial_guess)

        self.a_fit, self.b_fit = params[0], params[1]

        print(f"工作表 {sheet_name} 中最优化拟合得到的参数：a={self.a_fit}, b={self.b_fit}")

        plt.rcParams['font.family'] = 'simhei'
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(8, 8))
        plt.plot(self.K_x_times_a, self.correlation_func((self.A, self.L, self.V), self.a_fit, self.b_fit), 'o', c='r', label='（$K_{x}a$, $AL^aV^b$）')
        plt.xlabel('$K_{x}a$')
        plt.ylabel('$AL^aV^b$')
        plt.title(f'拟合结果vs.实际数据比较 - {sheet_name}')
        plt.legend()
        plt.grid(True)
        self._set_spine_width(plt.gca())
        plt.savefig(f'./拟合图结果/{sheet_name}.png', dpi=300)
        plt.show()

    @staticmethod
    def _set_spine_width(ax):
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    def analyze_all_sheets(self):
        for sheet_name in self.sheet_names[2:4]:
            self.analyze_oxygen_desorption(sheet_name)

class ResultCompressor:
    @staticmethod
    def compress_results(dir_to_zip, dir_to_save):
        with zipfile.ZipFile(dir_to_save, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(dir_to_zip):
                for file in files:
                    file_dir = os.path.join(root, file)
                    arc_name = os.path.relpath(file_dir, dir_to_zip)
                    zipf.write(file_dir, arc_name)
        print(f'压缩完成，文件保存为: {dir_to_save}')

if __name__ == '__main__':
    file_dir = r'./解吸原始记录表(非).xlsx'
    
    # 实例化并调用 PackedTowerAnalyzer
    tower_analyzer = PackedTowerAnalyzer(file_dir)
    tower_analyzer.plot_fluid_dynamics()

    # 实例化并调用 OxygenDesorptionAnalyzer
    oxygen_analyzer = OxygenDesorptionAnalyzer(file_dir)
    oxygen_analyzer.analyze_all_sheets()

    # 压缩结果
    image_result_dir = './拟合图结果'
    zip_file_path = './拟合图结果.zip'
    ResultCompressor.compress_results(image_result_dir, zip_file_path)

