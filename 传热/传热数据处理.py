# This project is created by aFei-CQUT
# ------------------------------------------------------------------------------------------------------------------------------------
#   About aFei-CQUT
# - Interests&Hobbies: Programing,  ChatGPT,  Reading serious books,  Studying academical papers.
# - CurrentlyLearning: Mathmodeling，Python and Mathmatica (preparing for National College Mathematical Contest in Modeling).
# - Email:2039787966@qq.com
# - Pronouns: Chemical Engineering, Computer Science, Enterprising, Diligent, Hard-working, Sophomore,Chongqing Institute of Technology,
# - Looking forward to collaborating on experimental data processing of chemical engineering principle
# ------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# 传热数据预处理
def heat_transfer_data_preprocess(Δp_kb, t_in, t_out):
    # 计算过程
    t_w = 98.4
    d_o = 0.022
    d_i = d_o - 2 * 0.001
    l = 1.2
    n = 1
    C_0 = 0.65
    S_i = np.pi * l * n * d_i
    S_o = np.pi * l * n * d_o
    A_i = (np.pi * d_i ** 2) / 4
    A_0 = 2.27 * 10 ** -4
    t_avg = 0.5 * (t_in + t_out)
    Cp = 1005
    ρ = (t_avg ** 2) / 10 ** 5 - (4.5 * t_avg) / 10 ** 3 + 1.2916
    λ = -(2 * t_avg ** 2) / 10 ** 8 + (8 * t_avg) / 10 ** 5 + 0.0244
    μ = (-(2 * t_avg ** 2) / 10 ** 6 + (5 * t_avg) / 10 ** 3 + 1.7169) / 10 ** 5
    Pr = (Cp * μ) / λ
    PrZeroFour = Pr ** 0.4
    V_t = 3600 * A_0 * C_0 * np.sqrt((2 * 10 ** 3 * Δp_kb) / ρ)
    V_xiu = ((t_avg + 273.15) * V_t) / (t_in + 273.15)
    u_m = V_xiu / (3600 * A_i)
    W_c = (ρ * V_xiu) / 3600
    Q = Cp * W_c * (t_out - t_in)
    α_i = Q / (t_avg * S_i)
    Nu_i = (d_i * α_i) / λ
    Re_i = (ρ * d_i * u_m) / μ
    NuOverPrZeroFour = Nu_i / Pr ** 0.4
    Δt1 = t_w - t_in
    Δt2 = t_w - t_out
    Δt_m = (Δt2 - Δt1) / (np.log(Δt2) - np.log(Δt1))
    K_o = Q / (Δt_m * S_o)

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


# 定义拟合函数
def fit_func(x, a, b):
    return a + b * x


# 作图函数
def plot_fit(data_for_fit, filename, title):
    # 使用curve_fit进行拟合
    ans_params, _ = curve_fit(fit_func, np.log10(data_for_fit[:, 0]), np.log10(data_for_fit[:, 1]))

    # 绘图
    plt.figure(figsize=(8, 6), dpi=125)
    plt.scatter(data_for_fit[:, 0], data_for_fit[:, 1], color='r', label=r'$\mathrm{Data}$')
    plt.plot(data_for_fit[:, 0], 10 ** fit_func(np.log10(data_for_fit[:, 0]), *ans_params), color='k',
             label=r'$\mathrm{Fit}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\mathrm{Re}$', fontsize=14, fontweight='bold')
    plt.ylabel(r'$\mathrm{Nu/Pr^{0.4}}$', fontsize=14, fontweight='bold')
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
    
# 读取Excel文件
ExcelFile = pd.ExcelFile('传热原始数据记录表(非).xlsx')
sheet_names = ExcelFile.sheet_names
sheet1 = pd.read_excel('传热原始数据记录表(非).xlsx', sheet_name=sheet_names[0], header=None)
data1 = np.array(sheet1.iloc[1:7, 1:4].values, dtype=float)
sheet2 = pd.read_excel('传热原始数据记录表(非).xlsx', sheet_name=sheet_names[1], header=None)
data2 = np.array(sheet2.iloc[1:7, 1:4].values, dtype=float)

# 总数据集
datasets = [
    {'Δp_kb': data1[:, 0], 't_in': data1[:, 1], 't_out': data1[:, 2]},
    {'Δp_kb': data2[:, 0], 't_in': data2[:, 1], 't_out': data2[:, 2]}
]

# 循环处理每个数据集
for idx, dataset in enumerate(datasets):
    Δp_kb = dataset['Δp_kb']
    t_in = dataset['t_in']
    t_out = dataset['t_out']

    # 数据处理
    ans_original_data, ans_calculated_data, data_for_fit = heat_transfer_data_preprocess(Δp_kb, t_in, t_out)

    # 使用curve_fit进行拟合
    ans_params, _ = curve_fit(fit_func, np.log10(data_for_fit[:, 0]), np.log10(data_for_fit[:, 1]))

    # 保存处理后的结果到不同的变量中
    if idx == 0:
        ans_original_data1 = ans_original_data
        ans_calculated_data1 = ans_calculated_data
        data_for_fit1 = data_for_fit
        ans_params1 = ans_params
    elif idx == 1:
        ans_original_data2 = ans_original_data
        ans_calculated_data2 = ans_calculated_data
        data_for_fit2 = data_for_fit
        ans_params2 = ans_params

# 设置作图参数
plt.rcParams['font.family'] = 'simhei'
plt.rcParams['axes.unicode_minus'] = False

# 绘制第一个数据集的拟合图并保存为 1.png
plot_fit(data_for_fit1, './拟合图结果/1.png', '无强化套管')

# 绘制第二个数据集的拟合图并保存为 2.png
plot_fit(data_for_fit2, './拟合图结果/2.png', '有强化套管')

# 合并绘图并保存为 3.png
plt.figure(figsize=(8, 6), dpi=125)
plt.scatter(data_for_fit1[:, 0], data_for_fit1[:, 1], color='r', label='无强化套管')
plt.scatter(data_for_fit2[:, 0], data_for_fit2[:, 1], color='b', label='有强化套管')
plt.plot(data_for_fit1[:, 0], 10 ** fit_func(np.log10(data_for_fit1[:, 0]), *ans_params1), color='r', label='数据集 1')
plt.plot(data_for_fit2[:, 0], 10 ** fit_func(np.log10(data_for_fit2[:, 0]), *ans_params2), color='b', label='数据集 2')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\mathrm{Re}$', fontsize=14, fontweight='bold')
plt.ylabel(r'$\mathrm{Nu/Pr^{0.4}}$', fontsize=14, fontweight='bold')
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

'''
拟合图结果压缩
'''
import zipfile
import os

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

