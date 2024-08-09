# This project is created by aFei-CQUT
# ------------------------------------------------------------------------------------------------------------------------------------
#   About aFei-CQUT
# - Interests&Hobbies: Programing,  ChatGPT,  Reading serious books,  Studying academical papers.
# - CurrentlyLearning: Mathmodeling，Python and Mathmatica (preparing for National College Mathematical Contest in Modeling).
# - Email:2039787966@qq.com
# - Pronouns: Chemical Engineering, Computer Science, Enterprising, Diligent, Hard-working, Sophomore,Chongqing Institute of Technology,
# - Looking forward to collaborating on experimental data processing of chemical engineering principle
# ------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

'''
填料塔流体力学心性能数据处理及图像拟合
'''
def linear_fit(x, a, b):
    return a * x + b

def taylor_fit(x, *coefficients):
    n = len(coefficients)
    y_fit = np.zeros_like(x)
    for i in range(n):
        y_fit += coefficients[i] * (x ** i)
    return y_fit

def Packed_Tower_Fluid_Dynamics_Performance(file_dir, sheet_name, threshold=0.95):
    # 读取Excel文件中的指定工作表数据
    df = pd.read_excel(file_dir, header=None, sheet_name=sheet_name)
    data = df.iloc[2:, 1:].apply(pd.to_numeric, errors='coerce').values
    
    # 提取各个变量
    V_空 = data[:, 0]  # 空塔气体流量,单位: 立方米每小时 (m³/h)
    t_空 = data[:, 1]  # 空塔气体温度,单位: 摄氏度 (°C)
    p_空气压力 = data[:, 2]  # 空塔气体压力,单位: 千帕 (kPa)
    Δp_全塔_mmH2O = data[:, 4]  # 全塔压降,单位: 毫米水柱 (mmH2O)
    
    # 解析塔的几何参数
    D = 0.1  # 解吸塔内径,单位: 米 (m)
    Z = 0.75  # 解吸塔填料层高度,单位: 米 (m)
    A = np.pi * (D / 2)**2  # 计算解吸塔截面积
    
    # 修正参数
    V_空_修 = V_空 * np.sqrt((1.013e5 / (p_空气压力 * 1e3 + 1.013e5)) * ((t_空 + 273.15) / (25 + 273.15)))
    
    # 计算空塔气速 u (单位: m/s)
    u = V_空_修 / A / 3600
    
    # 水的密度,单位: 千克每米三次方 (kg/m³)
    ρ_水 = 9.8
    
    # 重力加速度
    g = 9.8
    
    # 计算单位高度填料层压降 Δp/Z (单位: kPa/m)
    Δp_over_Z = ρ_水 * g * Δp_全塔_mmH2O * 1e-3 / Z
    
    # 计算皮尔逊相关系数
    corr, _ = pearsonr(u, Δp_over_Z)
    
    # 根据总的相关系数选择拟合方法并拟合数据
    if abs(corr) >= threshold:
        # 使用线性拟合
        popt, _ = curve_fit(linear_fit, u, Δp_over_Z)
        fit_label = f'线性拟合: $\\Delta p/Z = {popt[0]:.0f}u + {popt[1]:.0f}$'
    else:
        # 使用泰勒级数拟合
        degree = 4
        initial_guess = np.zeros(degree + 1)  # 初始化拟合系数
        popt, _ = curve_fit(taylor_fit, u, Δp_over_Z, p0=initial_guess)
        fit_label = '泰勒级数拟合: $\\Delta p/Z = '
        for i, coeff in enumerate(popt):
            fit_label += f'{coeff:.0f}u^{i} + '
        fit_label = fit_label[:-3] + '$'
    
    # 返回结果
    return {
        'u': u,
        'delta_p_over_z': Δp_over_Z,
        'corr': corr,
        'popt': popt,
        'fit_label': fit_label
    }

# 文件路径
file_dir = r'./解吸原始记录表(非).xlsx'

# 工作表列表
sheet_names = pd.ExcelFile(file_dir).sheet_names

# 存储结果的容器
results = []

# 遍历每个工作表并调用函数
for index, sheet_name in enumerate(sheet_names[0:2], start=1):
    result = Packed_Tower_Fluid_Dynamics_Performance(file_dir, sheet_name)
    results.append(result)

# 将结果分别存储在不同的变量中
ans1 = results[0]
ans2 = results[1]

# 设置绘图区中文、负号正常显示
plt.rcParams['font.family'] = 'simhei'
plt.rcParams['axes.unicode_minus'] = False

# 绘制第一幅图：干填料数据和拟合曲线
plt.figure(figsize=(10, 8))
plt.scatter(ans1['u'], ans1['delta_p_over_z'], color='red', label='干填料')
plt.plot(np.linspace(np.min(ans1['u']), np.max(ans1['u']), 1000),
         linear_fit(np.linspace(np.min(ans1['u']), np.max(ans1['u']), 1000), *ans1['popt']), 'k-', label=ans1['fit_label'])
plt.xlabel('空塔气速 u (m/s)')
plt.ylabel(r'单位高度填料层压降 $\Delta p/Z$ (kPa/m)')
plt.title('干填料 u - $\Delta p/Z$')
plt.legend()
plt.grid(True)
plt.minorticks_on()
plt.xlim(left=0, right=1.3)
plt.ylim(bottom=0, top=40)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.savefig(r'./拟合图结果/u - Δp_over_Z (干填料).png', dpi=300)
plt.show()

# 绘制第二幅图：湿填料数据和拟合曲线
plt.figure(figsize=(10, 8))
plt.scatter(ans2['u'], ans2['delta_p_over_z'], color='blue', label='湿填料')
plt.plot(np.linspace(np.min(ans2['u']), np.max(ans2['u']), 1000),
         taylor_fit(np.linspace(np.min(ans2['u']), np.max(ans2['u']), 1000), *ans2['popt']), 'k-', label=ans2['fit_label'])
plt.xlabel('空塔气速 u (m/s)')
plt.ylabel(r'单位高度填料层压降 $\Delta p/Z$ (kPa/m)')
plt.title('湿填料 u - $\Delta p/Z$')
plt.legend()
plt.grid(True)
plt.minorticks_on()
plt.xlim(left=0, right=1.3)
plt.ylim(bottom=0, top=40)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.savefig(r'./拟合图结果/u - Δp_over_Z (湿填料).png', dpi=300)
plt.show()

# 绘制整体图形
plt.figure(figsize=(10, 8))

# 干填料的数据和拟合曲线
plt.scatter(ans1['u'], ans1['delta_p_over_z'], color='red', label='干填料')
plt.plot(np.linspace(np.min(ans1['u']), np.max(ans1['u']), 1000),
         linear_fit(np.linspace(np.min(ans1['u']), np.max(ans1['u']), 1000), *ans1['popt']), 'k-', label=ans1['fit_label'])

# 湿填料的数据和拟合曲线
plt.scatter(ans2['u'], ans2['delta_p_over_z'], color='blue', label='湿填料')
plt.plot(np.linspace(np.min(ans2['u']), np.max(ans2['u']), 1000), 
         taylor_fit(np.linspace(np.min(ans2['u']), np.max(ans2['u']), 1000), *ans2['popt']), 'k-', label=ans2['fit_label'])

plt.xlabel('空塔气速 u (m/s)')
plt.ylabel(r'单位高度填料层压降 $\Delta p/Z$ (kPa/m)')
plt.title('u - $\Delta p/Z$ 干填料 vs. 湿填料')
plt.legend()
plt.grid(True)
plt.minorticks_on()
plt.xlim(left=0, right=1.3)
plt.ylim(bottom=0, top=40)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.savefig(r'./拟合图结果/u - Δp_over_Z (干填料 vs. 湿填料).png', dpi=300)
plt.show()

'''
氧解吸传质实验数据处理及图像拟合
'''

# 定义函数
def E(t):
    return (-8.5694e-5 * t ** 2 + 0.07714 * t + 2.56) * 1e9  # 亨利系数估计函数E(t),单位: 帕斯卡 (Pa)

def correlation_func(vars, a, b):
    A, L, V = vars
    return A * L**a * V**b

def Oxygen_Desorption_Mass_Transfer(file_dir, sheet_name):
    # 定义常量
    ρ_水 = 1e3  # 水的密度, 单位: 千克每米三次方 (kg/m³)
    ρ_空 = 1.29  # 空气的密度, 单位: 千克每米三次方 (kg/m³)
    g = 9.8  # 重力加速度, 单位: 米每平方秒 (m/s²)
    M_O2 = 32  # 氧气的摩尔质量, 单位: 千克每摩尔 (kg/mol)
    M_H2O = 18  # 水的摩尔质量, 单位: 千克每摩尔 (kg/mol)
    M_空 = 29  # 空气的摩尔质量, 单位: 千克每摩尔 (kg/mol)
    D = 0.1  # 解吸塔内径, 单位: 米 (m)
    Z = 0.75  # 解吸塔填料层高度, 单位: 米 (m)
    A = np.full((3,), np.pi * (D / 2) ** 2)  # 解吸塔截面积, 单位: 平方米 (m²)
    p_0 = 1.013e5  # 大气压, 单位: 帕斯卡 (Pa)

    # 初值猜测
    initial_guess = [1.0, 1.0]  # 替换为初始猜测的值

    df = pd.read_excel(file_dir, header=None, sheet_name=sheet_name)
    data = df.iloc[2:, 1:].apply(pd.to_numeric, errors='coerce').values

    # 提取有效数据
    # V_氧 = data[:, 0]  # 氧气流量, 单位: 立方米每小时 (m³/h)
    V_水 = data[:, 1]  # 水流量, 单位: 立方米每小时 (m³/h)
    V_空 = data[:, 2]  # 空气流量, 单位: 立方米每小时 (m³/h)
    ΔP_U管压差 = ρ_水 * g * data[:, 3] * 1e-3  # U管压差, 单位: 帕斯卡 (Pa)
    # ΔP_仪器显示 = data[:, 4] * 1e3  # 仪器显示的压差, 单位: 帕斯卡 (Pa)
    c_富氧水 = data[:, 5]  # 富氧水中的氧浓度, 单位: 毫克每升 (mg/L)
    c_贫氧水 = data[:, 6]  # 贫氧水中的氧浓度, 单位: 毫克每升 (mg/L)
    t_水 = data[:, 7]  # 水的温度, 单位: 摄氏度 (°C)

    L = ρ_水/M_H2O * V_水  # 水的质量流量, 单位: 千克每小时 (kmol/h)
    V = ρ_空/M_空 * V_空  # 空气的质量流量, 单位: 千克每小时 (kmol/h)
    x_1 = (c_富氧水 * 1 / M_O2) / ((c_富氧水 * 1 / M_O2) * 1e-3 + 1e6 / 18)  # 富氧水中的溶解氧浓度
    x_2 = (c_贫氧水 * 1 / M_O2) / ((c_贫氧水 * 1 / M_O2) * 1e-3 + 1e6 / 18)  # 贫氧水中的溶解氧浓度

    G_A = L / (x_1 - x_2)  # 氧气质量流量, 单位: 千克每小时 (kg/h)
    Ω = A  # 截面积, 单位: 平方米 (m²)
    V_p = Z * Ω  # 塔体积, 单位: 立方米 (m³)
    p = p_0 + 1 / 2 * ΔP_U管压差  # 实际压力, 单位: 帕斯卡 (Pa)
    m = E(t_水) / p  # 相平衡常数,量纲1

    y1, y2 = 21, 21  # 空气中氧的浓度,按理想气体估算
    x_1_star = y1 / m  # x_1*
    x_2_star = y2 / m  # x_2*
    Δx_m = ((x_1 - x_1_star) - (x_2 - x_2_star))/np.log((x_1 - x_1_star)/(x_2 - x_2_star))  # 液相浓度的对数平均值
    K_x_times_a = G_A / (V_p * Δx_m)  # 以液相浓度表示的总体积吸收系数,单位: kmol/(m³·h·Δx)

    # 使用 curve_fit 进行拟合
    params, covariance = curve_fit(correlation_func, (A, L, V), K_x_times_a, p0=initial_guess)

    # 提取拟合参数
    a_fit = params[0]
    b_fit = params[1]

    # 打印拟合结果
    print(f"工作表 {sheet_name} 中最优化拟合得到的参数：a={a_fit}, b={b_fit}")

    plt.rcParams['font.family'] = 'simhei'
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制拟合结果与实际数据的比较图
    plt.figure(figsize=(8, 6))
    plt.plot(K_x_times_a, correlation_func((A, L, V), a_fit, b_fit), 'o', c='r', label='（K_x_times_a, A * L^a * V^b）')
    plt.xlabel('K_x_times_a')
    plt.ylabel('A * L^a * V^b')
    plt.title(f'拟合结果vs.实际数据比较 - {sheet_name}')
    plt.legend()
    plt.grid(True)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.savefig(fr'./拟合图结果/{sheet_name}.png', dpi=300)
    plt.show()

# 遍历工作表
for sheet_name in sheet_names[2:4]:
    Oxygen_Desorption_Mass_Transfer(file_dir, sheet_name)

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