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
from scipy.integrate import trapz
from scipy.interpolate import interp1d


'''
导入数据
'''

# 设置绘图正常显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 125

# 文件路径
file_dir = './萃取原始数据记录表(非).xlsx'

# 使用 ExcelFile 读取文件
excel_file = pd.ExcelFile(file_dir)
sheet_names = excel_file.sheet_names

# 读取第一个工作表的内容
df1 = pd.read_excel(file_dir, sheet_name=sheet_names[0], header=None)
data1 = df1.iloc[1:, 1:].values

# 提取所需数据
n = data1[0]
Vs_S = data1[1] # L/h
Vs_B = data1[2] # L/h

t = data1[3]
c_NaOH = 0.01
V_to_be_titrated = data1[[4,6,8],:]
V_NaOH_used = data1[[5,7,9],:]

# 分子量
M_A = 78      # kg/kmol
M_B = 122     # kg/kmol
M_S = 18      # kg/kmol

# 密度
ρ_A = 876.7   # kg/m^3
ρ_B = 800     # kg/m^3
ρ_S = 1000    # kg/m^3

# 对修正煤油体积流量
Vs_B_rect = Vs_S * np.sqrt(ρ_S*(7900-ρ_B)/(ρ_B*(7900-ρ_S ))) # L/h

'''
数据预处理
'''

# 计算X_Rb, X_Rt,Y_Eb
ans1 = (c_NaOH * V_NaOH_used * 1e-6 * M_B) / (ρ_B * V_to_be_titrated * 1e-6)

# 获取X_Rb, X_Rt,Y_Eb
X_Rb = ans1[0]
X_Rt = ans1[1]
Y_Eb = ans1[2]

# 计算煤油和水的质量流量
B = ρ_B * Vs_B * 1e-3  # L转换为m^3
S = ρ_S * Vs_S * 1e-3  # L转换为m^3
B_rect = ρ_B * Vs_B_rect * 1e-3 # L转换为m^3

ans2 = np.array([B, S, B_rect])

'''
分配曲线
'''

# 读取第三个工作表的内容
df3 = pd.read_excel(file_dir, sheet_name=sheet_names[2], header=None)
data3 = df3.iloc[2:, :].values
X3_data = data3[:, 0].astype(float)
Y3_data = data3[:, 1].astype(float)

# 定义拟合模型函数,此处调用泰勒多项式返回函数值
def model_function(X, *coefficients):
    return np.polyval(coefficients, X)

# 泰勒多项式的阶数
order = 3

# 多项式拟合
coefficients = np.polyfit(X3_data, Y3_data, order)

# 使用拟合参数生成拟合曲线
X3_to_fit = np.linspace(min(X3_data), max(X3_data), 100)
Y_fitted = model_function(X3_to_fit, *coefficients)

# 计算操作线1的斜率和截距
k1 = (0 - Y_Eb[0]) / (X_Rt[0] - X_Rb[0])
b1 = Y_Eb[0] - k1 * X_Rb[0]

# 计算操作线2的斜率和截距
k2 = (0 - Y_Eb[1]) / (X_Rt[1] - X_Rb[1])
b2 = Y_Eb[1] - k2 * X_Rb[1]


# =============================================================================
# # 计算操作线1的斜率和截距
# k1 = B_rect[0] / S
# b1 = Y_Eb[0] - k1 * X_Rb[0]
# 
# # 计算操作线2的斜率和截距
# k2 = B_rect[0] / S
# b2 = Y_Eb[1] - k2 * X_Rb[1]
# 
# =============================================================================

# 定义操作线方程形式1: Y = kX + b
def operating_line_to_calculate_Y(X, k, b):
    return k * X + b

# 定义操作线方程形式2: X = (Y - b) / k
def operating_line_to_calculate_X(Y, k, b):
    return (Y - b) / k

# 生成用于绘制操作线的点
X_operating1 = np.linspace(0, X_Rb[0], 500)
Y_operating1 = operating_line_to_calculate_Y(X_operating1, k1, b1)

X_operating2 = np.linspace(0, X_Rb[1], 500)
Y_operating2 = operating_line_to_calculate_Y(X_operating2, k2, b2)

# 绘制分配曲线和操作线
plt.figure(figsize=(8, 6))

# 分配曲线
plt.scatter(X3_data, Y3_data, color='purple', marker='^', label='分配曲线数据点')
plt.plot(X3_to_fit, Y_fitted, color='k', label='分配曲线')

# 操作线1
plt.scatter([X_Rb[0], X_Rt[0]], [Y_Eb[0], 0], color='green', marker='o', label='操作线1定点')
eq1_text = f'操作线1方程: Y = {k1:.2f} * X + {b1:.2f}'
plt.text(X_Rb[1]/2, Y_Eb[0]/2 + 0.25 * max(Y3_data), eq1_text, fontsize=10, fontweight='bold', color='k')
plt.plot(X_operating1, Y_operating1, linestyle='--', color='green', label='操作线1方程')

# 操作线2
plt.scatter([X_Rb[1], X_Rt[1]], [Y_Eb[1], 0], color='orange', marker='o', label='操作线2定点')
eq2_text = f'操作线2方程: Y = {k2:.2f} * X + {b2:.2f}'
plt.text(X_Rb[1]/2, Y_Eb[1]/2 + 0.15* max(Y3_data), eq2_text, fontsize=10, fontweight='bold', color='k')
plt.plot(X_operating2, Y_operating2, linestyle='--', color='orange', label='操作线2方程')

# 图表设置
plt.title('分配曲线与操作线')
plt.xlabel('X 数据')
plt.ylabel('Y 数据')
plt.legend()
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.grid(True, which='both')
plt.minorticks_on()
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)

# 保存图像
plt.savefig('./拟合图结果/1', dpi=300)

# 显示图形
plt.show()


'''
图解积分
'''
# 图解积分数据存储变量data5_for_graph_integral
data5_for_graph_integral = []
# k1,k2放到一个数组中
k = np.array([k1,k2])
# b1,b2放到一个数组中
b = np.array([b1,b2])


# 定义每组数据的剔除区间范围
min_Y5_group1 = 0
max_Y5_group1 = 0.002


min_Y5_group2 = 0
max_Y5_group2 = 0.002


# 遍历低转速、高转速数据
for i in range(len(Y_Eb)):
    Y5_Eb_data = np.linspace(0, Y_Eb[i], 20)
    
# =============================================================================
#     这个物料衡算有错：X_Rb_data = X_Rb[i] + (S[i] / B_rect[i]) * (Y5_Eb_data - 0)
# =============================================================================
    
    # 计算对应的 X_Rb_data
    X_Rb_data = operating_line_to_calculate_X(Y5_Eb_data, k[i], b[i])
    
    # 计算对应的 Y5star_data
    Y5star_data = model_function(X_Rb_data, *coefficients)

    # 计算 1/(Y5star_data - Y5_Eb_data)
    one_over_Y5star_minus_Y5 = 1 / (Y5star_data - Y5_Eb_data)
    
    # 记录图解积分数据
    data5_for_graph_integral.append(Y5_Eb_data)
    data5_for_graph_integral.append(X_Rb_data)
    data5_for_graph_integral.append(Y5star_data)
    data5_for_graph_integral.append(one_over_Y5star_minus_Y5)
    
    # 使用插值使得曲线平滑
    interp_func = interp1d(Y5_Eb_data, one_over_Y5star_minus_Y5, kind='cubic')
    Y5_Eb_data_smooth = np.linspace(Y5_Eb_data.min(), Y5_Eb_data.max(), 40)
    one_over_Y5star_minus_Y5_smooth = interp_func(Y5_Eb_data_smooth)
    
    # 根据不同组数据应用剔除区间范围
    if i == 0:
        min_Y, max_Y = min_Y5_group1, max_Y5_group1
    else:
        min_Y, max_Y = min_Y5_group2, max_Y5_group2
    
    # 剔除区间范围外的值
    mask = (Y5_Eb_data_smooth >= min_Y) & (Y5_Eb_data_smooth <= max_Y)
    Y5_Eb_data_smooth = Y5_Eb_data_smooth[mask]
    one_over_Y5star_minus_Y5_smooth = one_over_Y5star_minus_Y5_smooth[mask]
    
    
    '''
    绘制原始数据和积分区域的阴影
    '''
    
    plt.figure(figsize=(8, 6))
    
    plt.scatter(Y5_Eb_data_smooth, one_over_Y5star_minus_Y5_smooth, color='r', label=f'$\\frac{{1}}{{Y_{{5}}^*-Y_{{5}}}}$ 数据组 {i+1}')
    plt.plot(Y5_Eb_data_smooth, one_over_Y5star_minus_Y5_smooth, linestyle='-', color='k')
    plt.fill_between(Y5_Eb_data_smooth, one_over_Y5star_minus_Y5_smooth, alpha=0.5, color='gray')
    
    # 坐标轴名称&图例&图标题
    plt.xlabel('$Y$')
    plt.ylabel('$\\frac{1}{Y^{*}-Y}$')
    plt.legend()
    plt.title(f'$\\frac{{1}}{{Y^*-Y}} - Y$       数据组 {i+1}')
    
    # 图表设置
    plt.xlim(left=min_Y)
    plt.ylim(bottom=0)
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    
    # 数值积分:梯形积分(trapezoidal integration)
    integral_value = trapz(one_over_Y5star_minus_Y5_smooth, Y5_Eb_data_smooth)
    plt.text(0.5, -0.15, f'数值积分结果: {integral_value:.5f}', transform=plt.gca().transAxes, horizontalalignment='center')
    
    # 保存图片
    plt.savefig(f'./拟合图结果/{i+2}', dpi=300)
    plt.show()

ans3 = np.array(data5_for_graph_integral)


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

