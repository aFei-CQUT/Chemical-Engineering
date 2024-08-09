# This project is created by aFei-CQUT
# ------------------------------------------------------------------------------------------------------------------------------------
#   About aFei-CQUT
# - Interests&Hobbies: Programing,  ChatGPT,  Reading serious books,  Studying academical papers.
# - CurrentlyLearning: Mathmodeling,Python and Mathmatica (preparing for National College Mathematical Contest in Modeling).
# - Email:2039787966@qq.com
# - Pronouns: Chemical Engineering, Computer Science, Enterprising, Diligent, Hard-working, Sophomore,Chongqing Institute of Technology,
# - Looking forward to collaborating on experimental data processing of chemical engineering principle
# ------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import sympy
import matplotlib.pyplot as plt

file_dir = r'./精馏原始记录表(非).xlsx'
xls = pd.ExcelFile(file_dir)
sheet_names = xls.sheet_names
df = pd.read_excel(file_dir, header=None, sheet_name=sheet_names[0])
# =============================================================================
# 现有程序不适合直接处理实验数据,相应的参数和公式说明如下（只是举例,数据不一定对）
# 
# α = 2.5
# R = 2.0
# 
# A组分在原料液中的体积分数
# sF = 21
# sD = 90
# sW = 40
# 
# A、B组分的密度
# ρA = 0.789
# ρB = 1000
# 
# A、B组分的比热容
# cA = 70
# cB = 90
# 
# A、B组分的摩尔分数
# xA = 0.4
# xB = 0.6
# 
# A、B组分的摩尔汽化热
# rA = 850
# rB = 2260
# 
# A、B组分的摩尔质量
# MA = 46
# MB = 18
# 
# A、B组分的平均比热容、平均摩尔汽化热
# cpm = xA * cA * MA + xB * cB * MB
# rm = xA * rA * MA + xB * rB * MB
# 
# 查询参数后代入
# tS = None
# tF = 31
# 
# 热状态参数
# q = (cpm * (tS - tF) + rm) / rm
# 
# F = 100
# D = None
# W = None
# 
# xF = (sF * ρA / MA) / ((sF * ρA / MA) + (1 - sF) * ρB / MB)
# xD = (sD * ρA / MA) / ((sD * ρA / MA) + (1 - sD) * ρB / MB)
# xW = (sW * ρA / MA) / ((sW * ρA / MA) + (1 - sW) * ρB / MB)
# =============================================================================

# 定义平衡线方程 y_e(x)
def y_e(x):
    y = α * x / (1 + (α - 1) * x)
    return y

# 定义反平衡线方程 x_e(y)
def x_e(y):
    x = y / (α - (α - 1) * y)
    return x

# 精馏段操作方程 y_np1(x)
def y_np1(x):
    y = R / (R + 1) * x + xD / (R + 1)
    return y

# 提馏段操作方程 y_mp1(x)
def y_mp1(x):
    y = (L + q * F) / (L + q * F - W) * x - W / (L + q * F - W) * xW
    return y

# q 线方程 y_q(x)
def y_q(x):
    if q == 1:
        y = 0
    else:
        y = q / (q - 1) * x - 1 / (q - 1) * xF
    return y

'''
精馏图解理论板层数
'''

# 待求量以 None 表示
F, D, W, xF, xD, xW, R = [100, None, None, 0.5, 0.97, 0.04, 2]
q, α = [1, 2.5]

# 提馏段操作方程需要 D、W
# 使用 Sympy 列出物料衡算的矩阵形式并求解
A = sympy.Matrix([[1, 1], [xD, xW]])
b = sympy.Matrix([F, xF * F])
D, W = A.solve(b)
L = R * D

# 计算相应数据数组
# x 数据数组
x_array = np.linspace(0, 1, 50)
# y_q 数据数组
y_q_array = y_q(x_array)
# y_e 数据数组
y_e_array = y_e(x_array)
# y_np1 数据数组
y_np1_array = y_np1(x_array)
# y_mp1 数据数组
y_mp1_array = y_mp1(x_array)

# 确定 Q 点
xQ = ((R + 1) * xF + (q - 1) * xD) / (R + q)
yQ = (xF * R + q * xD) / (R + q)

# 逐板计算,求解每个塔板的平衡情况
# n 从 0 开始计,n=0 时有 y_0 = x_D
yn = np.array([xD])
xn = np.array([])
NT = None
while x_e(yn[-1]) > xW:
    xn = np.append(xn, x_e(yn[-1]))
    if xn[-1] > xQ:
        yn = np.append(yn, y_np1(xn[-1]))
    else:
        yn = np.append(yn, y_mp1(xn[-1]))
else:
    xn = np.append(xn, x_e(yn[-1]))
    NT = len(xn)

# 图解法计算理论塔板数的图示数据
xNT = np.array([xD])
yNT = np.array([xD])
for n, i in enumerate(xn):
    xNT = np.append(xNT, i)
    yNT = np.append(yNT, yn[n])
    xNT = np.append(xNT, i)
    if i >= xQ:
        yNT = np.append(yNT, y_np1(i))
    else:
        yNT = np.append(yNT, y_mp1(i))

# 作图设置
plt.figure(figsize=(8, 6), dpi=125)
plt.xlim(0, 1)
plt.ylim(0, 1)

# 设置 matplotlib.pyplot 字体显示正常
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 作图
# 对角线
y_array = x_array
plt.plot(x_array, y_array, ls="-", label="对角线")

xQ_plot = [0.5, 0.5]
yQ_plot = [0.5, 0.7]

plt.plot(xQ_plot, yQ_plot, label="q 线")
plt.plot(x_array, y_e_array, label="平衡线")
plt.plot(x_array, y_np1_array, label="精馏操作线")
plt.plot(x_array, y_mp1_array, label="提馏操作线")
plt.plot(xn, yn, label="塔板操作平衡点", ls=":", marker="+", markersize=10)
plt.plot(xNT, yNT, label="图解法—理论塔板", ls=":")

# 画点
plt.plot(xD, xD, marker=".", markersize=10)  
plt.plot(xW, xW, marker=".", markersize=10)
plt.plot(xQ_plot, yQ_plot, marker=".", markersize=10)

# 点注释
plt.annotate("$W$ 点", xy=(xW, xW), xytext=(xW + 0.05, xW), arrowprops=dict(arrowstyle="->"))
plt.annotate("$D$ 点", xy=(xD, xD), xytext=(xD, xD - 0.05), arrowprops=dict(arrowstyle="->"))
plt.annotate("$Q$ 点", xy=(xQ, yQ), xytext=(xQ, yQ - 0.05), arrowprops=dict(arrowstyle="->"))
plt.legend()

# 设置坐标轴顶部线条粗细
plt.gca().spines["top"].set_linewidth(2)
# 设置坐标轴底部线条粗细
plt.gca().spines["bottom"].set_linewidth(2)
# 设置坐标轴左侧线条粗细
plt.gca().spines["left"].set_linewidth(2)
# 设置坐标轴右侧线条粗细
plt.gca().spines["right"].set_linewidth(2)
plt.gca().grid()
# 图中显示所需理论板数
plt.text(x=0.6, y=0.4, s="所需理论板数：%d" % (len(xn) - 1))
# 图标题
plt.title("图解理论板数")
plt.ylabel("$y$")
plt.xlabel("$x$")
plt.savefig(r'./拟合图结果/图解理论板数.png', dpi=300)
plt.show()

ans_df_x_y = pd.DataFrame({
    "x": x_array,
    "y": y_array,
})
ans_df_xn_yn = pd.DataFrame({
    "yn": yn,
    "xn": xn
})

ans_df_Q = pd.DataFrame({"xQ": [xQ], "yQ": [yQ]})

NT_minus_1 = NT - 1

print(f"\n理论板层数为{NT_minus_1}")

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
