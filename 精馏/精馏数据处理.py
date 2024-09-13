import pandas as pd
import numpy as np
import sympy
import matplotlib.pyplot as plt
import zipfile
import os

# 读取Excel文件
file_dir = r'./精馏原始记录表(非).xlsx'
df = pd.read_excel(file_dir, header=0)

# 定义计算x_乙醇/1的函数
def calculate_x_ethanol(s, ρA=0.789, ρB=1.0, MA=46, MB=18):
    return (s * ρA / MA) / ((s * ρA / MA) + ((100 - s) * ρB / MB))

# 计算x_乙醇/1并添加到DataFrame
df['x_乙醇/1'] = df['20°C酒精度(查表)/°'].apply(lambda s: calculate_x_ethanol(s))

# 提取所需的数据
xD = df.loc[2, 'x_乙醇/1']  # 馏出液中乙醇的摩尔分数（部分回流(R=4)的馏出液）
xW = df.loc[3, 'x_乙醇/1']  # 釜残液中乙醇的摩尔分数（部分回流(R=4)的釜残液）
xF = df.loc[4, 'x_乙醇/1']  # 进料液中乙醇的摩尔分数（部分回流(R=4)的原料液）

# 设置其他参数
R = 4    # 回流比
α = 2.5  # 相对挥发度,这里使用一个估计值,实际上可能需要根据实验数据计算
q = 1    # 热状态参数,假设为1,即沸点进料
F = 100  # 假设进料量为100

# 使用Sympy求解物料平衡方程
A = sympy.Matrix([[1, 1], [xD, xW]])
b = sympy.Matrix([F, xF * F])
D, W = A.solve(b)  # D为馏出液量,W为釜残液量
L = R * D          # 回流量

# 定义平衡线方程
def y_e(x):
    return α * x / (1 + (α - 1) * x)

# 定义反平衡线方程
def x_e(y):
    return y / (α - (α - 1) * y)

# 定义精馏段操作线方程
def y_np1(x):
    return R / (R + 1) * x + xD / (R + 1)

# 定义提馏段操作线方程
def y_mp1(x):
    return (L + q * F) / (L + q * F - W) * x - W / (L + q * F - W) * xW

# 定义q线方程
def y_q(x):
    if q == 1:
        return x
    else:
        return q / (q - 1) * x - 1 / (q - 1) * xF

# 计算Q点坐标
xQ = ((R + 1) * xF + (q - 1) * xD) / (R + q)
yQ = (xF * R + q * xD) / (R + q)

# 逐板计算
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

# 准备绘图数据
x_array = np.linspace(0, 1, 50)
y_array = x_array
y_e_array = y_e(x_array)
y_np1_array = y_np1(x_array)
y_mp1_array = y_mp1(x_array)

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

# 绘图设置
plt.figure(figsize=(8, 6), dpi=125)
plt.xlim(0, 1)
plt.ylim(0, 1)

# 设置matplotlib.pyplot字体显示正常
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 绘制各条线
plt.plot(x_array, y_array, ls="-", label="对角线")
plt.plot(x_array, y_e_array, label="平衡线")
plt.plot(x_array, y_np1_array, label="精馏操作线")
plt.plot(x_array, y_mp1_array, label="提馏操作线")
plt.plot(xn, yn, label="塔板操作平衡点", ls=":", marker="+", markersize=10)
plt.plot(xNT, yNT, label="图解法—理论塔板", ls=":")

# 绘制特殊点
plt.plot(xD, xD, marker=".", markersize=10)
plt.plot(xW, xW, marker=".", markersize=10)
plt.plot(xQ, yQ, marker=".", markersize=10)

# 添加注释
plt.annotate("$W$ 点", xy=(xW, xW), xytext=(xW + 0.05, xW), arrowprops=dict(arrowstyle="->"))
plt.annotate("$D$ 点", xy=(xD, xD), xytext=(xD, xD - 0.05), arrowprops=dict(arrowstyle="->"))
plt.annotate("$Q$ 点", xy=(xQ, yQ), xytext=(xQ, yQ - 0.05), arrowprops=dict(arrowstyle="->"))

# 设置图例和坐标轴
plt.legend()
plt.gca().spines["top"].set_linewidth(2)
plt.gca().spines["bottom"].set_linewidth(2)
plt.gca().spines["left"].set_linewidth(2)
plt.gca().spines["right"].set_linewidth(2)
plt.gca().grid()

# 添加理论板数信息
plt.text(x=0.6, y=0.4, s="所需理论板数：%d" % (len(xn) - 1))

# 设置标题和轴标签
plt.title("图解理论板数")
plt.ylabel("$y$")
plt.xlabel("$x$")

# 保存图片
plt.savefig(r'./拟合图结果/图解理论板数.png', dpi=300)
plt.show()

# 打印理论板层数
print(f"\n理论板层数为{NT - 1}")

# 准备结果数据
ans_df_x_y = pd.DataFrame({
    "x": x_array,
    "y": y_array,
})
ans_df_xn_yn = pd.DataFrame({
    "yn": yn,
    "xn": xn
})
ans_df_Q = pd.DataFrame({"xQ": [xQ], "yQ": [yQ]})

# 将结果保存到Excel文件
with pd.ExcelWriter('计算结果.xlsx') as writer:
    ans_df_x_y.to_excel(writer, sheet_name='x_y数据', index=False)
    ans_df_xn_yn.to_excel(writer, sheet_name='xn_yn数据', index=False)
    ans_df_Q.to_excel(writer, sheet_name='Q点数据', index=False)
    df.to_excel(writer, sheet_name='原始数据', index=False)

print("计算结果已保存到'计算结果.xlsx'文件中。")

# 压缩拟合图结果
dir_to_zip = r'./拟合图结果'
dir_to_save = r'./拟合图结果.zip'

with zipfile.ZipFile(dir_to_save, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(dir_to_zip):
        for file in files:
            file_dir = os.path.join(root, file)
            arc_name = os.path.relpath(file_dir, dir_to_zip)
            zipf.write(file_dir, arc_name)

print(f'压缩完成，文件保存为: {dir_to_save}')
