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

# 设置中文显示和负号正常显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 加载Excel文件
file_dir = r'./干燥原始数据记录表(非).xlsx'
excel_file = pd.ExcelFile(file_dir)

# 获取工作表名
sheet_names = excel_file.sheet_names

# 从第一个工作表读取数据
df1 = pd.read_excel(file_dir, header=None, sheet_name=sheet_names[0])
data1 = df1.iloc[:, 1].values

# 将第一个工作表的数据分配给变量
m_1 = data1[0] * 1e-3
m_2 = data1[1] * 1e-3
W2 = data1[2] * 1e-3
G_prime = data1[3] * 1e-3
ΔP = data1[4]


# 从第二个工作表读取数据
df2 = pd.read_excel(file_dir, header=None, sheet_name=sheet_names[1])
data2 = df2.iloc[1:, 1:].values

# 将第二个工作表的数据分配给变量
τ = data2[:,0]/60
W1 = data2[:,1] * 1e-3
t = data2[:,2]
tw = data2[:,3]

# 定义常数
r_tw = 2490
S = 2.64 * 1e-2

# 进行计算
τ_bar = (τ[:-1] + τ[1:]) / 2
G = (W1 - W2)
X = (G - G_prime) / G_prime

# 保存 G, X 计算结果
ans1 = np.array([G * 1000,X]).T

# 计算相邻数据点的平均值
X_bar = (X[:-1] + X[1:]) / 2
U = -(G_prime / S) * (np.diff(X) / np.diff(τ))

# 确保从恒速干燥阶段取得 U_c
U_c = np.mean(U[15:])

# 保存 X_bar, U计算结果
ans2 = np.array([X_bar,U]).T

# 绘制 X vs τ 图
plt.figure(figsize=(8, 6), dpi=125)
plt.scatter(τ_bar, X_bar, marker='o', color='r', label='平均拟合')
plt.plot(τ_bar, X_bar, linestyle='-', color='k', label='平均拟合')
plt.title("干燥曲线", fontsize=12)
plt.xlabel(r"$\tau/h$", fontsize=12)
plt.ylabel(r"$X/(kg·kg^{-1}$干基)", fontsize=12)
plt.grid(True, which='both')
plt.minorticks_on()  # 打开次要刻度线

# 设置坐标轴样式
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)

# 设置坐标轴从 (0, 0) 开始
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.legend(fontsize=14)
plt.savefig('./拟合图结果/1.png', dpi=300)
plt.show()

# 绘制 U vs X 图
plt.figure(figsize=(8, 6), dpi=125)
plt.scatter(X_bar, U, marker='o', color='r')  # 红点
plt.title(r"干燥速率曲线", fontsize=12)
plt.xlabel(r"$X/(kg·kg^{-1}$干基)", fontsize=12)
plt.ylabel(r"$U/(kg·m^{-2}·h^{-1})$", fontsize=12)
plt.grid(True, which='both')
plt.minorticks_on()  # 打开次要刻度线

# 设置坐标轴样式
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)

# 设置坐标轴从 (0, 0) 开始
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.savefig('./拟合图结果/2.png', dpi=300)
plt.show()


# 定义额外的常数
C_0 = 0.65
A_0 = (np.pi * 0.040 ** 2) / 4
ρ_空气 = 1.29
t0 = 25

# 进行进一步的计算
α = (U_c * r_tw) / (t - tw)
V_t0 = C_0 * A_0 * np.sqrt(2 * ΔP / ρ_空气)
V_t = V_t0 * (273 + t) / (273 + t0)


'''
图像整合到同一页中
'''
import matplotlib.image as mpimg
# 读取图像文件
images = []
for i in range(1,3):
    img = mpimg.imread(f'./拟合图结果/{i}.png')
    images.append(img)

# 创建一个4x2布局的图
fig, axes = plt.subplots(2, 1, figsize=(16,9),dpi=125)

# 遍历每个子图并显示相应的图像
for ax, img in zip(axes.flatten(), images):
    ax.imshow(img)
    ax.axis('off')  # 隐藏坐标轴

# 调整布局，减少图像之间的间隙
plt.subplots_adjust(wspace=0.01, hspace=0.01)

# 保存图像并移除多余的边框
plt.savefig(r'./拟合图结果/拟合图整合图.png', bbox_inches='tight',dpi=300)

# 显示图像
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
