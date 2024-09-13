import pandas as pd
import numpy as np
import sympy
import matplotlib.pyplot as plt
import zipfile
import os

# 读取Excel文件
file_dir = r'./精馏原始记录表(非).xlsx'
df = pd.read_excel(file_dir, header=0)

# 设置其他参数
R = 4                # 回流比
αm = 3.0             # 相对挥发度，准确来讲应该是整个过程的平均相对挥发度由塔顶、塔底温度定性此处取一个示例值3.0
F = 80               # 进料量,很早做的实验,需要根据实际情况调整
tS = 30              # 泡点温度，很早做的实验，搞忘了，此处推测不会太高
tF = 26              # 进料温度，后续可以根据实验数据确定

# 计算进料热状态参数
ρA, ρB = 0.789, 1.0                             # 乙醇和水的密度
cA, cB = 2.4*1e+3, 4.189*1e+3                   # 乙醇和水的比热容
rA, rB = 850, 2260                              # 摩尔汽化热
MA, MB = 46, 18                                 # 乙醇和水的摩尔质量
xA, xB = 0.1, 0.9                               # 进料时乙醇和水的摩尔分数，用来确定cpm，rm
cpm = xA * cA * MA + xB * cB * MB               # 平均比热容
rm = xA * rA * MA + xB * rB * MB                # 平均摩尔汽化热
q = (cpm * (tS - tF) + rm) / rm if tS else 1.5  # 进料热状态参数由上述逻辑计算，若少了任意一个必要的参数则假设为1.5

# 定义计算x_乙醇/1的函数
def calculate_x_ethanol(s, ρA=0.789, ρB=1.0, MA=46, MB=18):
    # s 为乙醇在混合液中的体积分数
    return (s * ρA / MA) / ((s * ρA / MA) + ((100 - s) * ρB / MB))

# 回流比 R = ∞ 时，计算x_乙醇
xD_inf = calculate_x_ethanol(df.loc[0, '20°C酒精度(查表)/°'])  # 回流比 R = ∞ 时，馏出液中乙醇的摩尔分数
xW_inf = calculate_x_ethanol(df.loc[1, '20°C酒精度(查表)/°'])  # 回流比 R = ∞ 时，釜残液中乙醇的摩尔分数

# 回流比 R = 4 时，计算x_乙醇
xD = calculate_x_ethanol(df.loc[2, '20°C酒精度(查表)/°'])      # 回流比 R = 4 时，馏出液中乙醇的摩尔分数
xW = calculate_x_ethanol(df.loc[3, '20°C酒精度(查表)/°'])      # 回流比 R = 4 时，釜残液中乙醇的摩尔分数

# 只要进的料一样，无论时 R = ∞ 或 4 ，xF应该一致
xF = calculate_x_ethanol(df.loc[4, '20°C酒精度(查表)/°'])      # 进料液中乙醇的摩尔分数

# 使用Sympy求解物料平衡方程
A = sympy.Matrix([[1,1],[xD,xW]])
b = sympy.Matrix([F,xF * F])
D,W = A.solve(b)                                # D为馏出液量,W为釜残液量
L = R * D                                       # 回流量

# 定义平衡线方程
def y_e(x):
    return αm * x / (1 + (αm - 1) * x)

# 定义反平衡线方程
def x_e(y):
    return y / (αm - (αm - 1) * y)

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

# 设置最大迭代次数以防出现在恒浓区的情况
max_iterations = 20
iteration = 0

# 开始循环计算,并输出每次的xn和yn值
while x_e(yn[-1]) > xW and iteration < max_iterations:
    iteration += 1
    xn = np.append(xn,x_e(yn[-1]))
    print(f"迭代次数: {iteration},当前 xn 值：{xn[-1]}")
    
    if xn[-1] > xQ:
        yn = np.append(yn,y_np1(xn[-1]))
    else:
        yn = np.append(yn,y_mp1(xn[-1]))
        
    print(f"迭代次数: {iteration},当前 yn 值：{yn[-1]}")
    
if iteration == max_iterations:
    print("达到最大迭代次数,可能陷入死循环")

# 最终计算NT
xn = np.append(xn,x_e(yn[-1]))
NT = len(xn)

# 打印最终理论板层数
print(f"\n理论板层数为 {NT - 1}")

# 准备绘图数据
x_array = np.linspace(0,1,50)
y_array = x_array
y_e_array = y_e(x_array)
y_np1_array = y_np1(x_array)
y_mp1_array = y_mp1(x_array)

# 图解法计算理论塔板数的图示数据
xNT = np.array([xD])
yNT = np.array([xD])
for n,i in enumerate(xn):
    xNT = np.append(xNT,i)
    yNT = np.append(yNT,yn[n])
    xNT = np.append(xNT,i)
    if i >= xQ:
        yNT = np.append(yNT,y_np1(i))
    else:
        yNT = np.append(yNT,y_mp1(i))

# 绘图设置
plt.figure(figsize=(8,6),dpi=125)
plt.xlim(0,1)
plt.ylim(0,1)

# 设置matplotlib.pyplot字体显示正常
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 绘制各条线
plt.plot(x_array,y_array,ls="-",label="对角线")
plt.plot(x_array,y_e_array,label="平衡线")
plt.plot(x_array,y_np1_array,label="精馏操作线")
plt.plot(x_array,y_mp1_array,label="提馏操作线")
plt.plot(xn,yn,label="塔板操作平衡点",ls=":",marker="+",markersize=10)
plt.plot(xNT,yNT,label="图解法—理论塔板",ls=":")

# 绘制特殊点
plt.plot(xD,xD,marker=".",markersize=10)
plt.plot(xW,xW,marker=".",markersize=10)
plt.plot(xQ,yQ,marker=".",markersize=10)

# 添加注释
plt.annotate("$W$ 点",xy=(xW,xW),xytext=(xW + 0.05,xW),arrowprops=dict(arrowstyle="->"))
plt.annotate("$D$ 点",xy=(xD,xD),xytext=(xD,xD - 0.05),arrowprops=dict(arrowstyle="->"))
plt.annotate("$Q$ 点",xy=(xQ,yQ),xytext=(xQ,yQ - 0.05),arrowprops=dict(arrowstyle="->"))

# 设置图例和坐标轴
plt.legend()
plt.gca().spines["top"].set_linewidth(2)
plt.gca().spines["bottom"].set_linewidth(2)
plt.gca().spines["left"].set_linewidth(2)
plt.gca().spines["right"].set_linewidth(2)
plt.gca().grid()

# 添加理论板数信息
plt.text(x=0.6,y=0.4,s="所需理论板数：%d" % (len(xn) - 1))

# 设置标题和轴标签
plt.title("图解理论板数")
plt.ylabel("$y$")
plt.xlabel("$x$")

# 保存图片
plt.savefig(r'./拟合图结果/图解理论板数.png',dpi=300)
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
ans_df_Q = pd.DataFrame({"xQ": [xQ],"yQ": [yQ]})

# 压缩拟合图结果
dir_to_zip = r'./拟合图结果'
dir_to_save = r'./拟合图结果.zip'

with zipfile.ZipFile(dir_to_save,'w',zipfile.ZIP_DEFLATED) as zipf:
    for root,dirs,files in os.walk(dir_to_zip):
        for file in files:
            file_dir = os.path.join(root,file)
            arc_name = os.path.relpath(file_dir,dir_to_zip)
            zipf.write(file_dir,arc_name)

print(f'压缩完成,文件保存为: {dir_to_save}')
