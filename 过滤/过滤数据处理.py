import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.image as mpimg
import zipfile
import os

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 125
plt.rcParams['savefig.dpi'] = 300

class FilterDataProcessor:
    def __init__(self, data_file):
        self.data_file = data_file  # 数据文件路径
        self.data = self.load_data()  # 加载数据
        self.delta_theta_over_delta_q_to_fit_lists = []  # 初拟合 Δθ/Δq 列表
        self.q_to_fit_lists = []  # 初拟合 q 列表
        self.delta_theta_over_delta_q_to_refit_lists = []  # 重新拟合 Δθ/Δq 列表
        self.q_to_refit_lists = []  # 重新拟合 q 列表
        self.inifit_slopes = []  # 初拟合斜率
        self.inifit_intercepts = []  # 初拟合截距
        self.refit_slopes = []  # 重新拟合斜率
        self.refit_intercepts = []  # 重新拟合截距
        self.ans1_inifit = []  # 初拟合结果1
        self.ans2_inifit = []  # 初拟合结果2
        self.ans3_refit = []  # 重新拟合结果1
        self.ans4_refit = []  # 重新拟合结果2

    def load_data(self):
        """
        加载 Excel 数据文件。
        """
        imported_data = pd.read_excel(self.data_file, sheet_name=None)
        sheet_name = list(imported_data.keys())[0]  # 获取第一个工作表名称
        return imported_data[sheet_name]  # 返回第一个工作表的数据

    @staticmethod
    def add_auxiliary_lines(q_list, delta_theta_over_delta_q_list):
        """
        添加辅助线到图表中。
        """
        for i in range(len(delta_theta_over_delta_q_list)):
            plt.axvline(x=q_list[i], color='black', linestyle='dashed')
            plt.hlines(y=delta_theta_over_delta_q_list[i], xmin=q_list[i], xmax=q_list[i + 1], color='black')
            plt.axvline(x=q_list[i + 1], color='black', linestyle='dashed')

    @staticmethod
    def inifit(q_to_fit_list, delta_theta_over_delta_q_to_fit_list):
        """
        初拟合数据。
        """
        fit_data = np.column_stack((q_to_fit_list, delta_theta_over_delta_q_to_fit_list))
        model = LinearRegression()
        model.fit(fit_data[:, 0].reshape(-1, 1), fit_data[:, 1])
        return model, fit_data

    @staticmethod
    def detect_outliers(fit_data, threshold=2):
        """
        检测异常值。
        """
        z_scores = np.abs((fit_data[:, 1] - np.mean(fit_data[:, 1])) / np.std(fit_data[:, 1]))
        outliers = np.where(z_scores > threshold)[0]
        return outliers

    @staticmethod
    def refit_after_outliers_removed(fit_data, outliers):
        """
        移除异常值后重新拟合数据。
        """
        filtered_data = np.delete(fit_data, outliers, axis=0)
        X = filtered_data[:, 0].reshape(-1, 1)
        y = filtered_data[:, 1]
        model = LinearRegression()
        model.fit(X, y)
        return model, filtered_data, outliers

    def process_data(self):
        """
        处理数据并生成图表。
        """
        plot_ranges_initial = [
            {'x_min': 0, 'x_max': 0.200, 'y_min': 0, 'y_max': 140000},
            {'x_min': 0, 'x_max': 0.200, 'y_min': 0, 'y_max': 25000},
            {'x_min': 0, 'x_max': 0.200, 'y_min': 0, 'y_max': 8000}
        ]

        plot_ranges_refit = [
            {'x_min': 0, 'x_max': 0.200, 'y_min': 0, 'y_max': 15000},
            {'x_min': 0, 'x_max': 0.200, 'y_min': 0, 'y_max': 5000},
            {'x_min': 0, 'x_max': 0.200, 'y_min': 0, 'y_max': 4000}
        ]

        for i in range(3):
            selected_data = self.data.iloc[1:12, 1 + 3*i:4 + 3*i]  # 选择数据
            data_array = selected_data.values
            data_array[:, 0] = data_array[:, 0] / 100  # 将第一列转换为标准单位

            S = 0.0475  # 截面积
            deltaV = 9.446 * 10**-4  # 体积变化
            deltaQ = deltaV / S  # 流量变化

            delta_theta_list = np.diff(data_array[:, 1])  # 角度变化
            delta_q_list = np.full(len(delta_theta_list), deltaQ)  # 流量变化列表
            delta_theta_over_delta_q_list = delta_theta_list / delta_q_list  # 角度变化率

            q_list = np.linspace(0, 0 + len(delta_theta_list) * deltaQ, len(delta_theta_list) + 1)  # 流量列表
            q_to_fit_list = (q_list[:-1] + q_list[1:]) / 2  # 拟合用流量列表
            delta_theta_over_delta_q_to_fit_list = delta_theta_over_delta_q_list  # 拟合用角度变化率列表

            model, fit_data = self.inifit(q_to_fit_list, delta_theta_over_delta_q_to_fit_list)  # 初拟合

            self.q_to_fit_lists.append(q_to_fit_list)  # 保存初拟合流量列表
            self.delta_theta_over_delta_q_to_fit_lists.append(delta_theta_over_delta_q_to_fit_list)  # 保存初拟合角度变化率列表

            inifit_slope = model.coef_[0]  # 初拟合斜率
            inifit_intercept = model.intercept_  # 初拟合截距
            self.inifit_slopes.append(inifit_slope)  # 保存初拟合斜率
            self.inifit_intercepts.append(inifit_intercept)  # 保存初拟合截距

            self.ans1_inifit.append(q_to_fit_list)  # 保存初拟合结果1
            self.ans1_inifit.append(delta_theta_over_delta_q_to_fit_list)  # 保存初拟合结果1
            self.ans2_inifit.append(inifit_slope)  # 保存初拟合结果2
            self.ans2_inifit.append(inifit_intercept)  # 保存初拟合结果2

            print(f'第{i+1}组数据初拟合结果:')
            print('初拟合斜率:', inifit_slope)
            print('初拟合截距:', inifit_intercept)

            plt.figure(figsize=(8, 6))
            plt.scatter(fit_data[:, 0], fit_data[:, 1], color='red', label='拟合数据')
            plt.plot(fit_data[:, 0], model.predict(fit_data[:, 0].reshape(-1, 1)), color='blue', label='拟合线')

            center_x = np.mean(fit_data[:, 0])  # 中心 x 坐标
            center_y = np.mean(fit_data[:, 1])  # 中心 y 坐标
            equation_text = f'y = {inifit_slope:.2f} * x + {inifit_intercept:.2f}'  # 拟合方程文本
            plt.text(center_x, center_y, equation_text, color='black', fontsize=15, 
                     fontproperties='SimHei', verticalalignment='top', weight='bold')

            self.add_auxiliary_lines(q_list, delta_theta_over_delta_q_list)  # 添加辅助线
            outliers = self.detect_outliers(fit_data)  # 检测异常值
            plt.scatter(fit_data[outliers, 0], fit_data[outliers, 1], color='green', label='异常值')

            current_range_initial = plot_ranges_initial[i]  # 当前初拟合范围
            plt.xlim(current_range_initial['x_min'], current_range_initial['x_max'])
            plt.ylim(current_range_initial['y_min'], current_range_initial['y_max'])

            plt.xlabel('q 值')
            plt.ylabel('Δθ/Δq')
            plt.legend(loc='upper left')
            plt.figtext(0.5, 0.01, f'第{i+1}组数据初拟合', ha='center', fontsize=15)

            plt.gca().spines['top'].set_linewidth(2)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['right'].set_linewidth(2)
            plt.minorticks_on()

            plt.savefig(f'./拟合图结果/{2*i+1}.png')
            plt.show()

            if len(outliers) > 0:
                model, filtered_data, _ = self.refit_after_outliers_removed(fit_data, outliers)  # 重新拟合

                self.delta_theta_over_delta_q_to_refit_lists.append(filtered_data[:, 1])  # 保存重新拟合角度变化率列表
                self.q_to_refit_lists.append(filtered_data[:, 0])  # 保存重新拟合流量列表

                refit_slope = model.coef_[0]  # 重新拟合斜率
                refit_intercept = model.intercept_  # 重新拟合截距
                self.refit_slopes.append(refit_slope)  # 保存重新拟合斜率
                self.refit_intercepts.append(refit_intercept)  # 保存重新拟合截距

                self.ans3_refit.append(filtered_data[:, 0])  # 保存重新拟合结果1
                self.ans3_refit.append(filtered_data[:, 1])  # 保存重新拟合结果1
                self.ans4_refit.append(refit_slope)  # 保存重新拟合结果2
                self.ans4_refit.append(refit_intercept)  # 保存重新拟合结果2

                print(f'第{i+1}组数据排除异常值后重新拟合结果:')
                print('排除异常值后斜率:', model.coef_[0])
                print('排除异常值后截距:', model.intercept_)

                plt.figure(figsize=(8, 6))
                plt.scatter(filtered_data[:, 0], filtered_data[:, 1], color='red', label='拟合数据')
                plt.plot(filtered_data[:, 0], model.predict(filtered_data[:, 0].reshape(-1, 1)), color='blue', label='拟合线')

                center_x_refit = np.mean(filtered_data[:, 0])  # 中心 x 坐标
                center_y_refit = np.mean(filtered_data[:, 1])  # 中心 y 坐标
                equation_text_refit = f'y = {refit_slope:.2f} * x + {refit_intercept:.2f}'  # 拟合方程文本
                plt.text(center_x_refit, center_y_refit, equation_text_refit, color='black',
                         fontsize=15, fontproperties='SimHei', verticalalignment='top', weight='bold')

                self.add_auxiliary_lines(q_list, delta_theta_over_delta_q_list)  # 添加辅助线

                current_range_refit = plot_ranges_refit[i]  # 当前重新拟合范围
                plt.xlim(current_range_refit['x_min'], current_range_refit['x_max'])
                plt.ylim(current_range_refit['y_min'], current_range_refit['y_max'])

                plt.xlabel('q 值')
                plt.ylabel('Δθ/Δq')
                plt.legend(loc='upper left')

                plt.gca().spines['top'].set_linewidth(2)
                plt.gca().spines['bottom'].set_linewidth(2)
                plt.gca().spines['left'].set_linewidth(2)
                plt.gca().spines['right'].set_linewidth(2)
                plt.minorticks_on()

                plt.figtext(0.5, 0.01, f'第{i+1}组数据排除异常值后重新拟合', ha='center', fontsize=15)
                plt.savefig(f'./拟合图结果/{2*i+2}.png')
                plt.show()

        self.ans1_inifit = np.array(self.ans1_inifit).T  # 转换初拟合结果1为数组
        self.ans2_inifit = np.array(self.ans2_inifit).reshape(3, 2)  # 转换初拟合结果2为数组
        self.ans3_refit = np.array(self.ans3_refit).T  # 转换重新拟合结果1为数组
        self.ans4_refit = np.array(self.ans4_refit).reshape(3, 2)  # 转换重新拟合结果2为数组

        plt.figure(figsize=(8, 6))

        for i in range(3):
            plt.scatter(self.q_to_fit_lists[i], self.delta_theta_over_delta_q_to_fit_lists[i], label=f'第{i+1}组数据')
            plt.plot(self.q_to_fit_lists[i], self.inifit_slopes[i] * self.q_to_fit_lists[i] + self.inifit_intercepts[i], label=f'拟合线{i+1}')
            self.add_auxiliary_lines(q_list, self.delta_theta_over_delta_q_to_fit_lists[i])

        plt.xlim(0, current_range_refit['x_max'])
        plt.xlabel('q 值')
        plt.ylabel('Δθ/Δq')
        plt.legend(loc='upper left')
        plt.figtext(0.5, 0.01, '三组数据保留所有数据点初拟合对比', ha='center', fontsize=15)

        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.minorticks_on()

        plt.savefig('./拟合图结果/7.png')
        plt.show()

        plt.figure(figsize=(8, 6))

        for i in range(3):
            plt.scatter(self.q_to_refit_lists[i], self.delta_theta_over_delta_q_to_refit_lists[i], label=f'第{i+1}组数据')
            plt.plot(self.q_to_refit_lists[i], self.refit_slopes[i] * self.q_to_refit_lists[i] + self.refit_intercepts[i], label=f'拟合线{i+1}')
            self.add_auxiliary_lines(q_list, self.delta_theta_over_delta_q_to_refit_lists[i])

        plt.xlim(0, 0.200)
        plt.xlabel('q 值')
        plt.ylabel('Δθ/Δq')
        plt.legend(loc='upper left')
        plt.figtext(0.5, 0.01, '三组数据排除异常值后再拟合对比', ha='center', fontsize=15)

        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.minorticks_on()

        plt.savefig('./拟合图结果/8.png')
        plt.show()

        images = []
        for i in range(1, 9):
            img = mpimg.imread(f'./拟合图结果/{i}.png')
            images.append(img)

        fig, axes = plt.subplots(4, 2, figsize=(10, 12))

        for ax, img in zip(axes.flatten(), images):
            ax.imshow(img)
            ax.axis('off')

        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.savefig(r'./拟合图结果/拟合图整合图.png', bbox_inches='tight')
        plt.show()

        dir_to_zip = r'./拟合图结果'
        dir_to_save = r'./拟合图结果.zip'

        with zipfile.ZipFile(dir_to_save, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(dir_to_zip):
                for file in files:
                    file_dir = os.path.join(root, file)
                    arc_name = os.path.relpath(file_dir, dir_to_zip)
                    zipf.write(file_dir, arc_name)

        print(f'压缩完成，文件保存为: {dir_to_save}')
        
# 使用示例
if __name__ == '__main__':
    file_path = r'./过滤原始数据记录表(非).xlsx'
    filter_data_processor = FilterDataProcessor(file_path)
    filter_data_processor.process_data()
