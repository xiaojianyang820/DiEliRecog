import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from FANS import FANS
from sklearn.metrics import roc_curve, auc
from typing import Tuple


def PLR(train_data: np.ndarray, train_label: np.ndarray, test_data: np.ndarray,
        cs: np.ndarray = np.arange(0.1, 1, 0.05), seed: int = 100) -> np.ndarray:
    """
    PLR算法函数

    :param train_data: np.ndarray,
        训练数据的特征矩阵
    :param train_label: np.ndarray,
        训练数据的标签向量
    :param test_data: np.ndarray,
        测试数据的特征矩阵
    :param cs: np.ndarray,
        交叉验证中使用的一范数惩罚系数C的候选值集合
    :param seed: int,
        交叉验证过程中使用的随机种子
    :return: np.ndarray,
        PLR算法给出的概率性预测向量（标签为1的概率）
    """
    whole_data = np.vstack((train_data, test_data))
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(whole_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    plr_classifier = LogisticRegressionCV(penalty='l1', Cs=cs, cv=3, solver='liblinear', random_state=seed)
    plr_classifier.fit(train_data, train_label)
    pro_predict = plr_classifier.predict_proba(test_data)[:, 1]
    return pro_predict


class CoTraining(object):
    """
    双空间分歧协同训练识别算法
    """
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, iteration: int = 10,
                 fans_cs: np.ndarray = np.arange(0.1, 1, 0.05), plr_cs: np.ndarray = np.arange(0.1, 0.5, 0.05),
                 times: float = 0.99, threshold_stop: float = 0.01, seed: int = 100):
        """
        初始化方法

        :param x_train: np.ndarray,
            有标签组的设计矩阵
        :param y_train: np.ndarray,
            有标签组的标签列
        :param x_test: np.ndarray,
            无标签组的设计矩阵
        :param iteration: int, optional=10
            总迭代次数
        :param fans_cs: np.ndarray, optional
            FANS算法的正则系数交叉验证候选集合
        :param plr_cs: np.ndarray, optional
            PLR算法的正则系数交叉验证候选集合
        :param times: float, optional=0.99
            扩张系数
        :param threshold_stop: float, optional=0.01
            算法停止运行的边界条件，即新增高置信样本数量与已有高置信组样本数量比例低于这一个边界系数后，停止计算
        :param seed: int, optional=100
            全局的随机种子
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.fans_train_set_x = self.x_train
        self.fans_train_set_y = self.y_train
        self.plr_train_set_x = self.x_train
        self.plr_train_set_y = self.y_train
        self.state_dict = self.making_dict()
        self.iteration = iteration
        self.fans_high_confidence = []
        self.plr_high_confidence = []
        self.fans_cs = fans_cs
        self.plr_cs = plr_cs
        self.times = times
        self.evaluation = {}
        self.current_fans_train_num = len(X_train)
        self.current_plr_train_num = len(X_train)
        self.seed = seed
        self.threshold_stop = threshold_stop

    def start(self, istest: bool = False, y_test: np.ndarray = None) -> np.ndarray:
        """
        双空间分歧协同训练识别算法的主过程

        :param istest: bool,
            是否为测试模式
        :param y_test: np.ndarray,
            用于计算评价指标的无标签组标签列
        :return: np.ndarray,
            测试数据的概率性预测结果（标签为1的概率）
        """
        indicators = ['total_accu_ratio', 'total_auc', 'hcg_num', 'hcg_accu_ratio', 'fans_hcg_num', 'plr_hcg_num']
        for item in indicators:
            self.evaluation[item] = []
        for item in range(self.iteration):
            print(f'第{item}次循环开始计算')
            # 构造FANS算法，并对FANS算法进行训练
            fans = FANS(self.fans_train_set_x, self.fans_train_set_y, self.x_test, cs=self.fans_cs, seed=self.seed)
            fans_pro_predict = fans.main()
            # 基于FANS算法所给出的结果，更新PLR算法的训练集和高置信组
            self.fans_select(fans_pro_predict)
            if istest:
                _, _ = self.evaluate(y_test, fans_pro_predict, 'FANS')

            # 计算出此时plr算法训练集的规模
            plr_train_num = self.plr_train_set_x.shape[0]
            # 如果plr算法训练集增长比例不够条件，则停止训练
            if (plr_train_num - self.current_plr_train_num) / (self.current_plr_train_num + 1) < self.threshold_stop:
                break
            else:
                print('本轮更新后，PLR算法增加了%d个训练样本' % (plr_train_num - self.current_plr_train_num))
                self.current_plr_train_num = plr_train_num

            # 构造PLR算法，并对PLR算法进行训练
            plr_pro_predict = PLR(self.plr_train_set_x, self.plr_train_set_y, self.x_test, cs=self.plr_cs,
                                  seed=self.seed)
            # 基于PLR算法所给出的结果，更新FANS算法的训练集和高置信组
            self.plr_select(plr_pro_predict)
            if istest:
                _, _ = self.evaluate(y_test, plr_pro_predict, 'PLR')

            # 计算出此时fans算法训练集的规模
            fans_train_num = self.fans_train_set_x.shape[0]
            # 如果plr算法训练集增长比例不够条件，则停止训练
            if (fans_train_num - self.current_fans_train_num) / (self.current_fans_train_num + 1) < self.threshold_stop:
                break
            else:
                print('本轮更新后，FANS算法增加了%d个训练样本' % (fans_train_num - self.current_fans_train_num))
                self.current_fans_train_num = fans_train_num

        predict = np.vstack((fans_pro_predict, plr_pro_predict))
        predict = np.mean(predict, axis=0)
        # 将高置信样本集合的概率性预测调整为1或者0
        for i, item in enumerate(self.state_dict[:, 0]):
            if item:
                label = self.state_dict[i, 1]
                if label == 1:
                    label = 1.0
                elif label == 0:
                    label = 0.0
                predict[i] = label

        return predict

    def making_dict(self) -> np.ndarray:
        """
        初始化无标签组的状态字典state_dict
        第一个位置：是否进入高置信组
        第二个位置：相应的伪标签
        第三个位置：在历次循环中算法给的标签
        第四个位置：第一次进入高置信组时的置信度
        :return: np.ndarray,
            初始化好的无标签组状态字典
        """
        state_dict = []
        for item in range(self.x_test.shape[0]):
            state_dict.append([0, -1, [], -1])
        return np.array(state_dict, dtype=object)

    def select_sample(self, pro_predict: np.ndarray, num: int) -> np.ndarray:
        """
        基于概率性预测向量计算出高置信样本集合索引

        :param pro_predict: np.ndarray,
            概率性预测向量
        :param num: int,
            高置信样本数量
        :return: np.ndarray,
            高置信样本集合索引
        """
        pro_predict = np.abs(pro_predict - 0.5)
        arged_index = np.argsort(pro_predict)[-int(num * self.times):]
        certain_index = self.x_test.shape[0] * [False]
        for item in arged_index:
            certain_index[item] = True
        return np.array(certain_index)

    def fans_select(self, fans_pro_predict: np.ndarray) -> None:
        """
        基于FANS算法对无标签集合的概率性预测向量进行PLR算法训练集扩充

        :param fans_pro_predict: np.ndarray,
            FANS算法对无标签集合的概率性预测向量
        :return: None
        """
        # 选择出置信度最高的特定数量的无标签组索引
        certain_index = self.select_sample(fans_pro_predict, self.plr_train_set_x.shape[0]*1.0)
        # 使用这一组高置信组索引去更新无标签组状态字典
        for i, item in enumerate(certain_index):
            if item:
                self.state_dict[i][2].append(fans_pro_predict[i])
                if self.state_dict[i][0] == 1:
                    continue
                else:
                    self.state_dict[i][0] = 1
                    self.state_dict[i][1] = self.pro_2_int(np.array([fans_pro_predict[i]]))[0]
                    self.state_dict[i][3] = fans_pro_predict[i]
        # 如果这次统计的高置信组并不在plr算法的已有高置信组中，那将其添加进去
        sub_certain_index = []
        for i, item in enumerate(certain_index):
            if item and (i not in self.plr_high_confidence):
                sub_certain_index.append(True)
                self.plr_high_confidence.append(i)
            else:
                sub_certain_index.append(False)
        # 本次需要新增的高置信组索引
        certain_index = np.array(sub_certain_index)
        # 这一部分新增高置信组索引的特征和标签
        psudo_x = self.x_test[certain_index]
        psudo_y = self.pro_2_int(fans_pro_predict[certain_index])
        # 将这一部分高置信组样本合并到plr算法的已有训练样本中
        self.plr_train_set_x = np.vstack((self.plr_train_set_x, psudo_x))
        self.plr_train_set_y = np.hstack((self.plr_train_set_y, psudo_y))

    def plr_select(self, plr_pro_predict: np.ndarray) -> None:
        """
        基于PLR算法对无标签集合的概率性预测向量进行FANS算法训练集扩充

        :param plr_pro_predict: np.ndarray,
            FANS算法对无标签集合的概率性预测向量
        :return:
        """
        certain_index = self.select_sample(plr_pro_predict, self.fans_train_set_x.shape[0]*0.95)
        for i, item in enumerate(certain_index):
            if item:
                self.state_dict[i][2].append(plr_pro_predict[i])
                if self.state_dict[i][0] == 1:
                    continue
                else:
                    self.state_dict[i][0] = 1
                    self.state_dict[i][1] = self.pro_2_int(np.array([plr_pro_predict[i]]))[0]
                    self.state_dict[i][3] = plr_pro_predict[i]
        sub_certain_index = []
        for i, item in enumerate(certain_index):
            if item and (i not in self.fans_high_confidence):
                sub_certain_index.append(True)
                self.fans_high_confidence.append(i)
            else:
                sub_certain_index.append(False)
        certain_index = np.array(sub_certain_index)
        psudo_x = self.x_test[certain_index]
        psudo_y = self.pro_2_int(plr_pro_predict[certain_index])
        self.fans_train_set_x = np.vstack((self.fans_train_set_x, psudo_x))
        self.fans_train_set_y = np.hstack((self.fans_train_set_y, psudo_y))

    @classmethod
    def pro_2_int(cls, pro: np.ndarray) -> np.ndarray:
        """
        将概率性预测向量转换为0-1预测向量

        :param pro: np.ndarray,
             概率性预测向量
        :return: np.ndarray,
            0-1预测向量
        """
        data = []
        for item in pro:
            if item > 0.5:
                data.append(1)
            else:
                data.append(0)
        return np.array(data)

    def evaluate(self, y_test: np.ndarray, pro_predict: np.ndarray, method: str) -> Tuple[float, float]:
        """
        对某一个模型的预测结果进行评估

        :param y_test: np.ndarray,
            测试集的真实标签向量
        :param pro_predict: np.ndarray,
            测试集的概率性预测向量
        :param method: str,
            测试的具体模型名称
        :return: Tuple[float, float],
            分类结果的AUC值和准确率
        """
        for i, item in enumerate(self.state_dict[:, 0]):
            if item:
                label = self.state_dict[i, 3]
                pro_predict[i] = label
        int_pro_predict = self.pro_2_int(pro_predict)
        print(f'{method}: 准确率为 {np.mean(int_pro_predict == y_test) * 100: .2f} %')

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pro_predict)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        accu = 0.0
        total = 0.0
        for i, item in enumerate(y_test):
            if self.state_dict[i][0] == 1:
                total += 1
                if item == self.state_dict[i][1]:
                    accu += 1
        print(f'{method}:测试总体数据的AUC得分为{roc_auc}')
        print(f'高置信组的规模为{total}')
        print(f'高置信组的准确率为{accu / total}')

        self.evaluation['total_accu_ratio'].append(np.mean(int_pro_predict == y_test))
        self.evaluation['total_auc'].append(roc_auc)
        self.evaluation['hcg_num'].append(total)
        self.evaluation['hcg_accu_ratio'].append(accu / total)
        self.evaluation['fans_hcg_num'].append(len(self.fans_high_confidence))
        self.evaluation['plr_hcg_num'].append(len(self.plr_high_confidence))
        return roc_auc, accu / total


if __name__ == '__main__':
    import random
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    # 网络信贷平台复购数据集
    data_1 = np.loadtxt('data/loan_rebuy/user_info.csv', delimiter=',')
    data_2 = np.loadtxt('data/loan_rebuy/consump.csv')
    data_3 = np.loadtxt('data/loan_rebuy/relation1_score.txt')
    data_4 = np.loadtxt('data/loan_rebuy/relation2_score.txt')
    data_5 = np.loadtxt('data/loan_rebuy/tags_score.txt')
    train = pd.read_csv('data/loan_rebuy/train.txt')
    X = np.hstack((data_1, data_2, data_3, data_4, data_5))
    Y = np.array(train['lable'])
    with_label_sample_num = 200
    without_label_sample_num = 25000
    PLR_CS = np.arange(0.1, 0.5, 0.05)
    FANS_CS = np.arange(0.1, 1, 0.05)
    times = 0.99
    iteration = 50
    seed = 503
    thred_ratio = 0.01

    '''
    # 垃圾邮件分类数据集
    #fans_cs: np.ndarray = np.arange(0.1, 10, 0.1)
    data = pd.read_csv('data/spam_email/spambase.csv', header=None)
    X = data.iloc[:, :-1].values[:-950]
    Y = data.iloc[:, -1].values[:-950]
    with_label_sample_num = 20
    without_label_sample_num = 3400
    PLR_CS = np.arange(0.1, 0.5, 0.05)
    FANS_CS = np.arange(0.1, 10, 0.05)
    times = 0.99
    iteration = 100
    seed = 501
    thred_ratio = 0.01
    '''
    np.random.seed(seed)
    random.seed(seed)
    population = set(range(X.shape[0]))
    train_index = np.array(random.sample(population, with_label_sample_num))
    population -= set(train_index)
    test_index = np.array(random.sample(population, without_label_sample_num))
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    print(times, seed, thred_ratio, PLR_CS)
    # %%
    cotraining = CoTraining(X_train, Y_train, X_test, times=times, iteration=iteration, plr_cs=PLR_CS, seed=seed,
                            threshold_stop=thred_ratio, fans_cs=FANS_CS)
    co_predict = cotraining.start(istest=True, y_test=Y_test)
    fans = FANS(X_train, Y_train, X_test, seed=seed, cs=FANS_CS)
    fans_predict = fans.main()
    plr_predict = PLR(X_train, Y_train, X_test, cs=PLR_CS, seed=seed)

    def plot_func(pro_predict: np.ndarray, y_test: np.ndarray, label: str, ls: str, color: str):
        confidence = np.abs(pro_predict-0.5)
        confidence_sort_index = np.argsort(confidence)[::-1]
        sorted_pro_predict = pro_predict[confidence_sort_index]
        sorted_y_test = y_test[confidence_sort_index]

        accu_list = []
        for index in range(40, len(pro_predict), 40):
            accu = np.mean([int(round(j) == i) for i, j in zip(sorted_y_test[:index], sorted_pro_predict[:index])])
            accu_list.append(accu)
        plt.plot(range(40, len(pro_predict), 40), accu_list, ls=ls, label=label, color=color, alpha=0.8)

    hcg_num = sum(cotraining.state_dict[:, 0])
    figure = plt.figure(figsize=(12, 8))
    ax = figure.add_subplot(111)
    plot_func(co_predict, Y_test, label='CoTraining', ls='-', color='red')
    plot_func(fans_predict, Y_test, label='FANS', ls='--', color='navy')
    plot_func(plr_predict, Y_test, label='PLR', ls='--', color='green')
    plt.legend(loc='upper right', fontsize=13)
    plt.xlabel('Examples Sorted by Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.vlines(x=hcg_num, ymax=0.99, ymin=0.70, colors='k', lw=1.5, alpha=0.6, linestyles='--')
    figure.savefig("1.png", dpi=300)

    auc_co = roc_auc_score(Y_test, co_predict)
    auc_fans = roc_auc_score(Y_test, fans_predict)
    auc_plr = roc_auc_score(Y_test, plr_predict)
    print()
    print(f'CoTraining: {auc_co: .3f}\nFANS      : {auc_fans: .3f}\nPLR       : {auc_plr: .3f}')

