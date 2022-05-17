import numpy as np
import pandas as pd
from scipy import stats
from sklearn import cross_validation
import random
from sklearn.linear_model import LogisticRegressionCV


class FANS(object):
    """
        特征扩充与选择算法（Feature Argument and Select, FANS）的实现
    """
    def __init__(self, train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray,
                 bw_method: str = 'silverman', iteration: int = 4, cs: np.ndarray = np.arange(0.1, 10, 0.1),
                 threshold: float = 0.005, seed: int = 100):
        """
        初始化方法

        :param train_data: np.ndarray,
            有标签数据的设计矩阵
        :param train_labels: np.ndarray,
            有标签数据的类别标签
        :param test_data: np.ndarray,
            无标签数据的设计矩阵
        :param bw_method: str, optional=silverman
            高斯核密度估计选择窗宽所依据的准则，可选值有两个，scott，silverman
        :param iteration: int, optional=4
            对有标签组进行多少次切分来进行投票表决，默认值是4
        :param cs: np.ndarray, optional=np.arange(0.1, 10, 0.1)
            L1惩罚系数的测试范围
        :param threshold: float, optional=0.005
            稳定性阈值，当除数小于这一阈值时，就截断到这个阈值，以防止出现数值异常问题
        :param seed: int, optional=100
            随机数种子
        """
        self.train_data = np.array(train_data)
        self.train_label = np.array(train_labels)
        self.test_data = np.array(test_data)
        self.train_num = self.train_data.shape[0]
        self.test_num = self.test_data.shape[0]
        self.feature_num = self.train_data.shape[1]
        self.bw_method = bw_method
        self.iteration = iteration
        self.cs = cs
        self.threshold = threshold
        self.seed = seed

        try:
            assert self.train_data.shape[1] == self.test_data.shape[1]
        except Exception as e:
            print('有标签组与无标签组的特征数量不一致')
            raise e

    def main(self) -> np.ndarray:
        """
        FANS算法的主流程

        :return: np.ndarray,
            FANS算法对测试集部分给出的概率性预测结果向量（类别标签为1的概率）
        """
        voted_pro_predict = []
        for t in range(self.iteration):
            print('.', end='')
            # 将训练数据拆分为相等的两部分，
            # 分别用于KDE密度估计，单一指标类别密度特征转换以及Penalized Logistic Regression（PLR）学习
            #kfold = cross_validation.KFold(self.train_num, n_folds=2, shuffle=True, random_state=self.seed+t)
            kfold = cross_validation.StratifiedKFold(self.train_label, n_folds=2, shuffle=True, random_state=self.seed+t)
            for kde_part, plr_part in kfold:
                # 拆分出KDE部分和PLR部分
                x_kde, y_kde = self.train_data[kde_part], self.train_label[kde_part]
                x_plr_train, y_plr_train = self.train_data[plr_part], self.train_label[plr_part]
                # 基于KDE数据进行密度估计，然后对PLR数据进行密度比转换
                transformed_x_plr_train = self.con_transformer(x_kde, y_kde, x_plr_train)
                transformed_x_plr_test = self.con_transformer(x_kde, y_kde, self.test_data)
                # 构建PLR模型，并进行训练和预测
                plr_classifier = LogisticRegressionCV(penalty='l1', Cs=self.cs, cv=3, solver='liblinear',
                                                      random_state=self.seed)
                plr_classifier.fit(transformed_x_plr_train, y_plr_train)
                plr_pro_predict = plr_classifier.predict_proba(transformed_x_plr_test)[:, 1]
                voted_pro_predict.append(plr_pro_predict)
        # 将多次预测的结果进行合并并平均
        voted_pro_predict = np.array(voted_pro_predict, dtype=float)
        final_pro_predict = np.mean(voted_pro_predict, axis=0)
        return final_pro_predict

    def con_transformer(self, x_kde: np.ndarray, y_kde: np.ndarray, x_plr: np.ndarray) -> np.ndarray:
        """
        完成分类别kde估计以及数值特征向密度比特征的转换

        :param x_kde: np.ndarray,
            用于KDE估计的数据的特征矩阵
        :param y_kde: np.ndarray,
            用于KDE估计的数据的标签向量
        :param x_plr: np.ndarray,
            用于模型估计的数据的特征矩阵
        :return: np.ndarray,
            密度比转换之后得到的特征矩阵
        """
        transformed_data = []
        x_plr = x_plr.T
        kde_generator = self.con_kde_generator(x_kde, y_kde)
        for item in x_plr:
            item_set = set(item)
            pro_dict = {}
            kde_0, kde_1 = next(kde_generator)
            for i in item_set:
                pro_0, pro_1 = kde_0(i), kde_1(i)
                pro_0 = max(pro_0[0], self.threshold)
                pro_1 = max(pro_1[0], self.threshold)
                pro_dict[i] = np.log(pro_1 / pro_0)
            map_pro = lambda value: pro_dict[value]
            item = np.array(list(map(map_pro, item)))
            transformed_data.append(item)
        return np.array(transformed_data).T

    def con_kde_generator(self, xs: np.ndarray, ys: np.ndarray):
        """
        协程：核密度估计器生成器

        :param xs: np.ndarray,
            设计矩阵
        :param ys: np.ndarray,
            标签列
        :return:
        """
        for item in range(self.feature_num):
            s_0 = xs[ys == 0][:, item]
            s_1 = xs[ys == 1][:, item]
            kde_0 = self.kde(s_0)
            kde_1 = self.kde(s_1)
            yield kde_0, kde_1

    def kde(self, x: np.ndarray) -> stats.gaussian_kde:
        """
        核密度估计

        :param x: np.ndarray,
            一维数组，KDE方法将估计出这一列数据上的非参数分布
        :return: stats.gaussian_kde,
            基于数据构造出来的密度估计器
        """
        # 如果这一列数据上没有波动的话，高斯核密度估计会报错，
        # 这种情况下需要增加一定比例的随机波动，以便构造出密度估计器
        try:
            return stats.gaussian_kde(x, bw_method=self.bw_method)
        except Exception as e:
            x += np.random.randn(x.shape[0]) * 0.1
            #x = x + np.sin(np.arange(x.shape[0])) * 0.1
            return stats.gaussian_kde(x, bw_method=self.bw_method)


if __name__ == '__main__':
    '''
    print('FANS算法在网络借贷平台复购数据集上测试')
    data_1 = np.loadtxt('data/loan_rebuy/user_info.csv', delimiter=',')
    data_2 = np.loadtxt('data/loan_rebuy/consump.csv')
    data_3 = np.loadtxt('data/loan_rebuy/relation1_score.txt')
    data_4 = np.loadtxt('data/loan_rebuy/relation2_score.txt')
    data_5 = np.loadtxt('data/loan_rebuy/tags_score.txt')
    train = pd.read_csv('data/loan_rebuy/train.txt')
    X = np.hstack((data_1, data_2, data_3, data_4, data_5))
    Y = np.array(train['lable'])
    population = set(range(X.shape[0]))
    train_index = np.array(random.sample(population, 500))
    population -= set(train_index)
    test_index = np.array(random.sample(population, 20000))
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    fans_model = FANS(train_data=X_train, train_labels=Y_train, test_data=X_test)
    Y_test_pred = fans_model.main()
    Y_test_pred = [1 if i > 0.5 else 0 for i in Y_test_pred]
    accu = np.mean([int(i == j) for i, j in zip(Y_test_pred, Y_test)])

    print(f'\n训练集为500个样本点，分类准确率为：{accu*100:.2f}%')
    '''
    '''
    print('FANS算法在DOTA2胜负判定上测试')
    data = pd.read_csv('data/dota2/dota2Train.csv', header=None)
    X = data.iloc[:, 1:]
    Y = data.iloc[:, 0]
    Y = Y.replace(-1, 0)
    X = X.values.astype(np.float32)
    Y = Y.values
    population = set(range(X.shape[0]))
    train_sample_num = 1000
    test_sample_num = 20000
    train_index = np.array(random.sample(population, train_sample_num))
    population -= set(train_index)
    test_index = np.array(random.sample(population, test_sample_num))
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    fans_model = FANS(train_data=X_train, train_labels=Y_train, test_data=X_test)
    Y_test_pred = fans_model.main()
    Y_test_pred = [1 if i > 0.5 else 0 for i in Y_test_pred]
    accu = np.mean([int(i == j) for i, j in zip(Y_test_pred, Y_test)])

    print(f'\n训练集为{train_sample_num}个样本点，分类准确率为：{accu * 100:.2f}%')
    '''
    print('FANS算法在垃圾邮件检测上测试')
    data = pd.read_csv('data/spam_email/spambase.csv', header=None)
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    population = set(range(X.shape[0]))
    train_sample_num = 30
    test_sample_num = 3500
    train_index = np.array(random.sample(population, train_sample_num))
    population -= set(train_index)
    test_index = np.array(random.sample(population, test_sample_num))
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    fans_model = FANS(train_data=X_train, train_labels=Y_train, test_data=X_test)
    Y_test_pred = fans_model.main()
    Y_test_pred = [1 if i > 0.5 else 0 for i in Y_test_pred]
    accu = np.mean([int(i == j) for i, j in zip(Y_test_pred, Y_test)])

    print(f'\n训练集为{train_sample_num}个样本点，分类准确率为：{accu * 100:.2f}%')

