from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
import pandas as pd
import numpy as np


class DectisionTree:
    def __init__(self, data, label):
        # 데이터 관련
        self.df = pd.DataFrame(data = data)
        self.count_rows = len(self.df)
        self.count_columns = len(self.df.columns)
        # 라벨 관련
        self.df['label'] = label
        self.label_unique = np.unique(self.df['label'])

        self.data_summary()
        self.check()

    def find_best_IG(self):
        # 수치형 데이터만 가능하게 구현
        list_expectation = []
        for column_number in range(0, self.count_columns):
            for division_point in np.unique(self.df.iloc[:, column_number]):
                list_expectation.append([self.getInformationGain(column_number, division_point), column_number, division_point])
        return self.isMaxInformationGain(list_expectation)

    @staticmethod
    def isMaxInformationGain(list_expectation):
        max_value = None
        temp = 0
        for i in list_expectation:
            if i[0] >= temp:
                temp = i[0]
                max_value = i
        return max_value

    def getInformationGain(self, column_number, division_point):
        # 부모 엔트로피 구하기
        parent = self.getParentEntropy()

        # 서브셋 엔트로피 구하기
        children = self.getChildrenEntropy(column_number, division_point)
        return parent - children

    def getParentEntropy(self):
        sum = 0
        for label in self.label_unique:
            value = len(self.df[self.df['label'] == label])
            total = len(self.df)
            sum = sum + self.getIndividualEntropy(total, value)
        return sum

    def getChildrenEntropy(self, column_number, division):
        entropy_id = []
        for label in self.label_unique:
            value = len(self.df[(self.df.iloc[:, column_number] <= division) & (self.df['label'] == label)])
            total = len(self.df[(self.df.iloc[:, column_number] <= division)])
            if total == 0:
                continue
            entropy_id.append(self.getIndividualEntropy(total, value))

        for label in self.label_unique:
            value = len(self.df[~(self.df.iloc[:, column_number] <= division) & (self.df['label'] == label)])
            total = len(self.df[~(self.df.iloc[:, column_number] <= division)])
            if total == 0:
                continue
            entropy_id.append(self.getIndividualEntropy(total, value))
        return sum(entropy_id)

    def getIndividualEntropy(self, total, value):
        probability = value / total
        if probability == 0:
            return 0
        return -1 * probability * np.log2(probability)

