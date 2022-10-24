"""

tree.py

DecisionTree 구현

1. label이 하나만 나올 때 까지 재귀적으로 도는 것이 목표.
2. 재귀적으로 돌때는 노드를 생성해서 도는걸로.
3. 이때 데이터는 어떻게 분기를 잡아야 할까? 재귀를 도는 것 마냥 구현해도 되나?

"""
import pandas as pd
import numpy as np


class DecisionTree:
    def __init__(self, data, label):
        # 데이터 관련
        self.df = pd.DataFrame(data=data)
        self.count_rows = len(self.df)
        self.count_columns = len(self.df.columns)
        self.df['label'] = label
        self.label_unique = np.unique(self.df['label'])
        # 실제 동작
        self.check()

    def check(self):
        temp = self.entropy_best()
        print(temp)

    def entropy_best(self):
        # 수치형 데이터만 가능하게 구현
        list_expectation = []
        for column_number in range(0, self.count_columns):
            for division_point in np.unique(self.df.iloc[:, column_number]):
                list_expectation.append([self.information_gain(column_number, division_point), column_number, division_point])
        return self.find_max(list_expectation)

    @staticmethod
    def find_max(list_expectation):
        max_value = None
        temp = 0
        for i in list_expectation:
            if i[0] >= temp:
                temp = i[0]
                max_value = i
        return max_value

    def information_gain(self, column_number, division_point):
        # 부모 엔트로피 구하기
        parent = self.entropy_parent()
        # 서브셋 엔트로피 구하기
        children = self.entropy_children(column_number, division_point)
        return parent - children

    def entropy_parent(self):
        sum = 0
        for label in self.label_unique:
            value = len(self.df[self.df['label'] == label])
            total = len(self.df)
            sum = sum + self.entropy_individual(total, value)
        # print(F"parent :{sum}")
        return sum

    def entropy_children(self, column_number, division):
        entropy_id = []
        for label in self.label_unique:
            value = len(self.df[(self.df.iloc[:, column_number] <= division) & (self.df['label'] == label)])
            total = len(self.df[(self.df.iloc[:, column_number] <= division)])
            if total == 0:
                continue
            entropy_id.append(self.entropy_individual(total, value))

        for label in self.label_unique:
            value = len(self.df[~(self.df.iloc[:, column_number] <= division) & (self.df['label'] == label)])
            total = len(self.df[~(self.df.iloc[:, column_number] <= division)])
            if total == 0:
                continue
            entropy_id.append(self.entropy_individual(total, value))
        return sum(entropy_id)

    def entropy_individual(self, total, value):
        P = value / total
        if P == 0:
            return 0
        return -1 * P * np.log2(P)
