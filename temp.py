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
        
    def data_summary(self):
        print("-------------------------------")
        print("         Load Data \n")
        print(F"[*] Valid Column number : {self.count_columns}")
        print(F"[*] Valud Row    number : {self.count_rows}")
        print("-------------------------------")
        
    def check(self):
        #self.test()
        self.entropy_best()
        pass
    
    def entropy_best(self):
        # 수치형 데이터만 가능하게 구현
        temp = []
        for column_number in range(0,self.count_columns):
            for division_point in np.unique(self.df.iloc[:,column_number]):
                temp.append([self.information_gain(column_number, division_point), column_number, division_point])
                
        ppp = None
        min_value = 0
        for i in temp:
            if i[0] >= min_value:
                min_value = i[0]
                ppp = i
        print(ppp)

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
        #print(F"parent :{sum}")
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
        P = value/total
        if P == 0:
            return 0
        return -1 * P * np.log2(P)
    
    def test(self):
        column_number = 3
        division = 0.8
        print(len(self.df[(self.df.iloc[:, column_number] <= division)]))
        print(len(self.df[~(self.df.iloc[:, column_number] <= division)]))
        print(self.df[(self.df.iloc[:, column_number] <= division)])


def main():
    iris = load_iris()
    a = DectisionTree(data = iris.data, label = iris.target)


if __name__ == "__main__":
    print("temp")
    main()