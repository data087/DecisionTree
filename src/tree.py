"""

tree.py

DecisionTree 구현

1. label이 하나만 나올 때 까지 재귀적으로 도는 것이 목표.
2. 재귀적으로 돌때는 노드를 생성해서 도는걸로.
3. 이때 데이터는 어떻게 분기를 잡아야 할까? 재귀를 도는 것 마냥 구현해도 되나?

"""
import pandas as pd
import numpy as np
import src.node

class DecisionTree:
    def __init__(self, data, label):
        # 데이터 관련
        #self.df = pd.DataFrame(data=data)
        #self.df['label'] = label
        #self.label_unique = np.unique(self.df['label'])

        # 실제 동작
        self.root = src.node.Node(data, label)
        self.get_node()

    def get_node(self):
        temp = self.create_pretty_str(self.root)
        print(temp)

        #print(self.root.node_entropy)
        #print(self.root.division_column)
        #print(self.root.division_point)

    def create_pretty_str(self, node):
        temp = F"Entrpy : {node.node_entropy}, column = {node.division_column} \
        division_point = {node.division_column_point}"
        return temp
