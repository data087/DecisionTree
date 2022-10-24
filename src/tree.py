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
        # 실제 동작
        root = src.node.Node(data, label, 0)
        self.get_node(root)

    def get_node(self, node):
        print(self.create_pretty_str(node))
        if node.children_left is not None:
            self.get_node(node.children_left)
        if node.children_right is not None:
            self.get_node(node.children_right)

    def create_pretty_str(self, node):
        temp = F"{node.depth * '-'}Entrpy : {node.node_entropy}, column = {node.division_column} \
        division_point = {node.division_column_point}"
        return temp
