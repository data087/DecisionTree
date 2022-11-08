"""

모듈에 대한 설명
어떻게 작성을 해야할까.

Args:
    asdfasdf
    asdfasdf
    asfdasdf

Return:
    asdfasdf
    asfdasdf

Usage:
    asdasd
    asdasd

Note:
    1. DecisionTree 클래스는 해당 데이터를 입력받는다.
    2. 노드의 Entropy의 값을 계산한다.
    3. 자식노드의 Entropy의 값을 계산한다.
    4. Information Gain값을 구하여 가장 큰 값을 도출한다.
       이때, 가장 큰 IG의 값이 중복된다면 가장 마지막 값을 사용한다.
    5. 가장 큰 IG의 값을 분기로 자식노드를 생성한다.
    6. 이 과정을 자식노드의 Label의 Unique값이 하나일 떄 까지 반복한다.



"""
import pandas as pd
import numpy as np
import src.node


class DecisionTree:
    def __init__(self, data, label):
        # 데이터 관련
        self.data = pd.DataFrame(data)
        self.data['label'] = label
        self.max_rows = None
        self.columns = None
        self.data_preprocessing()
        self.test_information()

    def data_preprocessing(self):
        self.columns = self.data.columns
        self.max_rows = len(self.data)

        print(self.columns)
        print(self.max_rows)
        data_index = [1,2,3,4,5]
        print(self.data.loc[data_index]["label"].unique())

    def build_decision_tree(self, data_index = None):
        data = self.data.iloc[data_index]
        if len(data["label"].unique()) < 2:
            return None

        parent_node = src.node.TreeNode()

        """
        1. 데이터의 인덱스 값을 받음. [+]
        2. data = df[data_index] [+]
        3. 가장 적절한 Information Gain 값 도출 [-]
        """

        parent_node.left = self.build_decision_tree()
        parent_node.right = self.build_decision_tree()

        return parent_node
    def calculate_max_information(self, data):
        """
        1. 데이터를 받음.
        2. 부모 엔트로피 계산
        3. 자식 엔트로피 계산
        4. 최대치가 되는 엔트로피 반환.
        :param
            data: 부모 node의 데이터

        :return:
            children1 : data_index
            children2 : data_index
        """
        # return data_index 두개
        pass

    def calculate_information(self):
        """


        :return:
        """
        pass

    def calculate_entropy(self, data):
        pass

    def select_candiate(self):
        pass

