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
"""
import pandas as pd
import numpy as np
import time


class TreeNode:
    def __init__(self, depth):
        # UI 관련
        self.entropy = None
        self.division_column_name = None
        self.division_column_point = None
        self.depth = depth + 1
        self.labelCount = None

        # 자식 노드 관련
        self.left = None
        self.right = None

        # 데이터 관련
        self.data_index = data_index
        self.df = pd.DataFrame(data=data)
        self.count_rows = len(self.df)
        self.count_columns = len(self.df.columns)
        self.df['label'] = label
        self.label_unique = np.unique(self.df['label'])
