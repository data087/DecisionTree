"""

utils.py

"""
import pandas as pd
import numpy as np


def load_data(file_name):
    file_path = "C:/Users/dbs08/PycharmProjects/DecisionTree/data/"
    file = pd.read_csv(file_path + file_name, header=None)
    count_rows = len(file)
    count_columns = len(file.columns)
    print("-------------------------------")
    print("         Load Data \n")
    print(F"[*] Valid Column number : {count_columns - 1} + 1 [label]")
    print(F"[*] Valid Row    number : {count_rows}")
    print("-------------------------------")

    x = file.iloc[:, 0:-1]
    y = file.iloc[:, -1]
    return x, y
