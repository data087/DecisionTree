import src.utils
import src.tree


def main():
    data_x, data_y = src.utils.load_data("iris.data")
    tree = src.tree.DecisionTree(data=data_x, label=data_y)


if __name__ == "__main__":
    main()
