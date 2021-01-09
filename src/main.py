# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd

file_name = "./datasets/training_set.csv"
file_name2 = "./datasets/test.csv"
file_name3 = "./datasets/training_set.xlsx"

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print_hi('PyCharm')
    data = pd.read_excel(file_name3)
    #data = np.genfromtxt(file_name,dtype=str,delimiter=',',skip_header=1)
    print(type(data))
    data = data.to_numpy()
    print(data.shape)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
