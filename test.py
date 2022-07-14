import numpy as np
import pandas as pd

df = pd.read_csv("./data/taobao_hot.csv", nrows=50)
res = [eval(item) for item in df["item_list"]]
print(res)