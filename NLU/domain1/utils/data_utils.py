import pandas as pd

data_file = "../data/simulator/new_data.json"

df = pd.read_json(data_file, orient="records")
print(df.tail())

df.to_csv("../data/simulator/train.csv", sep="\t", header=False, index=False,
          columns=["inputData", "tag", "intent"], encoding="utf-8")
df.to_csv("../data/simulator/test.csv", sep="\t", header=False, index=False,
          columns=["inputData"], encoding="utf-8")
data_file = "../data/simulator/data.json"


data_file = "../data/real/new_data0928.json"
df = pd.read_json(data_file, orient="records")
print(df.tail())

df.to_csv("../data/real/train.csv", sep="\t", header=False, index=False,
          columns=["inputData", "tag", "intent"], encoding="utf-8")
df.to_csv("../data/real/test.csv", sep="\t", header=False, index=False,
          columns=["inputData"], encoding="utf-8")
