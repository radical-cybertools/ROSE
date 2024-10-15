import numpy as np
import pandas as pd

path = "/scratch/pb_data"

train_csvs = []
test_csvs = []
for i in range(1, 21):
    df = pd.read_csv(path + "/" + str(i) + "SourcesRdm/train.csv", sep=",", header=None)
    print(df.head())
    exit()