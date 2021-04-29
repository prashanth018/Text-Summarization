import pandas as pd
import glob

path = r'/home/user/NLP-Project-Trial/bbc/all' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame.to_csv("BBCDataset.csv", index=False)
