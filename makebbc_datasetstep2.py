import pandas as pd
import os
import re

df1 = pd.read_csv('/home/user/NLP-Project-Trial/CombinedtechStory.csv')
df2 = pd.read_csv('/home/user/NLP-Project-Trial/CombinedtechSummary.csv')

det = pd.concat([df1,df2], join = 'outer', axis = 1)

det.to_csv("Combinedtech.csv", index=False)

