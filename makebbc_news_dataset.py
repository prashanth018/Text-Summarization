import pandas as pd
import os
import re

list_of_files =[]
for f_name in os.listdir('/home/user/NLP-Project-Trial/bbc/archive/BBC News Summary/Summaries/entertainment/'):
    if f_name.endswith('.txt'):
        list_of_files.append('/home/user/NLP-Project-Trial/bbc/archive/BBC News Summary/Summaries/entertainment/'+f_name)

df_list=[]
for f in list_of_files:
    with open(f,encoding='utf-8', errors= 'ignore') as project_file:
    	lines = project_file.read()
    	df_list.append(lines)
    
  	    
big_df = pd.DataFrame(df_list,columns=['story'] )
big_df.to_csv("CombinedentertainmentSummary.csv", index=False)
