import pandas as pd
import json
import os
import re


files_name = "/home/user/NLP-Project-Trial/archive/News_Category_Dataset_v2.json"
json_list=[]

with open(files_name, encoding='utf-8-sig', errors= 'ignore') as project_file:
	for jsonObj in project_file:
		data = json.loads(jsonObj)
		json_list.append(data)
df = pd.DataFrame.from_dict(json_list)
for list1 in json_list:
	print(list1["category"])
		
df.to_csv('/home/user/NLP-Project-Trial/archive/News_Category_Dataset_v2.csv', index=False, encoding = 'utf-8-sig')




#"category","headline", "authors","link", "short_description", "date"
