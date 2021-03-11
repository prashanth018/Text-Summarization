import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("Articles.csv")
    ct = 0
    for i in range(len(df)):
        if "Pay based on use" not in df.iloc[i]['text'] and "Gain a global perspective on the US" not in df.iloc[i]['text'] and "Become an FT subscriber to read" not in  df.iloc[i]['text']:
        # if "Pay based on use" and "Gain a global perspective on the US" not in df.iloc[i]['text']:
            print(df.iloc[i]['text'])
            print("*****************************************")
            ct += 1
    print(ct)
