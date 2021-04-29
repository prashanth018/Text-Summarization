import json
from time import sleep
import io
from newspaper import Article
from newspaper.article import ArticleException, ArticleDownloadState
import pandas as pd
import numpy
from newspaper import Config

iteration_from = 45000
iteration_end = 50000

df = pd.read_csv("C:/Users/user/Desktop/NLP/project-trial/News_Category_Dataset_v2.csv",encoding = 'utf-8-sig')
url_list = df['link'].to_list()
category = df['category'].tolist()
headline = df['headline'].tolist()
short_desc = df['short_description'].tolist()

url_list1 = url_list[iteration_from:iteration_end]
category1 = category[iteration_from:iteration_end]
headline1= headline[iteration_from:iteration_end]
short_desc1 = short_desc[iteration_from:iteration_end]
#print(category)	
#        "category","headline", "authors","link", "short_description", "date"
final_data = []
def scrape_url(urls):	
    """
    :param url_list: list of urls to scrape from the web
    :return: nothing, just create a text file containing article text of each url
    """
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36"

    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 10

    info_scrapped = {}
    
    info_scrapped["news_text"] = None

      # iterate through each article link
    #slept = 0
    print(urls)
    try:
        article_huff = Article(urls,config = config)
        article_huff.download()
        article_huff.parse()
        article_info_huff = article_huff.text
        info_scrapped["news_text"] = article_info_huff
        final_data.append(info_scrapped)

    except:
        pass
        info_scrapped["news_text"] = None
        final_data.append(info_scrapped)
    
    #while article_huff.download_state == ArticleDownloadState.NOT_STARTED:
            # Raise exception if article download state does not change after 11 seconds
    #    if slept > 12:
    #        raise ArticleException('Download never started')
    sleep(1)
    #slept += 1

       
    df2 = pd.DataFrame(final_data)
    df2['category'] = pd.Series(category1)
    df2['headline'] = pd.Series(headline1)
    df2['short_desc'] = pd.Series(short_desc1)
    df2['url'] = pd.Series(url_list1)
    df2.index += 1
    #print(df2)
    return df2


def __main__():
    #urls = read_file("/home/user/NLP-Project-Trial/archive/News_Category_Dataset_v2.json")
    complete_data=[]
    for i in range(iteration_from,iteration_end):
        url_texts = scrape_url(url_list[i].strip())
        url_texts.to_csv("C:/Users/user/Desktop/NLP/project-trial/Dataset/News_Category_Dataset-"+str(iteration_from)+"-"+str(iteration_end)+".csv")

__main__()


# We can write each of the following properties if required for classification or summarization
# article_info_huff = {'title': article_huff.title,
#                      'description': article_huff.meta_description,
#                      'text': article_huff.text,
#                      'publisher': 'huff'}
