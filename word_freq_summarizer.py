from newspaper import article
import pandas as pd
import numpy as np
import nltk
import re
import connexion
from time import sleep
from newspaper import Article
from newspaper.article import ArticleDownloadState, ArticleException


nltk.download('punkt')


# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        summarize("https://www.huffingtonpost.com/entry/hugh-grant-marries_us_5b09212ce4b0568a880b9a8c", 2)
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}

def casefolding(sentence):
    return sentence.lower()

def cleaning(sentence):
    return re.sub(r'[^a-z]', ' ', re.sub("’", '', sentence))

def tokenization(sentence):
    return sentence.split()

def sentence_split(paragraph):
    return nltk.sent_tokenize(paragraph)

def word_freq(data):
    w = []
    for sentence in data:
        for words in sentence:
            w.append(words)
    bag = list(set(w))
    res = {}
    for word in bag:
        res[word] = w.count(word)
    return res

def summary_ranking(news_text,n):
    sentence_list = sentence_split(str(news_text))
    data = []
    for sentence in sentence_list:
        data.append(tokenization(cleaning(casefolding(sentence))))
    data = (list(filter(None, data)))
    wordfreq = word_freq(data)
    ranking = []
    for words in data:
        temp = 0
        for word in words:
            temp += wordfreq[word]
        ranking.append(temp)
    
    result = ''
    sort_list = np.argsort(ranking)[::-1][:n]
        #print(sort_list)
    l = []
    for i in range(n):
        result += '{} '.format(sentence_list[sort_list[i]])
    return result

def summarize(url, summary_length):
    """
        :param summary_length: length of the summary in percentage
        :param url: url to scrape from the web
        :return: call the summarize function
        """
    # url = url.strip("https://")
    # print(url)
    article_huff = Article(url)
    slept = 0
    article_huff.download()
    while article_huff.download_state == ArticleDownloadState.NOT_STARTED:
        # Raise exception if article download state does not change after 12 seconds
        if slept > 13:
            raise ArticleException('Download never started')
    sleep(1)
    slept += 1
    n = summary_length
    
    article_huff.parse()
    news_text = article_huff.text
 
    summary = summary_ranking(news_text,n)
    # print(article_huff.title)
    return {"title": article_huff.title, "summary": summary}

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__)
application = app.app

# Read the API definition for our service from the yaml file
app.add_api("news_summ_freq_api.yaml")

#Start the app
if __name__ == "__main__":
    app.run(port=8080)

#print(summarize("https://www.huffingtonpost.com/entry/hugh-grant-marries_us_5b09212ce4b0568a880b9a8c", 2))
