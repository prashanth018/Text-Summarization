import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from word_embedding import embed_sentences
from dataload import loadTestData 
#from rouge import Rouge
import joblib
from rouge_metric import PyRouge
import nltk
from pprint import pprint
from newspaper import Article
from newspaper.article import ArticleDownloadState, ArticleException
from time import sleep
from nltk.tokenize import sent_tokenize
import connexion
from flask import Flask, render_template, request


# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='')
application = app.app


# Implement a simple health check function (GET)
@app.route('/health')
def health():
    # Test to make sure our service is actually healthy
    try:
        summarize("https://www.huffingtonpost.com/entry/hugh-grant-marries_us_5b09212ce4b0568a880b9a8c", 50)
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}


def dummy_loadTestData():
    testing_data = [ [ np.array(["This sentence is important for doc0." ,
                                 "Such a sentence is irrelevent for doc 0."]), 
                       np.random.rand(2,5,300), 
                       np.array(["This sentence is important for doc0."]) ],
                     [ np.array(["Lol that sentence is awesome for do1." , 
                                 "No way, this is irrelevent"]), 
                       np.random.rand(2,5,300), 
                                np.array(["Lol that sentence is awesome for do1."]) ] ]
    return testing_data


def evaluate(model, testing_data, batch_size = 128, upper_bound = 100, threshold = 1):
    """
        Build the actual summaries for test data and evaluate them
        To do: 
            - load the actual x_test (embed test sentences) and y_test (compute rouge score)
        
        Parameters: 
            testing_data           - np.array 
                                        ex: [ doc1, doc2, ... , docn]
                                         where doci = [sentences, x_test, summary]
                                             where sentences = np.array of string
                                                   x_test = np.array of matrices (embedded sentences)
                                                   summaries = np.array of sentences
        
        Returns: 
            Rouge evaluations
    """   
    #rouge = Rouge()
    rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
    r1evals = []
    r2evals = []
    summaries = []    

    all_predicted_summary = []
    all_true_summary = []
    
    
    for doc in testing_data: 
        sentences = doc[0]
        
        x_test_old = doc[1]
        s1 = x_test_old.shape[0]
        (s3,s4) = x_test_old[0].shape
        print(s1,s3,s4)
        x_test = np.random.rand(s1,1,190,s4)
        for i in range(s1) :
            x_test[i] = np.array( [ np.pad(x_test_old[i], ((190-s3,0),(0,0)), 'constant') ] )
            

        true_summary = doc[2]
        with graph.as_default():
            predicted_scores = model.predict(x_test, batch_size=batch_size)
        #argsorted_scores= np.argsort(predicted_scores)
        argsorted_scores = np.argpartition(np.transpose(predicted_scores)[0], 1)
        
        predicted_summary = []
        summary_length = 0
        
        i = 0
        
        while i < len(sentences) and summary_length < upper_bound: 
            sentence = sentences[argsorted_scores[i]]
            #if ( dummy_rouge( sentence , predicted_summary ) < threshold ):
            sentence = np.array([sentence])
            #print(sentence, predicted_summary)
            predicted_summary.append(sentence)
            summary_length += len(nltk.word_tokenize(sentence[0]))
                
            i+=1
            
        #evals.append(dummy_rouge( predicted_summary, true_summary, alpha = N))
        #r1score = rouge.saliency(predicted_summary, true_summary, alpha=1)
        #r2score = rouge.saliency(predicted_summary, true_summary, alpha=0)

        temp = []
        for s in predicted_summary:
            temp.append(s[0])
        predicted_summary = '\n'.join(temp)
        
        for s in true_summary:
            temp.append(s)
        #print("**********************")
        #print(predicted_summary)
        #print(true_summary)
        #print("**********************")
        
        all_predicted_summary.append(predicted_summary)
        all_true_summary.append(true_summary)

        #evals.append(rouge.saliency(predicted_summary, true_summary, alpha=N))
        summaries.append((predicted_summary, true_summary))


    scores = rouge.evaluate_tokenized(all_predicted_summary, all_true_summary)
    print(" *--*--*--*--*--*--*--*")

    return scores

def createSentenceEmbeddings(sentences):
    size = len(sentences)

    test_data = []
    count = 0
    max_size = 0
    
    documents_over_190 = 0
    sentences_over_190 = 0
    sentences_removed = 0
    over_190 = False

    arr = np.ones((len(sentences), 3), dtype=object) 
    arr[:,0] = "dummy"
    arr[:,1] = np.array(sentences)
    embedding = embed_sentences(arr)
    embedding = embedding[0::2]

    for e in embedding:
        if len(e) > max_size:
            max_size = len(e)
        if len(e) > 190:
            sentences_over_190 += 1
            over_190 = True
    if over_190:
        documents_over_190 += 1
        over_190 = False
        count -= len(sentences)
        sentences_removed += len(sentences)
        return

    count += len(sentences)
    test_data.append((np.array(sentences), np.array(embedding)))
    print("Finished", count, "of", size,"sentences --", count/size,"%", end='\r')

    return test_data


def preprocessing(sentences):
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    return clean_sentences

def predict(model, data, batch_size = 128, upper_bound = 100, threshold = 1):
    """
        Predict the summary
        
        Parameters: 
            data           - np.array 
                                        ex: [ doc1, doc2, ... , docn]
                                         where doci = [sentences, x_test]
                                             where sentences = np.array of string
                                                   x_test = np.array of matrices (embedded sentences)        
        Returns: 
            Summary
    """   
    
    for doc in data: 
        sentences = doc[0]
        
        x_test_old = doc[1]
        s1 = x_test_old.shape[0]
        (s3,s4) = x_test_old[0].shape
        print(s1,s3,s4)
        x_test = np.random.rand(s1,1,190,s4)
        for i in range(s1) :
            x_test[i] = np.array( [ np.pad(x_test_old[i], ((190-s3,0),(0,0)), 'constant') ] )
        
        predicted_scores = model.predict(x_test, batch_size=batch_size)
        #argsorted_scores= np.argsort(predicted_scores)
        argsorted_scores = np.argpartition(np.transpose(predicted_scores)[0], 1)
        
        predicted_summary = []
        summary_length = 0
        
        i = 0
        
        while i < len(sentences) and summary_length < upper_bound: 
            sentence = sentences[argsorted_scores[i]]
            #if ( dummy_rouge( sentence , predicted_summary ) < threshold ):
            sentence = np.array([sentence])
            #print(sentence, predicted_summary)
            predicted_summary.append(sentence)
            summary_length += len(nltk.word_tokenize(sentence[0]))
                
            i+=1

        temp = []
        for s in predicted_summary:
            temp.append(s[0])
        predicted_summary = ' '.join(temp)
        return predicted_summary

    return []


def create_summary(model, article, summary_length):
    sentences = [sent_tokenize(article)]
    sentences = [y for x in sentences for y in x]
    clean_sentences = preprocessing(sentences)
    #print(clean_sentences)
    sentences_vector = createSentenceEmbeddings(clean_sentences)
    #print(sentences_vector[0][1].shape)
    return predict(model, sentences_vector, upper_bound=summary_length)


@app.route('/summarize')
def summarize(url, summary_length):
    """
        :param summary_length: length of the summary in percentage
        :param url: url to scrape from the web
        :return: call the summarize function
        """
    # url = url.strip("https://")
    article_huff = Article(url)
    slept = 0
    article_huff.download()
    while article_huff.download_state == ArticleDownloadState.NOT_STARTED:
        # Raise exception if article download state does not change after 12 seconds
        if slept > 13:
            raise ArticleException('Download never started')
        sleep(1)
        slept += 1

    article_huff.parse()
    news_text, news_title = article_huff.text, article_huff.title
    summary = create_summary(model, article_huff.text, summary_length)
    return {"title": article_huff.title, "summary": summary, "article": news_text}


# pprint(summarize('https://www.huffpost.com/entry/marco-rubio-donald-trump_n_607e5df5e4b063a636fb3f39', 100))


global graph
graph = tf.get_default_graph()
model = load_model('../models/model-nfilt-200.h5')
model._make_predict_function()

# Read the API definition for our service from the yaml file
app.add_api("api.yaml")

#
# # Start the app
if __name__ == "__main__":
    app.run()
