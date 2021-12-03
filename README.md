We have traversed through 4 different algorithms in order to compare and analyse the summaries created by each of them. The four algorithms explored were: Text Rank Word frequency Seq2Seq CNN

We created a UI for our service. Swagger First, we created a Swagger API interface. This can be found in the following files: news_api.py news_api.yaml

In order to run the Swagger API, please run the following commands from the APIs folder in your terminal: python news_api.py

Then, in your browser, navigate to: http://localhost:8080/ui/

/summarize takes two parameters: Article URL: Only Huffington Post news article URLs Summary_length: Integer value within range 0-100 The article url is used to scrape the article text behind the scenes and the summary length value is used as a percentage. For instance, when summary_length is 10 then only the top 10% ranked sentences of the total sentences present in the article are returned as a summary, along with the title of the article.

/wf_summarize takes two parameters: Article URL: Only Huffington Post news article URLs Summary_length: Integer value within total length of the article sentences The article url is used to scrape the article text behind the scenes and the summary length value is used as the number of sentences we want in the summary. For instance: when summary_length is 10, then only the top 10 ranked sentences of the total sentences present in the article are returned as a summary along with the title of the article.
