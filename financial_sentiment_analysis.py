#Financial Sentiment Analysis

import scrapy

class FinancecrawlerItem(scrapy.Item):
	# define the fields for your item here like:
	# name = scrapy.Field()
	date = scrapy.Field()
	keywords = scrapy.Field()
	body = scrapy.Field()


from financeCrawler.items import FinancecrawlerItem

def getCleanStartUrlList(filename):
    """
    Takes as input the name of the txt file generated by next
    script. In this file each line is the url of a blog post published 
    in the period (1 January 2008 - 15 August 2014). The function returns
    a list of all the urls to be scraped.
    """
    myfile = open(filename, "r")
    urls = myfile.readlines()
    return [url.strip() for url in urls]
    
    

class BWSpider(scrapy.Spider):
    # name of the spider
    name = "busweek"
    # domains in which the spider can operate
    allowed_domains = ["businessweek.com"]
    # list of urls to be scraped
    urls = getCleanStartUrlList('businessweek.txt')
    start_urls = urls

    def parse(self, response):
        # the parse method is called by default on each url of the
        # start_urls list 
        item = FinancecrawlerItem()
        # the date, keywords and body attributes are retrieved from
        # the response page using the XPath query language
        item['date'] = response.xpath('//meta[@content][@name="pub_date"]/@content').extract()
        item['keywords'] = response.xpath('//meta[@content][@name="keywords"]/@content').extract() 
        item['body'] = response.xpath('//div[@id = "article_body"]/p/text()').extract()
        # the complete item filled with all its attributes 
        yield item

"""
The archive is nicely structured, thus the purpose of this script is 
to generate a txt file containing all the urls of the blog-posts 
published between 1 January 2008 and 15 August 2014.
In order to achieve this goal I implemented the following steps:
1- generate the urls of all the months in the time interval
2- generate the urls of all the days for each month
3- scrape each of the day-urls and get all the urls of the 
   posts published on that specific day.
4- repeat for all the days on which something was published 
"""
import scrapy
import urllib

def businessWeekUrl():
    totalWeeks = []
    totalPosts = []
    url = 'http://www.businessweek.com/archive/news.html#r=404'
    data = urllib.urlopen(url).read()
    hxs = scrapy.Selector(text=data)
    
    months = hxs.xpath('//ul/li/a').re('http://www.businessweek.com/archive/\\d+-\\d+/news.html')    
    admittMonths = 12*(2013-2007) + 8
    months = months[:admittMonths]

    for month in months:
        data = urllib.urlopen(month).read()
        hxs = scrapy.Selector(text=data)
        weeks = hxs.xpath('//ul[@class="weeks"]/li/a').re('http://www.businessweek.com/archive/\\d+-\\d+/news/day\\d+\.html')
        totalWeeks += weeks
    
    for week in totalWeeks:
        data = urllib.urlopen(week).read()
        hxs = scrapy.Selector(text=data)
        posts = hxs.xpath('//ul[@class="archive"]/li/h1/a/@href').extract()
        totalPosts += posts
    
    with open("businessweek.txt", "a") as myfile:
        for post in totalPosts:
            post = post + '\n'
            myfile.write(post)

businessWeekUrl()

scrapy crawl busweek -o businessweek.json

# import modules necessary for all the following functions
import re
import pandas as pd
from sklearn import preprocessing

def readJson(filename):
    """
    reads a json file and returns a clean pandas data frame
    """
    import pandas as pd
    df = pd.read_json(filename)
    
    def unlist(element):
        return ''.join(element)
    
    for column in df.columns:
        df[column] = df[column].apply(unlist)
    # gets only first 10 characters of date: year/month/day
    df['date'] = df['date'].apply(lambda x: x[:10])
    df['date'] = pd.to_datetime(df['date'])
    
    # if any removes duplicate posts
    df = df.drop_duplicates(subset = ['keywords'])
    # sorts dataframe by post date
    df = df.sort(columns='date')
    df['text'] = df['keywords'] + df['body'] 

    df = df.drop('body', 1)
    df = df.drop('keywords', 1)
    
    return df

def cleanText(text):
    """
    removes punctuation, stopwords and returns lowercase text in a list of single words
    """
    text = text.lower()    
    
    from bs4 import BeautifulSoup
    text = BeautifulSoup(text).get_text()
    
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    
    from nltk.corpus import stopwords
    clean = [word for word in text if word not in stopwords.words('english')]
    
    return clean

def loadPositive():
    """
    loading positive dictionary
    """
    myfile = open('/home/LoughranMcDonald_Positive.csv', "r")
    positives = myfile.readlines()
    positive = [pos.strip().lower() for pos in positives]
    return positive 

def loadNegative():
    """
    loading positive dictionary
    """
    myfile = open('/home/LoughranMcDonald_Negative.csv', "r")
    negatives = myfile.readlines()
    negative = [neg.strip().lower() for neg in negatives]
    return negative


def countNeg(cleantext, negative):
    """
    counts negative words in cleantext
    """
    negs = [word for word in cleantext if word in negative]
    return len(negs)

def countPos(cleantext, positive):
    """
    counts negative words in cleantext
    """
    pos = [word for word in cleantext if word in positive]
    return len(pos)   

def getSentiment(cleantext, negative, positive):
    """
    counts negative and positive words in cleantext and returns a score accordingly
    """
    positive = loadPositive()
    negative = loadNegative()
    return (countPos(cleantext, positive) - countNeg(cleantext, negative))

def updateSentimentDataFrame(df):
    """
    performs sentiment analysis on single text entry of dataframe and returns dataframe with scores
    """
    positive = loadPositive()
    negative = loadNegative()   
    
    df['text'] = df['text'].apply(cleanText)
    df['score'] = df['text'].apply(lambda x: getSentiment(x,negative, positive))

    return df

def prepareToConcat(filename):
    """
    load a csv file and gets a score for the day
    """
    df = pd.read_csv(filename, parse_dates=['date'])
    df = df.drop('text', 1)
    df = df.dropna()
    df = df.groupby(['date']).mean()
    name = re.search( r'/(\w+).csv', filename)
    df.columns.values[0] = name.group(1)
    return df

def mergeSentimenToStocks(stocks):
    df = pd.read_csv('/home/sentiment.csv', index_col = 'date')
    final = stocks.join(df, how='left')
    return final
