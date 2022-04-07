import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator,TabularDataset
from torchtext import  vocab,data
import csv
import sys
import pandas as pd
import numpy as np
import unicodedata
from cleantext import clean
from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n
import gensim.downloader as api
import re
import random
import time
import math


def normalised_text(text):
    def strip_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text(separator=" ")
        text=stripped_text
        return text
    
    text=strip_html_tags(text)
    text=text.replace("’","'")
    
    def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match)\
                                    if contraction_mapping.get(match)\
                                    else contraction_mapping.get(match.lower())                       
            expanded_contraction = first_char+expanded_contraction[1:]
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        text= expanded_text
        return text
    
    
    text=re.sub("[\(\[].*?[\)\]]", "", text)
    
    text=expand_contractions(text, contraction_mapping=CONTRACTION_MAP)
    text=text.replace("'s",'')
    
    text=clean(text.strip(),
    fix_unicode=True,               # fix various unicode errors
    to_ascii=True,                  # transliterate to closest ASCII representation
    lower=True,                     # lowercase text
    no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=True,                # replace all email addresses with a special token
    no_phone_numbers=True,         # replace all phone numbers with a special token
    no_numbers=True,               # replace all numbers with a special token
    no_digits=True,                # replace all digits with a special token
    no_currency_symbols=True,      # replace all currency symbols with a special token
    no_punct=True,
    replace_with_url="",
    replace_with_email="",
    replace_with_phone_number="",
    replace_with_number="",
    replace_with_digit="",
    replace_with_currency_symbol="",# fully remove punctuation
    lang="en"                       # set to 'de' for German special handling
    )
    return text
  
  
  
  CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
 "shes"  :"she is", 
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}
  

  if __name__ == "__main__":
    replies_dic={}       #dictionary for sorting replies wrt no. of likes.
    title=[]
    cleaned_replies=[]   
    cleaned_text=[]
    image_path=[]       #CHANGE--> add headline list here
    image_id=[] 
    tlikes=[]
    
    re.compile('<title>(.*)</title>')
    nlp = spacy.load('en_core_web_sm')
    iter=0
    for i in range(9479):
      replies_dic={}
      try:
        iter=iter+1

        ad="/home/puneet/code/Multimodal Feedback/data/" +  str(iter) + r"/twitter_data.csv"
        img_path="/home/puneet/code/Multimodal Feedback/data/" +  str(iter) + r"/news_img.jpg"

        mm_data=pd.read_csv(ad)  
        news_title=mm_data['TWEET'][0]             #CHANGE-->append headline list here.
        text=mm_data['NEWS CONTENT'][0]            #CHANGE--> 'NEWS CONTENT' for some of the data-files 
        likes=mm_data['LIKES'][0]
        replies=mm_data['REPLIES'][0]   

        #print(news_title)
        print(likes)
        
        likes=likes.split('-::-')
        x=re.split('-::-',replies)
        for (reply,like) in zip(x,likes):
            replies_dic[reply]=int(like)

        sort_orders = sorted(replies_dic.items(), key=lambda x: x[1], reverse=True) 

        #'sort_orders' contains {reply, likes}
        print(sort_orders)
        
        ctr=0

        #getting the replies
        rep_list=[]
        #lik_list=[]
        
        kk=len(sort_orders)
        for ii in range(kk):
            comm=sort_orders[ii][0]
            #lik=sort_orders[ii][1]
            rep_list.append(comm)
            #lik_list.append(lik)
            
        text=normalised_text(text)     
        
        
        for rep in rep_list:  
            rep=' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)"," ",rep).split())
            rep=normalised_text(rep)
            if not rep:
                continue
                
            else :
                title.append(news_title)
                cleaned_replies.append(rep)
                cleaned_text.append(text)
                image_path.append(img_path)    
                image_id.append(iter)
                #tlikes.append(lik)
                tlikes.append(likes[ctr])
                ctr+=1
      except:
          continue
            
        #if not (replies and text):
        #TO DO :: Remove repies with shorter length.
     df = pd.DataFrame(data={"title": title, "src": cleaned_text, "trg": cleaned_replies,"img_path":image_path,"img_id":image_id,"likes":tlikes})
     df.to_csv("train_data_sorted_with_title1.csv",index=False)   
