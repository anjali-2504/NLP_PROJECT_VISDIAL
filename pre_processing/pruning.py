import pandas as pd
from collections import OrderedDict
import spacy
import pickle

def tokenize_text(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in nlp(text)]

def tokenize_text1(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in nlp.tokenizer(text)]    

def tokenize_text2(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.tag_ for tok in nlp(text)]  

def make_col():
    for i in range(len(df)):
      arr1=[]
      arr1.append('<UNK>')
      for k in tokenize_text2(df['src'][i]):
        arr1.append(k)
      arr1.append('<PAD>')
      df['tags_src'][i]=arr1
      if(len(arr1)!=len(tokenize_text1(df['src'][i]))+2):
        print("yes")
    
    for i in range(len(df)):
      arr1=[]
      arr1.append('<UNK>')
      for k in tokenize_text2(df['trg'][i]):
        arr1.append(k)
      arr1.append('<PAD>')
      df['tags_trg'][i]=arr1
      if(len(arr1)!=len(tokenize_text1(df['trg'][i]))+2):
        print("yes")



def build_tags(lines):
    pos_vocab = OrderedDict()
    pos_vocab['<PAD>'] = 0
    pos_vocab['<UNK>'] = 1
    k = 2
    for line in lines:
        
        for tag in line:
            if tag not in pos_vocab:
                pos_vocab[tag] = k
                k += 1
    return pos_vocab
  
def get_pos_tag_index_seq(pos_seq, max_len):
    seq = list()
    ##print(pos_seq)
    for t in pos_seq:
        if t in pos_vocab:
            #print("yes")
            seq.append(pos_vocab[t])
        else:
            seq.append(pos_vocab['<UNK>'])
    pad_len = max_len - len(seq)
    for i in range(0, pad_len):
        seq.append(pos_vocab['<PAD>'])
    #if(len(seq)!=l1+2):
    ##  df.drop(i)
    #  print("yes",len(seq),l1+2,i)
    return seq

  
if __name__ == "__main__":  
   nlp = spacy.load('en_core_web_sm')
   path=r"train_data_sorted_with_title1.csv"
   df=pd.read_csv(path)
   
   df['tags_trg']=[sent for sent in df['trg']]  ###do for 'tags_Src' as well just for initilaising 
   df['tags_src']=[sent for sent in df['src']]  
   df['S_No']=[i for i in range(len(df))]
   print(len(pos_vocab))
   
   pos_tag_dict={}
   pos_vocab=build_tags(df['tags_src'])
   
   for i in range(len(df)):
     pos_tag_dict[i]=get_pos_tag_index_seq([x for x in df['tags_src'][i]],len(df['tags_src'][i])) 
   pos_tag_file=open("pos_tag_seq.pkl","wb")
   pickle.dump(pos_tag_dict,pos_tag_file)
   pos_tag_file.close()
  
   df.to_csv("train_data_sorted_with_title1.csv",index=False)
