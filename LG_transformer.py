
from nltk.tokenize import PunktSentenceTokenizer
import nltk
from collections import OrderedDict
from torch.types import Device
from nltk.corpus import state_union
import pandas as pd
import torch, torchvision
import pickle
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torchtext import  vocab,data
from torchtext.legacy.datasets import TranslationDataset, Multi30k
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, csv, sys, random, re, time, math, spacy, nltk
from PIL import Image
from numpy.random import RandomState
from tensorboardX import SummaryWriter
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torchtext
from tqdm import tqdm
import torch.nn as nn

def make_new_columns(df):
      df['S_No']=[i for i in range(len(df))]
      return df

def get_max_len(sample_batch):
    src_max_len = len(sample_batch.src[0])
    for idx in range(1, len(sample_batch)):
        if len(sample_batch.src[0]) > src_max_len:
            src_max_len = len(sample_batch[idx].SrcWords)
    return src_max_len

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.vis_fc=nn.Linear(NF, hid_dim)

        self.vis_fcg=nn.Linear(1000, hid_dim)
        self.fc=nn.Linear(2*hid_dim,hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)         
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src,vs102,vs103,vs102_g,vs103_g, src_mask,pos_tag_seq,pos_tag_seq1):
        
        #src = [batch size, src len]
        #src_mask = [batch size, src len]
         
        batch_size = src.shape[0]
        src_len = src.shape[1]
        vs102=torch.tanh(self.vis_fc(vs102))
        
        vs103=torch.tanh(self.vis_fc(vs103))
        
        vs102_g=torch.tanh(self.vis_fcg(vs102_g))
        
        vs103_g=torch.tanh(self.vis_fcg(vs103_g))

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
     
        shape=self.tok_embedding(src).shape
        vs102=vs102+vs102_g
        vs103=vs103+vs103_g
        
        if(shape[1]==102):
            src=self.fc(torch.cat([self.tok_embedding(src),vs102],2))
            src = self.dropout((src * self.scale) + self.pos_embedding(pos))
        if(shape[1]==103):
            src=self.fc(torch.cat([self.tok_embedding(src),vs103],2))
            src = self.dropout((src * self.scale) + self.pos_embedding(pos))   
       
        #src = [batch size, src len, hid dim]
 
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
 
        return src



class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #print("_SRC SIZE {}".format(_src.size()))
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
 
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention



class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x


import torch.nn as nn
class POSEmbeddings(nn.Module):
    def __init__(self, tag_len, tag_dim, drop_out_rate):
        super(POSEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(tag_len, tag_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, pos_seq):
        pos_embeds = self.embeddings(pos_seq)
        pos_embeds = self.dropout(pos_embeds)
        return pos_embeds

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

       
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
       
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention


class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src,vs102,vs103,vs102_g,vs103_g, trg,pos_tag_seq,pos_tag_seq1):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]  model(src, vs102,vs103,trg[:,:-1])

        src_mask = self.make_src_mask(src)
    
        trg_mask = self.make_trg_mask(trg)
        

       
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src,vs102,vs103,vs102_g,vs103_g, src_mask,pos_tag_seq,pos_tag_seq1)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention



def get_pos_tag_seq_batch(batch,batch_len,outputs):
          pos_tag_seq=[]
          max_len=len(outputs[batch.S_No[0].item()])
          for i in range(batch_len):
            if len(outputs[batch.S_No[i].item()]) > max_len:
              max_len=len(outputs[batch.S_No[i].item()])

          for i in range(batch_len):
            seq=outputs[batch.S_No[i].item()]
            pad_len=max_len-len(outputs[batch.S_No[i].item()])
            for j in range(0, pad_len):
              seq.append(0)
            pos_tag_seq.append(seq)

          return pos_tag_seq



class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        
        self.energy = nn.Linear(hid_dim*2, 1)
        
        self.softmax = nn.Softmax(dim=0)
        
        self.relu = nn.ReLU()
        
    def forward(self, hidden, encoder_states):
        
        seq_len = encoder_states.shape[0]
        h_reshaped = hidden.repeat(seq_len, 1, 1)

        emb_con = torch.cat((h_reshaped, encoder_states), dim=2)
        
        energy = self.relu(self.energy(emb_con))
        attention = self.softmax(energy)
        return attention


        
def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in tqdm(enumerate(iterator)):
        
        src= batch.src
        trg = batch.trg
        img_id=batch.img_id
        x=img_id.cpu().numpy()
        y=len(x)
        
        visual_features_102=torch.empty(y,max_length,NF).to(device)
        visual_features_103=torch.empty(y,max_length,NF).to(device)

        visual_features_102_global=torch.empty(y,max_length,1000).cuda()
        visual_features_103_global=torch.empty(y,max_length,1000).cuda()

        
        df=pd.read_csv(path)
        df1=pd.read_csv(path1)
        
        for i in range(y):
            q=df[str(x[i]-1)].to_numpy()
            r=np.zeros((max_length,1),dtype=q.dtype) + q
            r=torch.from_numpy(r).float()
            visual_features_102[i]=r
            q1=df1[str(x[i]-1)].to_numpy()
            r1=np.zeros((max_length,1),dtype=q1.dtype) + q1
            r1=torch.from_numpy(r1).float()
            visual_features_102_global[i]=r1

            
        vs102=visual_features_102.float()
        vs103=visual_features_103.float()
        vs102_global=visual_features_102_global.float()
        vs103_global=visual_features_103_global.float()


        ###adding extraaa ####
        batch_len=list(batch.S_No.size())[0]
        pos_tag_seq=get_pos_tag_seq_batch(batch,batch_len,outputs)
        pos_tag_seq=torch.tensor(pos_tag_seq).permute(1,0)
        item_appended=pos_tag_seq[[-1]]
        
        pos_tag_seq1=pos_tag_seq[1:103]
        pos_tag_seq=pos_tag_seq[1:102]
        

        pos_tag_seq=torch.cat((pos_tag_seq,item_appended),0)
        pos_embeds=POSEmbeddings(102, pos_embed_dim, 0.5)
        pos_tag_seq=pos_embeds(pos_tag_seq).permute(1,0,2).to(device)

        
        pos_tag_seq1=torch.cat((pos_tag_seq1,item_appended),0)
        pos_embeds1=POSEmbeddings(103, pos_embed_dim, 0.5)
        pos_tag_seq1=pos_embeds1(pos_tag_seq1).permute(1,0,2).to(device)
        
        optimizer.zero_grad()

        output, _ = model.forward(src, vs102,vs103,vs102_global,vs103_global,trg[:,:-1],pos_tag_seq,pos_tag_seq1)
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
              
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)




def evaluate(model, iterator, criterion):

    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src

            trg = batch.trg

            img_id=batch.img_id

            x=img_id.cpu().numpy()
            y=len(x)
            visual_features_102=torch.empty(y,max_length,NF).to(device)
            visual_features_103=torch.empty(y,max_length,NF).to(device)
            visual_features_102_global=torch.empty(y,max_length,1000).cuda()
            visual_features_103_global=torch.empty(y,max_length,1000).cuda()

            df=pd.read_csv(path)
            df1=pd.read_csv(path1)

            for i in range(y):
                q=df[str(x[i]-1)].to_numpy()

                r=np.zeros((max_length,1),dtype=q.dtype) + q
                r=torch.from_numpy(r).float()
                visual_features_102[i]=r
                q1=df1[str(x[i]-1)].to_numpy()

                r1=np.zeros((max_length,1),dtype=q1.dtype) + q1
                r1=torch.from_numpy(r1).float()
                visual_features_102_global[i]=r1
            

            vs102=visual_features_102.float()
            vs103=visual_features_103.float()
            vs102_global=visual_features_102_global.float()
            vs103_global=visual_features_103_global.float()

              ###adding extraaa ####
            batch_len=list(batch.S_No.size())[0]
            pos_tag_seq=get_pos_tag_seq_batch(batch,batch_len,outputs)
            pos_tag_seq=torch.tensor(pos_tag_seq).permute(1,0)
            item_appended=pos_tag_seq[[-1]]
        
            pos_tag_seq1=pos_tag_seq[1:103]
            pos_tag_seq=pos_tag_seq[1:102]
        

            pos_tag_seq=torch.cat((pos_tag_seq,item_appended),0)
            pos_embeds=POSEmbeddings(102, pos_embed_dim, 0.5)
            pos_tag_seq=pos_embeds(pos_tag_seq).permute(1,0,2).to(device)

        
            pos_tag_seq1=torch.cat((pos_tag_seq1,item_appended),0)
            pos_embeds1=POSEmbeddings(103, pos_embed_dim, 0.5)
            pos_tag_seq1=pos_embeds1(pos_tag_seq1).permute(1,0,2).to(device)

            output, _ = model(src,vs102,vs103,vs102_global,vs103_global, trg[:,:-1],pos_tag_seq,pos_tag_seq1)
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def get_output(path):
    dict_file = open(path, "rb")
    outputs = pickle.load(dict_file)
    dict_file.close() 
    return outputs


def translate_sentence(src, src_field, trg_field,img_id,path, model, device,pos_tag_seq,pos_tag_seq1, max_len = 50):
    
    model.eval()
    src_mask = model.make_src_mask(src)
    
    df=pd.read_csv(path)
    visual_features_102=torch.empty(1,max_length,NF).to(device)
    visual_features_103=torch.empty(1,max_length+1,NF).to(device)
    q=df[str(img_id-1)].to_numpy()
    r=np.zeros((max_length,1),dtype=q.dtype) + q
    r=torch.from_numpy(r).float()
    visual_features_102[0]=r

    df1=pd.read_csv(path1)
    visual_features_102_g=torch.empty(1,max_length,1000).to(device)
    visual_features_103_g=torch.empty(1,max_length+1,1000).to(device)

    q1=df1[str(img_id-1)].to_numpy()
    r1=np.zeros((max_length,1),dtype=q1.dtype) + q1
    r1=torch.from_numpy(r1).float()
    visual_features_102_g[0]=r1

    
    with torch.no_grad():
        enc_src = model.encoder(src,visual_features_102,visual_features_103,visual_features_102_g,visual_features_103_g, src_mask,pos_tag_seq,pos_tag_seq1)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
     #   print(pred_token)
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    #print("\n")
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention





def tokenize_text(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in nlp.tokenizer(text)]

def data_splitting():

    rng = RandomState()
    train_data = df.sample(frac=0.75, random_state=rng)
    val_test_data = df.loc[~df.index.isin(train_data.index)]
    val_data=val_test_data.sample(frac=0.15, random_state=rng)
    test_data=val_test_data.loc[~val_test_data.index.isin(val_data.index)]
    cols= ["src", "trg","img_id","img_path","S_No"]
    train_data.to_csv('train_data.csv', index= False,columns=cols) 
    val_data.to_csv('val_data.csv', index= False,columns=cols)
    test_data.to_csv('test_data.csv', index= False,columns=cols)

def splitting(tokenize_text):

    ID = torchtext.legacy.data.Field(sequential=False,use_vocab=False)
    S_no = torchtext.legacy.data.Field(sequential=False,use_vocab=False)
    
    SRC = Field(tokenize = tokenize_text, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True,
                batch_first=True,
                fix_length=max_length)
    TRG = Field(tokenize = tokenize_text, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True,
                batch_first=True,
                fix_length=max_length
                )
    
    datafields=[('src', SRC), ('trg', TRG),('img_id',ID),('img_path',None),('S_No',S_no)]
    
    #rng = RandomState()
    #train_data = df.sample(frac=0.75, random_state=rng)
    #val_test_data = df.loc[~df.index.isin(train_data.index)]
    #val_data=val_test_data.sample(frac=0.15, random_state=rng)
    #test_data=val_test_data.loc[~val_test_data.index.isin(val_data.index)]
    
    #cols= ["src", "trg","img_id","img_path","pos_tag_seq"]
    
    #cols= ["src", "trg","img_id","img_path","S_No"]
    
    
    #train_data.to_csv('train_data.csv', index= False,columns=cols) #columns=cols
    #val_data.to_csv('val_data.csv', index= False,columns=cols) #columns=cols
    #test_data.to_csv('test_data.csv', index= False,columns=cols)

    train_data, val_data ,test_data= torchtext.legacy.data.TabularDataset.splits(path=r"",train="/home/pkumar/code/auto_eval/train_data.csv", validation="/home/pkumar/code/auto_eval/val_data.csv",test="/home/pkumar/code/auto_eval/test_data.csv" , format='csv', skip_header=True, fields=datafields)
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=1)
    return train_data,val_data,test_data,SRC,TRG

def dataloader_train_val(train_data,val_data,test_data):


    train_iterator, x_iterator = BucketIterator.splits(
        (train_data, train_data), 
        batch_size = BATCH_SIZE,
       # sort_within_batch = True,
       # sort_key = lambda x : len(x.src), 
        device = device)
    
    valid_iterator, y_iterator = BucketIterator.splits(
        (val_data, val_data), 
    
        batch_size = BATCH_SIZE, 
      #  sort_within_batch = True,
     #   sort_key = lambda x : len(x.src),
        device = device)

    test_iterator, z_iterator = BucketIterator.splits(
    (test_data, test_data), 

    batch_size = 1, 
    #sort_within_batch = True,
    #sort_key = lambda x : len(x.src),
    device = device)    
    return train_iterator ,valid_iterator ,test_iterator

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def load_checkpt(model, optimizer, chpt_file):
    start_epoch = 0
    if (os.path.exists(chpt_file)):
        print("=> loading checkpoint '{}'".format(chpt_file))
        checkpoint = torch.load(chpt_file)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        
        print("=> loaded checkpoint '{}' (epoch {})".format(chpt_file, checkpoint['epoch']))
        
    else:
        print("=> Checkpoint NOT found '{}'".format(chpt_file))
    return model, optimizer, start_epoch



def plot_results(losses,epochs):
    plt.plot(np.arange(epochs),losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('loss'+str(epochs)+'.png')


def qual_eval(model, optimizer, start_epoch, chpt_file,path,base_path,outputs):
    model, optimizer, start_epoch = load_checkpt(model, optimizer, chpt_file)
    T_EPOCHS = start_epoch + EP_INT
    
    # Save the predicted Feedbacks in CSV file
    # Log with Tensorboard: Text, Comment, Image and Feedback
    test_pred=[]
    test_df=pd.read_csv("/home/pkumar/code/auto_eval/test_data.csv")      
    length=len(test_df)
    images = []

    print(length)
    for i in range(length):
        #src = vars(val_data.examples[i])['src']
        #trg = vars(val_data.examples[i])['trg']
        if (len(test_df['src'][i].split()) >= max_length-2):
           #df['src'][i] = " ".join(df['src'][i].split()[0:max_length])
           test_df['src'][i]=" ".join(test_df['src'][i].split()[0:max_length-2])
           src=test_df['src'][i]
           trg=test_df['trg'].iloc[i]
           img_path=base_path+test_df['img_path'].iloc[i]
           image_idx=test_df['img_id'].iloc[i]
           pos_tag_seq=torch.tensor(outputs[test_df['S_No'].iloc[i]])
           pos_tag_seq = torch.LongTensor(pos_tag_seq).unsqueeze(1)
   
           item_appended=pos_tag_seq[[-1]]
           pos_tag_seq1=pos_tag_seq[1:103]
           pos_tag_seq=pos_tag_seq[1:102]
   
           pos_tag_seq=torch.cat((pos_tag_seq,item_appended),0)
           pos_tag_seq1=torch.cat((pos_tag_seq1,item_appended),0)
           pos_embeds=POSEmbeddings(102, pos_embed_dim, 0.5)
           pos_tag_seq=pos_embeds(pos_tag_seq).permute(1,0,2).to(device)
           pos_embeds1=POSEmbeddings(103, pos_embed_dim, 0.5)
           pos_tag_seq1=pos_embeds1(pos_tag_seq1).permute(1,0,2).to(device)
            
               
           nlp = spacy.load('en_core_web_sm')
           #nlp = spacy.load('en') ##https://www.gitmemory.com/issue/OmkarPathak/pyresparser/46/777568505
           tokens = [token.text.lower() for token in nlp(src)]
   
   
           tokens = [SRC.init_token] + tokens + [SRC.eos_token]
           
           src_indexes = [SRC.vocab.stoi[token] for token in tokens]
          
           src= torch.LongTensor(src_indexes).unsqueeze(0).to(device)
           print(src.size())
           if(src.size()[1]==102):
              translation = translate_sentence(src, SRC, TRG, int(image_idx), path, model, device,pos_tag_seq,pos_tag_seq1)
   
              if not translation:
                   translation="*empty*"
   
              #Untokenization    
              translation1=translation[0:(len(translation)-1)]    
   
              translation2 = TreebankWordDetokenizer().detokenize(translation1[0])
              print(" translation {} ".format( translation2))
   
              #print(" image id {} ".format( image_idx))
   
              #print(" src : {}".format(test_df['src'][i]))
   
              #print(" trg : {}".format(trg))
   
              test_pred.append(str(translation2))
   
              image = Image.open(img_path)
              image = ToTensor()(image)   
              if (i%10==0): 
               #Log with Tensorboard: Text, Comment, Image and Feedback
                   log_writer.add_text(str(T_EPOCHS)+'Ep=>Ground-truth Comment of Sample/'+str(i+1), str(trg))
                   log_writer.add_text(str(T_EPOCHS)+'Ep=>News Text of Sample/'+str(i+1), str(src))#, i+1)
                   log_writer.add_text(str(T_EPOCHS)+'Ep=>Predicted Feedback of Sample/'+str(i+1), str(translation))
               #log_writer.add_image('Image', image, i+1)
               #image_grid = torchvision.utils.make_grid(images)
                   log_writer.add_image(str(T_EPOCHS)+'Ep:Image of Sample/'+str(i+1), image)  
           else:
              test_pred.append("NULL")  
        else:
           test_pred.append("NULL")    

    with open(str(T_EPOCHS)+'Ep_test_results.csv', 'w'): 
        pass
    with open(str(T_EPOCHS)+'Ep_test_comments.csv', 'w'): 
        pass
    with open(str(T_EPOCHS)+'Ep_test_feedbacks.csv', 'w'): 
        pass

    test_df["pred"] = test_pred 
    test_df.to_csv(str(T_EPOCHS)+'Ep_test_results.csv', index= False)
    test_df.to_csv(str(T_EPOCHS)+'Ep_test_comments.csv', index= True, columns=["trg"]) #index -> "key": value -> ["trg/pred"]
    test_df.to_csv(str(T_EPOCHS)+'Ep_test_feedbacks.csv', index= True, columns=["pred"])

    #Re-open and save with new column names
    df1 = pd.read_csv(str(T_EPOCHS)+'Ep_test_comments.csv')
    df1.columns = ['id', 'comment']
    df1.to_csv(str(T_EPOCHS)+'Ep_test_comments.csv', index= False)

    df1 = pd.read_csv(str(T_EPOCHS)+'Ep_test_feedbacks.csv')
    df1.columns = ['id', 'feedback']
    df1.to_csv(str(T_EPOCHS)+'Ep_test_feedbacks.csv', index= False)
    
    print('tensorboard --logdir "/home/puneet/code/autoeval/TBlogs"\n---')


def interval_train(model, optimizer, start_epoch, chpt_file):
    model, optimizer, start_epoch = load_checkpt(model, optimizer, chpt_file)
    T_EPOCHS = start_epoch + EP_INT

    print('Already trained for',start_epoch, 'epochs. Training now for', EP_INT, 'more')

    best_valid_loss = float('inf')
    cur_best_train_loss = float('inf')

    train_losses=[]
    train_ppls=[]
    val_ppls=[]
    val_losses=[]

    #for epoch in range(EP_INT):
    for epoch in tqdm(range(start_epoch, T_EPOCHS)):    
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)#, log_writer_train)
        valid_loss = evaluate(model, valid_iterator, criterion)#, log_writer_val)
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        train_ppls.append(math.exp(train_loss))
        val_ppls.append(math.exp(valid_loss))

        #Log with Tensorboard: Loss & PPL for train & val 
        log_writer.add_scalar('Train/Loss',float(train_loss), epoch+1)
        log_writer.add_scalar('Train/PPL', float(math.exp(train_loss)), epoch+1)    
        log_writer.add_scalar('Val/Loss',float(valid_loss), epoch+1)
        log_writer.add_scalar('Val/PPL', float(math.exp(valid_loss)), epoch+1)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        #if valid_loss < best_valid_loss:
        #    best_valid_loss = valid_loss
        #if train_loss < cur_best_train_loss:
        #    cur_best_train_loss = train_loss
            #torch.save(model.state_dict(), '/home/puneet/code/Multimodal Feedback/checkpoints/baseline3.pt'
            #torch.save({
            #    'epoch': T_EPOCHS,
            #    'state_dict': model.state_dict(),
            #    'optimizer': optimizer.state_dict(),
            #    'loss': train_loss,
            #    }, '/home/puneet/code/Multimodal Feedback/checkpoints/baseline3.pt')

        state = {'epoch': T_EPOCHS, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': train_loss}
        torch.save(state, chpt_file)

        print(f'\nEpoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    

    torch.cuda.empty_cache()
    return train_losses, val_losses,train_ppls,val_ppls

def quant_eval(model, optimizer, start_epoch, chpt_file):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.spice.spice import Spice
    import os, json, csv
    
    model, optimizer, start_epoch = load_checkpt(model, optimizer, chpt_file)
    T_EPOCHS = start_epoch + EP_INT    
    

    #print('Download Stanford models... Run once!')
    os.system("sh get_stanford_models.sh")

    with open(str(T_EPOCHS)+'Ep_test_comments.csv',"r") as f: 
            reader = csv.reader(f)
            gts = {rows[0]:rows[1:] for rows in reader}
            #print(mydict) #prints with single quotes
            #print (json.dumps(mydict)) #prints with double quotes

    with open(str(T_EPOCHS)+'Ep_test_feedbacks.csv',"r") as g: 
            reader = csv.reader(g)
            res = {rows[0]:rows[1:] for rows in reader}
            #print(json.dumps(mydict))

    '''with open('temp/test_comments.json', 'r') as file:
        gts = json.load(file)
    with open('temp/test_feedbacks.json', 'r') as file:
        res = json.load(file)
    '''

    def bleu():
        scorer = Bleu(n=4)
        score, scores = scorer.compute_score(gts, res)
        return score


    def cider():
        scorer = Cider()
        (score, scores) = scorer.compute_score(gts, res)
        return score

    def rouge():
        scorer = Rouge()
        score, scores = scorer.compute_score(gts, res)
        return score

    #bgts = gts[0].encode(encoding='UTF-8')
    #bres = res[0].encode(encoding='UTF-8')

    def spice():
        scorer = Spice()
        #print(gts, res)
        score, scores = scorer.compute_score(gts, res)
        return score

    def meteor():
        scorer = Meteor()
        #print(gts, res)
        score, scores = scorer.compute_score(bgts, bres)
        return score    
    s_cider=cider()
    s_rouge=rouge()
    s_bleu=bleu()
    #s_spice=spice()#
   # s_meteor=meteor()#
    
    print('\n----------------------\nbleu = %s' %s_bleu )
    print('cider = %s' %s_cider )
    print('rouge = %s' %s_rouge )
    #print('spice = %s' %s_spice )
    #print('meteor = %s' %s_meteor )
    
    b=" ".join(str(x) for x in s_bleu)
    print('\n----------------------')
    f = open('scores.txt', 'w') 
    f.write("\ncider: %f" % s_cider)
    f.write("\nrouge: %f" % s_rouge)
    #f.write("\nspice: %f" % s_spice)
    #f.write("\nmeteor: %f" % s_meteor)
    f.write("\nbleu :")
    f.write(b)
    f.close()
    
    #print(str(T_EPOCHS))
    #Log with Tensorboard: Eval metrics
    log_writer.add_text(str(T_EPOCHS)+'Ep=>Metrics/cider', str(s_cider))
    log_writer.add_text(str(T_EPOCHS)+'Ep=>Metrics/rouge', str(s_rouge))
    #log_writer.add_text(str(T_EPOCHS)+'Ep=>Metrics/spice', str(s_spice))
    #log_writer.add_text(str(T_EPOCHS)+'Ep=>Metrics/meteor', str(s_meteor))
    log_writer.add_text(str(T_EPOCHS)+'Ep=>Metrics/bleu-1', str(s_bleu[0]))
    log_writer.add_text(str(T_EPOCHS)+'Ep=>Metrics/bleu-2', str(s_bleu[1]))
    log_writer.add_text(str(T_EPOCHS)+'Ep=>Metrics/bleu-3', str(s_bleu[2]))
    log_writer.add_text(str(T_EPOCHS)+'Ep=>Metrics/bleu-4', str(s_bleu[3]))


def save_losses(array,filename):
    file = open(filename, "a") 
    file.write('\n')
    for i in array:
        file.write(str(i))
        file.write('\n')
    file.close()

def crop(df,max_length):
    for i in range (df.shape[0]):
      if len(df['src'][i].split()) < max_length:
        df = df.drop(i)
      else:
        df['src'][i] = " ".join(df['src'][i].split()[0:max_length])
    return  df 


if __name__ == "__main__":


    path_csv=r"/home/pkumar/code/auto_eval/train_data_sorted_with_title1.csv"
    path=r"/home/pkumar/code/auto_eval/visual_features_faster_rcnn_more.csv"
    path1=r"/home/pkumar/code/auto_eval/visual_features_transformer.csv"
    df=pd.read_csv(path_csv)  
    max_length=102  
    df=crop(df,max_length)
    
    pos_path="pos_tag_seq.pkl"
    outputs=get_output(pos_path)
    nlp = spacy.load('en_core_web_sm')
    print()
    print() 
    
    CLIP = 1
    SEED = 1234
    NF=5000
    BATCH_SIZE = 8 #16
    pos_embed_dim=64
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    ##added
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    log_writer = SummaryWriter('TBlogs/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    ###dataloader 
    
    train_data,val_data,test_data,SRC,TRG=splitting(tokenize_text)
    train_iterator,valid_iterator,test_iterator=dataloader_train_val(train_data,val_data,test_data)
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    HID_DIM = 64
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 128
    DEC_PF_DIM = 128


    ####declaring encoder, decoder, and the model
    enc = Encoder(INPUT_DIM,  HID_DIM,  ENC_LAYERS, ENC_HEADS,  ENC_PF_DIM, ENC_DROPOUT,  device,max_length)
    dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS,  DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device,max_length)
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)     
    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    ### declaring the optimizer 
    optimizer = optim.Adam(model.parameters(),lr=0.00001,eps=1e-08)
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


    EPOCHS = 1     
    EP_INT = 1
    start_epoch=0
    
    train_losses,valid_losses,train_ppls,val_ppls=interval_train(model,optimizer,start_epoch,"chpk.txt")
    
    save_losses(train_losses,"train_losses.txt")
    save_losses(valid_losses,"valid_losses.txt")
    save_losses(val_ppls,"valid_ppls.txt")
    save_losses(train_ppls,"train_ppls.txt")
    
    base_path="/home/pkumar/code/auto_eval/data/"
    ##quantitative and qualitative evaluation
    qual_eval(model,optimizer,start_epoch,"chpk.txt",path,base_path,outputs)
    quant_eval(model,optimizer,start_epoch,"chpk.txt")



    
    
    
    
    
