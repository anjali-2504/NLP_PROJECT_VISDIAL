##pip install nltk
##pip install tensorboardX
##path_to_Req.txt="requirements.txt"
##pip install -r "requirements.txt"
#from nltk.tokenize import PunktSentenceTokenizer
#import nltk
from collections import OrderedDict

#from torch.types import Device
#nltk.download('state_union')
#from nltk.corpus import state_union
import pandas as pd
import torch, torchvision
import pickle
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
#!pip install torchtext==0.11.0
from torchtext import  vocab,data
from torchtext.legacy.datasets import TranslationDataset, Multi30k
from torchtext.legacy.data import Field, BucketIterator, TabularDataset   
from torchvision.transforms import ToTensor
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, csv, sys, random, re, time, math, spacy
from PIL import Image
from numpy.random import RandomState
from tensorboardX import SummaryWriter
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torchtext
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')  




def make_new_columns(df):
      df['S_No']=[i for i in range(len(df))]
##    df['tags_trg']=[cs_toknizer.tokenize(sent) for sent in df['trg']]  ###do for 'tags_Src' as well just for initilaising 
##    df['tags_src']=[cs_toknizer.tokenize(sent) for sent in df['src']]
##
##    for i in range(len(df)):
##      sent=cs_toknizer.tokenize(df['src'][i])
##      ay=[]
##      for s in sent:
##        a=nltk.pos_tag(nltk.word_tokenize(s))
##        a.append(('<eos>','<PAD>'))
##        a.insert(0,('<sos>','<UNK>'))
##        ay.append(a)
##
##      df['tags_src'][i]=ay
##    
##    for i in range(len(df)):
##      sent=cs_toknizer.tokenize(df['trg'][i])
##      ay=[]
##      
##      for s in sent:
##        a=nltk.pos_tag(nltk.word_tokenize(s))
##        a.append(('<eos>','<PAD>'))
##        a.insert(0,('<sos>','<UNK>'))
##        ay.append(a)
##     
##      df['tags_trg'][i]=ay
##    
##    df['pos_tag_seq']=[get_pos_tag_index_seq([x[1] for x in df['tags_src'][i][0]],len(df['tags_src'][i][0])) for i in range(len(df))]
##
      return df

def build_tags(lines):
    pos_vocab = OrderedDict()
    pos_vocab['<PAD>'] = 0
    pos_vocab['<UNK>'] = 1
    k = 2
    for line in lines:
        
        for tag in line[0]:
            if tag[1] not in pos_vocab:
                pos_vocab[tag[1]] = k
                k += 1
    return pos_vocab

##def get_pos_tag_index_seq(pos_seq, max_len):
##    seq = list()
##    for t in pos_seq:
##        if t in pos_vocab:
##            seq.append(pos_vocab[t])
##        else:
##            seq.append(pos_vocab['<UNK>'])
##    pad_len = max_len - len(seq)
##    for i in range(0, pad_len):
##        seq.append(pos_vocab['<PAD>'])
##    return seq

def get_max_len(sample_batch):
    src_max_len = len(sample_batch.src[0])
    for idx in range(1, len(sample_batch)):
        if len(sample_batch.src[0]) > src_max_len:
            src_max_len = len(sample_batch[idx].SrcWords)
    return src_max_len




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



def get_pos_tag_seq_batch(batch,batch_len,outputs):
          pos_tag_seq=[]
          max_len=len(outputs[batch.S_No[0].item()])
          for i in range(batch_len):
            if len(outputs[batch.S_No[i].item()]) > max_len:
              max_len=len(outputs[batch.S_No[i].item()])
         ## print("max len {}".format(max_len))
          for i in range(batch_len):
            seq=outputs[batch.S_No[i].item()]
            pad_len=max_len-len(outputs[batch.S_No[i].item()])
            for j in range(0, pad_len):
              seq.append(0)
            pos_tag_seq.append(seq)

          return pos_tag_seq




class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        
        self.embedding = nn.Embedding(input_dim, emb_dim) #no dropout as only one layer!

        self.drop_rate=0.3

        self.pos_embeddings=POSEmbeddings(50, pos_embed_dim, self.drop_rate)
        
        self.rnn = nn.GRU(2*emb_dim, hid_dim)
        
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.vis_fc=nn.Linear(NF, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, visual_features,pos_tag_seq):
        
        vis_features= torch.tanh(self.vis_fc(visual_features))   #src = [src len, batch size]l
        
        embedded = self.dropout(self.embedding(src))
        pos_embeds = self.pos_embeddings(pos_tag_seq)

        if(pos_embeds.size()==embedded.size()):
           embedded=torch.cat((embedded, pos_embeds), -1)
           print("fine")
        else :
           embedded=torch.cat((embedded, embedded), -1)
           print("yes sizes not matching")
           print(pos_embeds.size(),embedded.size())

        
        #embedded = [src len, batch size, emb dim]
        ##word_inputs=torch.concat(embedded,)
        
        encoder_states, text_hidden = self.rnn(embedded) #no cell state!
        
        #outputs = [src len, batch size, hid dim * n directions]
        #text_hidden = [n layers * n directions, batch size, hid dim]
        vis_features=vis_features.unsqueeze(0)
        hidden = torch.tanh(self.fc(torch.cat((text_hidden,vis_features), 2)))
        
        #outputs are always from the top hidden layer
        
        return encoder_states,hidden

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
        
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        
        self.attention = Attention(hid_dim)
        
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_states):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #context = [n layers * n directions, batch size, hid dim]
        #n layers and n directions in the decoder will both always be 1, therefore:
        #hidden = [1, batch size, hid dim]
        #context = [1, batch size, hid dim]
        seq_len = encoder_states.shape[0]
        input = input.unsqueeze(0)
        #input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, emb dim]  
        
        
        attention = self.attention(hidden, encoder_states)
      
        attention = attention.permute(1,2,0)
        encoder_states = encoder_states.permute(1,0,2)
        context = torch.bmm(attention, encoder_states).permute(1,0,2)
        rnn_input = torch.cat((embedded, context), dim=2)   
        output, hidden = self.rnn(rnn_input, hidden)
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #seq len, n layers and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), 
                           dim = 1)
        #output = [batch size, emb dim + hid dim * 2]
        prediction = self.fc_out(output)
        #prediction = [batch size, output dim]
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, visual_features, trg,pos_tag_seq ,teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is the context
        encoder_states, hidden = self.encoder(src,visual_features,pos_tag_seq)
        
        #context also used as the initial hidden state of the decoder
        # hidden = context
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state and the context state
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_states)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


def tokenize_text(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in nlp.tokenizer(text)]


def splitting(tokenize_text):

    ID = torchtext.legacy.data.Field(sequential=False,use_vocab=False)
    S_no = torchtext.legacy.data.Field(sequential=False,use_vocab=False)
    
    SRC = Field(tokenize = tokenize_text, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True,
                include_lengths = True)
    TRG = Field(tokenize = tokenize_text, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True,
                )
    
    datafields=[('src', SRC), ('trg', TRG),('img_id',ID),('img_path',None),('S_No',S_no)]
    
    #rng = RandomState()
    #train_data = df.sample(frac=0.75, random_state=rng)
    #val_test_data = df.loc[~df.index.isin(train_data.index)]
    #val_data=val_test_data.sample(frac=0.15, random_state=rng)
    #test_data=val_test_data.loc[~val_test_data.index.isin(val_data.index)]
    ##
    ###cols= ["src", "trg","img_id","img_path","pos_tag_seq"]
    ##
    #cols= ["src", "trg","img_id","img_path","S_No"]
    ##
    ##
    #train_data.to_csv('train_data.csv', index= False,columns=cols) #columns=cols
    #val_data.to_csv('val_data.csv', index= False,columns=cols) #columns=cols
    #test_data.to_csv('test_data.csv', index= False,columns=cols)

    train_data, val_data ,test_data= torchtext.legacy.data.TabularDataset.splits(path=r"",train="train_data.csv", validation="val_data.csv",test="test_data.csv" , format='csv', skip_header=True, fields=datafields)
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=1)
    return train_data,val_data,test_data,SRC,TRG

def dataloader_train_val(train_data,val_data,test_data):


    train_iterator, x_iterator = BucketIterator.splits(
        (train_data, train_data), 
        batch_size = BATCH_SIZE,
        sort_within_batch = True,
        sort_key = lambda x : len(x.src), 
        device = device)
    
    valid_iterator, y_iterator = BucketIterator.splits(
        (val_data, val_data), 
    
        batch_size = BATCH_SIZE, 
        sort_within_batch = True,
        sort_key = lambda x : len(x.src),
        device = device)

    test_iterator, z_iterator = BucketIterator.splits(
    (test_data, test_data), 

    batch_size = BATCH_SIZE, 
    sort_within_batch = True,
    sort_key = lambda x : len(x.src),
    device = device)    
    return train_iterator ,valid_iterator ,test_iterator

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, iterator, optimizer, criterion, clip,path,device,outputs):
    
    model.train()
    
    epoch_loss = 0

    Batch_num=len(iterator)
    
    for i, batch in enumerate(tqdm(iterator)):
        
        src,src_len = batch.src
        batch_len=list(batch.S_No.size())[0]

        trg = batch.trg
        img_id=batch.img_id
        
        x=img_id.cpu().numpy()
        y=len(x)
        #visual_features=torch.empty(y,4096).cuda()
        visual_features=torch.empty(y,NF).to(device) ##.cuda()
        df1=pd.read_csv(path)
        for i in range(y):
            q=df1[str(x[i]-1)].to_numpy()
            a=torch.from_numpy(q).unsqueeze(0)
            visual_features[i]=a
        
        optimizer.zero_grad()
        pos_tag_seq=get_pos_tag_seq_batch(batch,batch_len,outputs)

        pos_tag_seq=torch.tensor(pos_tag_seq).permute(1,0).to(device)
        if(pos_tag_seq.size()!=src.size()):
          print("yes")
           

        #pos_tag_seq=torch.LongTensor(pos_tag_seq).unsqueeze(1).to(device)

        output = model(src.to(device),visual_features, trg,pos_tag_seq)
           
           #trg = [trg len, batch size]
           #output = [trg len, batch size, output dim]
           
        output_dim = output.shape[-1]
           
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)           
           #trg = [(trg len - 1) * batch size]
           #output = [(trg len - 1) * batch size, output dim]

           
        loss.backward()
           
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
           
        optimizer.step()
           
        epoch_loss += loss.item()

        #else :
         #  Batch_num-=1   
        
    return epoch_loss /Batch_num


def evaluate(model, iterator, criterion,path,device,outputs):
    
    model.eval()
    
    epoch_loss = 0
    Batch_num=len(iterator)
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src,src_len = batch.src
            trg = batch.trg
            batch_len=list(batch.S_No.size())[0]

            img_id=batch.img_id
        
            x=img_id.cpu().numpy()
            y=len(x)
            #visual_features=torch.empty(y,4096).cuda()
            visual_features=torch.empty(y,NF).to(device)
            df1=pd.read_csv(path)
            for i in range(y):
                q=df1[str(x[i]-1)].to_numpy()       ### should it be -1 or not
                a=torch.from_numpy(q).unsqueeze(0)
                visual_features[i]=a
            

            pos_tag_seq=get_pos_tag_seq_batch(batch,batch_len,outputs)

            pos_tag_seq=torch.tensor(pos_tag_seq).permute(1,0).to(device)

            #if(pos_tag_seq.size()==src.size()):

            output = model(src.to(device), visual_features, trg,pos_tag_seq, 0) #turn off teacher forcing
     
                 #trg = [trg len, batch size]
                 #output = [trg len, batch size, output dim]
     
            output_dim = output.shape[-1]
                 
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
     
                 #trg = [(trg len - 1) * batch size]
                 #output = [(trg len - 1) * batch size, output dim]
     
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            #else :
               # Batch_num-=1

    return epoch_loss / Batch_num

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


def translate_sentence(sentence, src_field, trg_field, img_id, path, model, device,pos_tag_seq, max_len = 50):

    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load("en_core_web_sm")
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    pos_tag_seq = torch.LongTensor(pos_tag_seq).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)]).to(device)

    df1=pd.read_csv(path)
    visual_features=torch.empty(1,NF).to(device)
#    visual_features=torch.empty(1,4096).cuda()
    q=df1[str(img_id-1)].to_numpy()
    a=torch.from_numpy(q).unsqueeze(0)
    visual_features[0]=a


    with torch.no_grad():
        encoder_states, hidden = model.encoder(src_tensor,visual_features,pos_tag_seq)#, src_len)
        
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    
    for i in tqdm(range(max_len)):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)          
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_states)

        pred_token = output.argmax(1).item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:]

def plot_results(losses,epochs):
    plt.plot(np.arange(epochs),losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('loss'+str(epochs)+'.png')
    #3plt.show()
    #plt.plot(np.arange(epochs),valid_losses)

def train_loop(epochs,model,train_iterator,optimizer,criterion,CLIP,losses,path,device):
    for epoch in range(epochs):
       loss=train(model, train_iterator, optimizer, criterion, CLIP,path,device)
       losses.append(loss)
       #print(loss)
    return losses
def build_pos_vocab(df):
    return build_tags(df['tags_src'])

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
    #s_meteor=meteor()#
    
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


def qual_eval(model, optimizer, start_epoch, chpt_file,path,base_path,outputs):
    model, optimizer, start_epoch = load_checkpt(model, optimizer, chpt_file)
    T_EPOCHS = start_epoch + EP_INT
    
    # Save the predicted Feedbacks in CSV file
    # Log with Tensorboard: Text, Comment, Image and Feedback
    test_pred=[]
    test_df=pd.read_csv("test_data.csv")  
    #test_df=pd.read_csv("_data.csv") 
    #path=r"visual_features_resnet.csv"      
    length=len(test_df)
    images = []

    

    #print(length)
    for i in range(length):
        #src = vars(val_data.examples[i])['src']
        #trg = vars(val_data.examples[i])['trg']
        src=test_df['src'][i]
        trg=test_df['trg'][i]
        img_path=base_path+test_df['img_path'][i]
        image_idx=test_df['img_id'][i]
        pos_tag_seq=torch.tensor(outputs[test_df['S_No'][i]])

        ##print(pos_tag_seq.size(),len(src),i)

        translation = translate_sentence(src, SRC, TRG, int(image_idx), path, model, device,pos_tag_seq)

        if not translation:
            translation="*empty*"

        #Untokenization    
        translation1=translation[0:(len(translation)-1)]    
        translation2 = TreebankWordDetokenizer().detokenize(translation1)
        print(translation2)
        test_pred.append(str(translation2))

        image = Image.open(img_path)
        image = ToTensor()(image)   
        #images.append(image)
        
        #print(str(T_EPOCHS))
        if (i%10==0): 
            #Log with Tensorboard: Text, Comment, Image and Feedback
            log_writer.add_text(str(T_EPOCHS)+'Ep=>Ground-truth Comment of Sample/'+str(i+1), str(trg))
            log_writer.add_text(str(T_EPOCHS)+'Ep=>News Text of Sample/'+str(i+1), str(src))#, i+1)
            log_writer.add_text(str(T_EPOCHS)+'Ep=>Predicted Feedback of Sample/'+str(i+1), str(translation))
            #log_writer.add_image('Image', image, i+1)
            #image_grid = torchvision.utils.make_grid(images)
            log_writer.add_image(str(T_EPOCHS)+'Ep:Image of Sample/'+str(i+1), image)        

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
   # model.to(device) 
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

        #train_loss = train(model, train_iterator, optimizer, criterion, CLIP)#, log_writer_train)
        train_loss =train(model, train_iterator, optimizer, criterion, CLIP,path,device,outputs)
        valid_loss = evaluate(model,valid_iterator,criterion,path,device,outputs)#, log_writer_val)
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

        state = {'epoch': T_EPOCHS, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': train_loss, 'valid_loss':valid_loss}
        torch.save(state, chpt_file)

        print(f'\nEpoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    torch.cuda.empty_cache()
    return train_losses,val_losses,train_ppls,val_ppls


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
    #s_meteor=meteor()#
    
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

def get_output():
    dict_file = open("pos_tag_seq.pkl", "rb")
    outputs = pickle.load(dict_file)
    dict_file.close() 
    return outputs

def save_losses(array,filename):
    file = open(filename, "a") 
    file.write('\n') # append mode
    for i in array:
        file.write(str(i))
        file.write('\n')
    file.close()


if __name__ == "__main__":


    path_csv=r"train_data_sorted_with_title1.csv"
    path=r"/home/pkumar/code/auto_eval/visual_features_nasnet.csv"
    df=pd.read_csv(path_csv)
    df=make_new_columns(df)

    outputs=get_output()
    nlp = spacy.load('en_core_web_sm')

    
    CLIP = 1
    SEED = 1234
    NF=1000
    BATCH_SIZE = 16 #16
    pos_embed_dim=64
    ENC_HID_DIM = 125
    ENC_EMB_DIM = 64
    DEC_EMB_DIM = 64
    HID_DIM = 128
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
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
    
    

    ####declaring encoder, decoder, and the model
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)     
    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')  

    ### declaring the optimizer 
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


    EPOCHS = 1     
    EP_INT = 1
    start_epoch=0
    
    train_losses,valid_losses,train_ppls,val_ppls=interval_train(model,optimizer,start_epoch,"chpk.txt")
    print(train_losses)
    print(valid_losses)
    print(train_ppls)
    print(val_ppls)
    
    save_losses(train_losses,"train_losses.txt")
    save_losses(train_ppls,"train_ppls.txt")
    save_losses(val_ppls,"valid_ppls.txt")
    save_losses(valid_losses,"valid_losses.txt")
    
    base_path="/home/pkumar/code/auto_eval/data/"
    qual_eval(model,optimizer,start_epoch,"chpk.txt",path,base_path,outputs)
    quant_eval(model,optimizer,start_epoch,"chpk.txt")
    
    
   
