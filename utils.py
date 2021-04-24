import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import nltk
nltk.download('punkt')
from nltk import word_tokenize

import math
from sklearn.metrics import accuracy_score, confusion_matrix ,classification_report
import re
import numpy as np
import torch.nn.functional as F
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
import random
SEED = 4211
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



def TRAIN(model, train_loader, valid_loader, num_epochs, criterion, optimizer, val_loss, device, save_name):
  ### to be used in ploting graph ####
  train_loss_arr = []
  val_loss_arr = []
  epoch_arr = []
  ### to be used in ploting graph ###
  best_val_acc = float("-Inf")
  if val_loss==None:
    best_val_loss = float("Inf")
  else:
    best_val_loss= val_loss
    print("Resume training")
  
  model.to(device)

  for epoch in range(num_epochs):
    epoch_arr.append(epoch+1)

    model.train()
    running_loss = 0.0
    for batches in train_loader:
      labels = batches.Sentiment.to(device)
      texts,text_len = batches.text
      texts,text_len = texts.to(device),text_len
      labels = labels.view(-1).to(device)

      ## Training
      # Forward pass2
      outputs = model(texts,text_len).squeeze()
      loss = criterion(outputs, labels)

      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    with torch.no_grad():
      running_corrects = 0.0
      model.eval()
      running_loss = 0.0
      for batch in valid_loader:
        labels = batch.Sentiment
        (texts,text_len) = batch.text
        texts,text_len = texts.to(device),text_len
        labels = labels.view(-1).to(device)

        # Forward pass1
        outputs = model(texts,text_len).squeeze()
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        # get accuracy (this will store # of correct cases, and will be used to find the accuracy later)
        running_corrects += get_corrects(outputs,labels)

      val_loss = running_loss/ len(valid_loader)
      val_acc = running_corrects / 3211
      
      train_loss_arr.append(train_loss)
      val_loss_arr.append(val_loss)


      print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Validation accuracy {:.4f}' 
                  .format(epoch+1, num_epochs, train_loss, val_loss, val_acc)) ## val_acc

      if val_loss < best_val_loss :
        best_val_loss = val_loss
        

      if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(save_name, model, optimizer, best_val_acc)

      
  plotLoss(train_loss_arr,val_loss_arr,epoch_arr,'Loss.png',best_val_acc)
  print('Finished Training')
  return train_loss_arr,val_loss_arr

def clean_text(text):
  #change text to lowercase
  text = text.lower()
  #remove tag
  text=re.sub('<.*?>','',text)
  #remove mentions
  text=re.sub(r'@\w+','',text)
  #remove hashtag
  text=re.sub(r'#\w+','',text)
  #remove new line
  text=re.sub('\n','',text)
  #remove links
  text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',text)
  #remove numbers
  text=re.sub(r'[0-9]+','',text)
  #remove punctuation
  text = re.sub(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*','',text)
  
  text=re.sub('\s{2,}', ' ', text.strip())
  text=re.sub(r'\bamp\b|\bthi\b|\bha\b',' ',text)
  return text

def save_checkpoint(save_path, model, optimizer, val_acc):
    if save_path==None:
        return
    save_path = save_path 
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_acc': val_acc}

    torch.save(state_dict, save_path)
    print(f'Model saved to {save_path}')

def load_checkpoint(save_path, model, optimizer):
    save_path = save_path 
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_acc = state_dict['val_acc']
    print(f'Model loaded from {save_path}, with val acc: {val_acc}')
    return val_acc

def get_corrects(outputs,labels):
  correct = 0
  pred = torch.argmax(outputs,dim=1)
  for i in range(len(outputs)):
    if pred[i] == labels[i]:
      correct += 1
  return correct

def plotLoss(train,val,epoch,save_path,best_acc):
  plt.plot(epoch,train,label = "train loss")
  plt.plot(epoch,val, label = "validation loss")
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.title('train vs validation loss\n best accuracy is %.3f' %(best_acc))
  plt.legend()
  plt.savefig(PATH+'/'+save_path,dpi=300)

PATH = '/content/drive/MyDrive/4211/project1'
df = pd.read_csv(PATH + '/TheSocialDilemma.csv')
df['Sentiment'] = df['Sentiment'].map({'Negative':0, 'Neutral':1, 'Positive':2})
df = df[['text','Sentiment']]

df['text'] = df['text'].apply(lambda x: clean_text(x))

df = df[df['text'].map(len) >4]

df['text'].replace('', np.nan, inplace=True)
df.dropna(subset=['text'], inplace=True)
print(df.isnull().sum())


# split train, test, val
negs = df.loc[df['Sentiment'] == 0]
neus = df.loc[df['Sentiment'] == 1]
poss = df.loc[df['Sentiment'] == 2]

train_negs,test_negs = train_test_split(negs,test_size = 500, random_state = 4211)
train_neus,test_neus = train_test_split(neus,test_size = 500, random_state = 4211)
train_poss,test_poss = train_test_split(poss,test_size = 500, random_state = 4211)

train_df = pd.concat([train_negs,train_neus,train_poss],axis=0,ignore_index=True)
test_df = pd.concat([test_negs,test_neus,test_poss],axis=0,ignore_index=True)

negs = train_df.loc[train_df['Sentiment'] == 0]
neus = train_df.loc[train_df['Sentiment'] == 1]
poss = train_df.loc[train_df['Sentiment'] == 2]

train_negs,val_negs = train_test_split(negs,test_size = 1000, random_state = 4211)
train_neus,val_neus = train_test_split(neus,test_size = 1000, random_state = 4211)
train_poss,val_poss = train_test_split(poss,test_size = 1000, random_state = 4211)

train_df = pd.concat([train_negs,train_neus,train_poss],axis=0,ignore_index=True)
val_df = pd.concat([val_negs,val_neus,val_poss],axis=0,ignore_index=True)

val_df.to_csv(PATH+'/validation_df.csv',index=False)
train_df.to_csv(PATH+'/train_df.csv',index=False)
test_df.to_csv(PATH+'/test_df.csv',index=False)
#################################################################

#create dataset and dataloader
txt_field = Field(tokenize=word_tokenize, lower=True, batch_first=True, include_lengths=True)
label_field = Field(sequential=False, use_vocab=False, batch_first=True)

train, val, test= TabularDataset.splits(path=PATH, train='train_df.csv', validation='validation_df.csv', test = 'test_df.csv',format='csv', 
                                  fields=[('text', txt_field), ('Sentiment', label_field)], skip_header=True)

# get the number of existing vocabs in the training set
txt_field.build_vocab(train, min_freq=0)
all_vocab = len(txt_field.vocab)

# get the number of vocab with min_freq = 2
txt_field.build_vocab(train, min_freq=2)
min_2_voacab = len(txt_field.vocab)
print('The number of OOV is %d' %( all_vocab-min_2_voacab ))

# create iterators
train_iter, val_iter = BucketIterator.splits((train, val), batch_size= 512, sort_key=lambda x: len(x.text),sort_within_batch=True)
test_iter = BucketIterator(test,batch_size=1, train=False,sort=False)