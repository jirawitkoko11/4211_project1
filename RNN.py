from utils import *

##Base model
class Base_RNN(nn.Module):
  def __init__(self, n_vocab, embedding_dim = 50, n_hidden = 64, n_layers = 1, dropout = 0.1):
    super(Base_RNN, self).__init__()
    self.emb = nn.Embedding(n_vocab,embedding_dim)
    self.rnn = nn.RNN(
        input_size=embedding_dim,
        hidden_size = n_hidden,
        num_layers = n_layers,
        batch_first = True
    )
    self.drop = nn.Dropout(dropout)
    self.fc = nn.Linear(64,3)
    self.sigmoid = nn.Sigmoid()
  def forward(self,sent, sent_len):
    #shape [bacth, seq_len]
    sent_emb = self.emb(sent)
    #shape [bacth, seq_len, emb_dim]

    packed_embedded = nn.utils.rnn.pack_padded_sequence(sent_emb, sent_len,batch_first=True)
    outputs, hidden = self.rnn(sent_emb)
    outputs = self.drop(hidden.squeeze())
    outputs = self.fc(outputs)

    return outputs


##Improved model
class Base_GRU(nn.Module):
  def __init__(self, n_vocab, embedding_dim = 50, n_hidden = 64, n_layers = 1, dropout = 0.1):
    super(Base_GRU, self).__init__()
    self.emb = nn.Embedding(n_vocab,embedding_dim)
    self.rnn = nn.GRU(
        input_size=embedding_dim,
        hidden_size = n_hidden,
        num_layers = n_layers,
        batch_first = True
    )
    self.drop = nn.Dropout(dropout)
    self.fc = nn.Linear(64,3)
    self.sigmoid = nn.Sigmoid()
  def forward(self,sent, sent_len):
    #shape [bacth, seq_len]
    sent_emb = self.emb(sent)
    #shape [bacth, seq_len, emb_dim]

    packed_embedded = nn.utils.rnn.pack_padded_sequence(sent_emb, sent_len,batch_first=True)
    outputs, hidden = self.rnn(sent_emb)
    outputs = self.drop(hidden.squeeze())
    outputs = self.fc(outputs)
    # print(outputs)
    # return self.sigmoid(outputs)
    return outputs


##Improved model
class Base_LSTM(nn.Module):
  def __init__(self, n_vocab, embedding_dim = 50, n_hidden = 64, n_layers = 1, dropout = 0.1):
    super(Base_LSTM, self).__init__()
    self.emb = nn.Embedding(n_vocab,embedding_dim)
    self.rnn = nn.LSTM(
        input_size=embedding_dim,
        hidden_size = n_hidden,
        num_layers = n_layers,
        batch_first = True
    )
    self.drop = nn.Dropout(dropout)
    self.fc = nn.Linear(64,3)
    self.sigmoid = nn.Sigmoid()
  def forward(self,sent, sent_len):
    #shape [bacth, seq_len]
    sent_emb = self.emb(sent)
    #shape [bacth, seq_len, emb_dim]

    packed_embedded = nn.utils.rnn.pack_padded_sequence(sent_emb, sent_len,batch_first=True)
    outputs, (hidden, cell) = self.rnn(sent_emb)
    outputs = self.drop(hidden.squeeze())
    outputs = self.fc(outputs)
    # print(outputs)
    # return self.sigmoid(outputs)
    return outputs


class LSTM_bidirectional(nn.Module):
  def __init__(self, n_vocab, embedding_dim = 50, n_hidden = 64, n_layers = 1, dropout = 0.1):
    super(LSTM_bidirectional, self).__init__()
    self.emb = nn.Embedding(n_vocab,embedding_dim)
    self.rnn = nn.LSTM(
        input_size=embedding_dim,
        hidden_size = n_hidden,
        num_layers = n_layers,
        batch_first = True,
        bidirectional = True
    )
    self.drop = nn.Dropout(dropout)
    self.fc = nn.Linear(64,3)

  def forward(self,sent, sent_len):
    #shape [bacth, seq_len]
    sent_emb = self.emb(sent)
    #shape [bacth, seq_len, emb_dim]

    packed_embedded = nn.utils.rnn.pack_padded_sequence(sent_emb, sent_len,batch_first=True)
    outputs, (hidden, cell) = self.rnn(sent_emb)
    _1,_2 = torch.split(hidden,1,dim=0)
    hidden = torch.add(_1,_2)*(1/2)
    outputs = self.drop(hidden.squeeze())
    outputs = self.fc(outputs)
    # print(outputs)
    # return self.sigmoid(outputs)
    return outputs


##Improved model
class GRU_xavier(nn.Module):
  def __init__(self, n_vocab, embedding_dim = 50, n_hidden = 64, n_layers = 1, dropout = 0.1):
    super(GRU_xavier, self).__init__()
    self.emb = nn.Embedding(n_vocab,embedding_dim)
    nn.init.xavier_uniform_(self.emb.weight)
    self.rnn = nn.GRU(
        input_size=embedding_dim,
        hidden_size = n_hidden,
        num_layers = n_layers,
        batch_first = True
    )
    self.drop = nn.Dropout(dropout)
    self.fc = nn.Linear(64,3)
    
  def forward(self,sent, sent_len):
    #shape [bacth, seq_len]
    sent_emb = self.emb(sent)
    #shape [bacth, seq_len, emb_dim]

    packed_embedded = nn.utils.rnn.pack_padded_sequence(sent_emb, sent_len,batch_first=True)
    outputs, hidden = self.rnn(sent_emb)
    outputs = self.drop(hidden.squeeze())
    outputs = self.fc(outputs)
    # print(outputs)
    # return self.sigmoid(outputs)
    return outputs


##Improved model
class GRU_he(nn.Module):
  def __init__(self, n_vocab, embedding_dim = 50, n_hidden = 64, n_layers = 1, dropout = 0.1):
    super(GRU_he, self).__init__()
    self.emb = nn.Embedding(n_vocab,embedding_dim)
    nn.init.kaiming_uniform_(self.emb.weight)
    self.rnn = nn.GRU(
        input_size=embedding_dim,
        hidden_size = n_hidden,
        num_layers = n_layers,
        batch_first = True
    )
    self.drop = nn.Dropout(dropout)
    self.fc = nn.Linear(64,3)
    nn.init.kaiming_normal_(self.emb.weight)
    nn.init.kaiming_normal_(self.rnn.weight_hh_l0)
    nn.init.kaiming_normal_(self.rnn.weight_ih_l0)
    nn.init.kaiming_normal_(self.fc.weight)
  def forward(self,sent, sent_len):
    #shape [bacth, seq_len]
    sent_emb = self.emb(sent)
    #shape [bacth, seq_len, emb_dim]

    packed_embedded = nn.utils.rnn.pack_padded_sequence(sent_emb, sent_len,batch_first=True)
    outputs, hidden = self.rnn(sent_emb)
    outputs = self.drop(hidden.squeeze())
    outputs = self.fc(outputs)
    # print(outputs)
    # return self.sigmoid(outputs)
    return outputs

#glove.twitter.27B.50d, requires_grad=False
txt_field.build_vocab(train, min_freq=2, vectors='glove.twitter.27B.50d')
pretrained_embeddings = txt_field.vocab.vectors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = GRU_xavier(len(txt_field.vocab)).to(device)
model.emb.weight.data.copy_(nn.Parameter(pretrained_embeddings, requires_grad=False))
n_epochs = 100
val_loss = None
save_name = 'GRU_xavier_twitter50d_F.pt'
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), weight_decay=0.0005)
emb1L,ebd1VL = TRAIN(model, train_iter, val_iter,  n_epochs, criterion, optimizer, val_loss, device, save_name)


#glove.twitter.27B.50d, requires_grad=True
txt_field.build_vocab(train, min_freq=2, vectors='glove.twitter.27B.50d')
pretrained_embeddings = txt_field.vocab.vectors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = GRU_xavier(len(txt_field.vocab)).to(device)
model.emb.weight.data.copy_(nn.Parameter(pretrained_embeddings, requires_grad=True))
n_epochs = 100
val_loss = None
save_name = 'GRU_xavier_twitter50d_T.pt'
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), weight_decay=0.0005)
emb2L,ebd2VL = TRAIN(model, train_iter, val_iter,  n_epochs, criterion, optimizer, val_loss, device, save_name)


#glove.6B.50d, requires_grad=False
txt_field.build_vocab(train, min_freq=2, vectors='glove.6B.50d')
pretrained_embeddings = txt_field.vocab.vectors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = GRU_xavier(len(txt_field.vocab)).to(device)
model.emb.weight.data.copy_(nn.Parameter(pretrained_embeddings, requires_grad=False))
n_epochs = 100
val_loss = None
save_name = 'GRU_xavier_6B50d.pt'
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), weight_decay=0.0005)
emb3L,ebd3VL = TRAIN(model, train_iter, val_iter,  n_epochs, criterion, optimizer, val_loss, device, save_name)


#glove.6B.50d, requires_grad=True
txt_field.build_vocab(train, min_freq=2, vectors='glove.6B.50d')
pretrained_embeddings = txt_field.vocab.vectors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = GRU_xavier(len(txt_field.vocab)).to(device)
model.emb.weight.data.copy_(nn.Parameter(pretrained_embeddings, requires_grad=True))
n_epochs = 100
val_loss = None
save_name = 'GRU_xavier_6B50d.pt'
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), weight_decay=0.0005)
emb4L,ebd4VL = TRAIN(model, train_iter, val_iter,  n_epochs, criterion, optimizer, val_loss, device, save_name)


## plot graph 
epoch = range(0, 100)
plt.plot(epoch,emb1L,label = "1's train loss")
plt.plot(epoch,ebd1VL, label = "1's validation loss")
plt.plot(epoch,emb2L,label = "2's train loss")
plt.plot(epoch,ebd2VL, label = "2's validation loss")
plt.plot(epoch,emb3L,label = "3's train loss")
plt.plot(epoch,ebd3VL, label = "3's validation loss")
plt.plot(epoch,emb4L,label = "4's train loss")
plt.plot(epoch,ebd4VL, label = "4's validation loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('train vs validation loss')
plt.legend()
plt.savefig('emb.png',dpi=300)
plt.show()
plt.close()