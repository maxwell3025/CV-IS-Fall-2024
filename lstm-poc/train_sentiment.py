import collections
import nltk
from nltk import corpus as nltk_corpus
import numpy
import pandas
import re
import sentiment_rnn
from sklearn import model_selection
import torch
from torch import nn
from torch.utils import data as torch_data

def pad_and_clip_data(sentences: numpy.ndarray, seq_len: int):
    """Pad and clip a ragged matrix of tokens to a given length.

    This function homogenizes dimension 1, so we expect the shape of the
    input tensor to be (num_sequences, ?), and we will output a tensor with the
    shape (num_sequences, seq_len).
    We also pad with zeros.

    Args:
        sentences: A rank-2 numpy tensor representing out tokens.
        seq_len: An integer representing our desired length.

    Returns:
        Our original tensor after being padded and clipped
    """
    features = numpy.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = numpy.array(review)[:seq_len]
    return features

def preprocess_string(s: str):
    """Formats a string as a concrete paragraph.

    Args:
        s: The string that we want to format.

    Returns:
        The formatted string.
    """
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def tokenize(x_train, y_train, x_val, y_val):
    """Turns the data into Python lists"""
    nltk.download("stopwords")
    word_list = []

    stop_words = set(nltk_corpus.stopwords.words("english")) 
    for sentence in x_train:
        for word in sentence.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != "":
                word_list.append(word)
  
    corpus = collections.Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
    
    # tokenize
    train_tokens, test_tokens = [],[]
    for sentence in x_train:
            train_tokens.append([onehot_dict[preprocess_string(word)] for word in sentence.lower().split() 
                                     if preprocess_string(word) in onehot_dict.keys()])
    for sentence in x_val:
            test_tokens.append([onehot_dict[preprocess_string(word)] for word in sentence.lower().split() 
                                    if preprocess_string(word) in onehot_dict.keys()])
            
    train_labels = [1 if label =='positive' else 0 for label in y_train]  
    test_labels = [1 if label =='positive' else 0 for label in y_val] 
    return (
        train_tokens,
        train_labels,
        test_tokens,
        test_labels,
        onehot_dict
    )

# Here, we load our training data from a CSV file
base_csv = './data/IMDB Dataset.csv'
df = pandas.read_csv(base_csv)

X, y = df['review'].values, df['sentiment'].values
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    X, y, stratify=y
)

print(f'shape of train data is {x_train.shape}')
print(f'shape of test data is {x_test.shape}')

x_train, y_train, x_test, y_test, vocab = tokenize(
    x_train,
    y_train,
    x_test,
    y_test,
)

x_train_pad = pad_and_clip_data(x_train, 500)
x_test_pad = pad_and_clip_data(x_test, 500)

# Here, we move our data from numpy arrays into pytorch DataLoaders
batch_size = 50
train_data = torch_data.TensorDataset(torch.from_numpy(x_train_pad), torch.tensor(y_train))
valid_data = torch_data.TensorDataset(torch.from_numpy(x_test_pad), torch.tensor(y_test))
train_loader = torch_data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = torch_data.DataLoader(valid_data, shuffle=True, batch_size=batch_size)

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device
# variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# Model Hyperparameters
no_layers = 2
vocab_size = len(vocab) + 1 #extra 1 for padding
embedding_dim = 64
output_dim = 1
hidden_dim = 256

model = sentiment_rnn.SentimentRNN(
    no_layers=no_layers,
    vocab_size=vocab_size,
    hidden_dim=hidden_dim,
    embedding_dim=embedding_dim,
    output_dim=output_dim,
    drop_prob=0.5
).to(device)

print(model)

# Begin training our model
lr=0.001

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# function to predict accuracy
def calculate_accuracy(pred, label, batch_size):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item() / batch_size

clip = 5
epochs = 5 
valid_loss_min = numpy.Inf
# train for some number of epochs
epoch_tr_loss, epoch_vl_loss = [], []
epoch_tr_acc, epoch_vl_acc = [], []

for epoch in range(epochs):
    train_losses = []
    train_acc = 0.0
    model.train()
    # initialize hidden state 
    h = model.init_hidden(batch_size)
    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)   
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = model.init_hidden(batch_size)
        
        model.zero_grad()
        output,h = model(inputs,h)
        
        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        train_losses.append(loss.item())
        # calculating accuracy
        accuracy = calculate_accuracy(output,labels, batch_size)
        train_acc += accuracy
        print(f"Training Accuracy: {accuracy * 100:.2f}%")
        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
 
    val_h = model.init_hidden(batch_size)
    val_losses = []
    val_acc = 0.0
    model.eval()
    for inputs, labels in valid_loader:
        val_h = tuple([each.data for each in val_h])

        inputs, labels = inputs.to(device), labels.to(device)

        output, val_h = model(inputs, val_h)
        val_loss = criterion(output.squeeze(), labels.float())

        val_losses.append(val_loss.item())
        
        accuracy = calculate_accuracy(output,labels, batch_size)
        val_acc += accuracy
            
    epoch_train_loss = numpy.mean(train_losses)
    epoch_val_loss = numpy.mean(val_losses)
    epoch_train_acc = train_acc/len(train_loader)
    epoch_val_acc = val_acc/len(valid_loader)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(f'Epoch {epoch+1}') 
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
    if epoch_val_loss <= valid_loss_min:
        # torch.save(model.state_dict(), '../working/state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
        valid_loss_min = epoch_val_loss
    print(80*'=')

# Prediction

def predict_text(text):
        word_seq = numpy.array([vocab[preprocess_string(word)] for word in text.split() 
                         if preprocess_string(word) in vocab.keys()])
        word_seq = numpy.expand_dims(word_seq,axis=0)
        pad =  torch.from_numpy(pad_and_clip_data(word_seq,500))
        inputs = pad.to(device)
        batch_size = 1
        h = model.init_hidden(batch_size)
        h = tuple([each.data for each in h])
        output, h = model(inputs, h)
        return(output.item())

def print_prediction(text, label):
    print(text)
    print('='*80)
    print(f'Actual sentiment is  : {label}')
    print('='*80)
    probability = predict_text(text)
    status = "positive" if probability > 0.5 else "negative"
    probability = (1 - probability) if status == "negative" else probability
    print(f'Predicted sentiment is {status} with a probability of {probability}')
    print('='*80)
    
print_prediction(df["review"][30], df["sentiment"][30])
print_prediction(df["review"][32], df["sentiment"][32])
print_prediction("Personally, I thought that this movie was terrible!", "negative")