import numpy
import random
import re
import sentiment_rnn
import simple_lstm
import synthetic_languages
import torch
from torch import nn
from torch.utils import data as torch_data

def pad_and_clip_data(sentences: list[list[int]], seq_len: int) -> numpy.ndarray:
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
            features[ii, -len(review):] = numpy.array(review)[:seq_len] + 1
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

# Here, we load our training data from a CSV file
parity_language = synthetic_languages.parity(64)

x_train = []
y_train = []
x_test = []
y_test = []

# Generate training examples
for i in range(10000):
    result = synthetic_languages.sample_one(
        task=parity_language,
        length=random.randrange(64),
    )
    if result == None: break
    data, label = result

    data = data.tolist()
    label = label.item()
    x_train.append(data)
    y_train.append(label)

# Generate test examples
for i in range(100):
    result = synthetic_languages.sample_one(
        task=parity_language,
        length=random.randrange(64),
    )
    if result == None: break
    data, label = result

    data = data.tolist()
    label = label.item()
    x_test.append(data)
    y_test.append(label)

# print(f'shape of train data is {x_train.shape}')
# print(f'shape of test data is {x_test.shape}')

x_train_pad = pad_and_clip_data(x_train, 64)
x_test_pad = pad_and_clip_data(x_test, 64)

x_train_pad = torch.from_numpy(x_train_pad)
x_test_pad = torch.from_numpy(x_test_pad)

x_train_pad = nn.functional.one_hot(x_train_pad).float()
x_test_pad = nn.functional.one_hot(x_test_pad).float()

batch_size = 50
train_data = torch_data.TensorDataset(x_train_pad, torch.tensor(y_train))
valid_data = torch_data.TensorDataset(x_test_pad, torch.tensor(y_test))
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
vocab_size = 3 #extra 1 for padding
embedding_dim = 64
output_dim = 1
hidden_dim = 256

# model = sentiment_rnn.SentimentRNN(
#     no_layers=no_layers,
#     vocab_size=vocab_size,
#     hidden_dim=hidden_dim,
#     embedding_dim=embedding_dim,
#     output_dim=output_dim,
#     drop_prob=0.5
# ).to(device)

model = simple_lstm.SimpleLSTM(
    input_dim=vocab_size,
    hidden_dim=64,
    output_dim=2,
    drop_prob=0.5
).to(device)

print(model)

# Begin training our model
lr=0.001

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# function to predict accuracy
def calculate_accuracy(pred, label, batch_size):
    assert pred.shape[0] == batch_size
    pred = torch.argmax(pred, dim=1)
    return torch.sum(pred == label).item() / batch_size

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
    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)   
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        
        model.zero_grad()
        output = model(inputs)
        
        # calculate the loss and perform backprop
        loss = criterion(output, labels)
        loss.backward()
        train_losses.append(loss.item())
        # calculating accuracy
        accuracy = calculate_accuracy(output, labels, batch_size)
        train_acc += accuracy
        print(f"Training Accuracy: {accuracy * 100:.2f}%")
        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
 
    val_losses = []
    val_acc = 0.0
    model.eval()
    for inputs, labels in valid_loader:

        inputs, labels = inputs.to(device), labels.to(device)

        output = model(inputs)
        val_loss = criterion(output, labels)

        val_losses.append(val_loss.item())
        
        accuracy = calculate_accuracy(output, labels, batch_size)
        val_acc += accuracy
            
    epoch_train_loss = numpy.mean(train_losses)
    epoch_val_loss = numpy.mean(val_losses)
    epoch_train_acc = train_acc/len(train_loader)
    epoch_val_acc = val_acc/len(valid_loader)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print('='*80)
    print(f'Epoch {epoch+1}') 
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
    if epoch_val_loss <= valid_loss_min:
        # torch.save(model.state_dict(), '../working/state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
        valid_loss_min = epoch_val_loss
    print(80*'=')

# Prediction

def predict_text(string):
        word_seq = numpy.array(string)
        word_seq = numpy.expand_dims(word_seq,axis=0)
        word_seq = pad_and_clip_data(word_seq,64)
        word_seq = torch.from_numpy(word_seq)
        word_seq = nn.functional.one_hot(word_seq).float()
        word_seq = word_seq.to(device)
        assert word_seq.shape == (1, 64, 3)
        output = model(word_seq)
        assert output.shape == (1, 2)
        # Since output is an array of logits, we need to convert it into
        # probabilities
        output = torch.softmax(output, 1)
        return(output[0, 1])

def print_prediction(text, label):
    print("="*80)
    print("Test Case:")
    print()
    print(text)
    print()
    print(f"Actual parity is:   {label}")
    print()
    probability = predict_text(text)
    status = "odd" if probability > 0.5 else "even"
    probability = (1 - probability) if status == "even" else probability
    print(f"Predicted parity is {status} with a probability of {probability}")
    print("="*80)
    
print_prediction([0, 1, 0, 0, 1], "even")
print_prediction([0, 1, 0, 0, 0], "odd")
print_prediction([1, 0, 0, 1], "even")
print_prediction([1, 0, 0, 0], "odd")
