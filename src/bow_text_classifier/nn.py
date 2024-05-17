import random

import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


# create a simple neural network with embedding layer, bias, and xavier initialization
class BoW(torch.nn.Module):
    def __init__(self, nwords, ntags):
        super(BoW, self).__init__()
        self.Embedding = nn.Embedding(nwords, ntags)
        nn.init.xavier_uniform_(self.Embedding.weight)

        type = (
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        )
        self.bias = torch.zeros(ntags, requires_grad=True).type(type)

    def forward(self, x):
        emb = self.Embedding(x)
        out = torch.sum(emb, dim=0) + self.bias
        out = out.view(1, -1)
        return out


# function to convert sentence into tensor using word_to_index dictionary
def sentence_to_tensor(sentence, word_to_index):
    return torch.LongTensor(
        [
            word_to_index[_word] if _word in word_to_index else word_to_index["<unk>"]
            for _word in sentence.split(" ")
        ]
    )


# perform training of the Bow model
def train_bow(model, optimizer, criterion, train_data, test_data, type):
    for ITER in range(10):
        # perform training
        model.train()
        random.shuffle(train_data)
        total_loss = 0.0
        train_correct = 0
        for sentence, tag in train_data:
            sentence = torch.tensor(sentence).type(type)
            tag = torch.tensor([tag]).type(type)
            output = model(sentence)
            predicted = torch.argmax(output.data.detach()).item()

            loss = criterion(output, tag)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if predicted == tag:
                train_correct += 1
        # perform testing of the model
        model.eval()
        test_correct = 0
        for _sentence, tag in test_data:
            _sentence = torch.tensor(_sentence).type(type)
            output = model(_sentence)
            predicted = torch.argmax(output.data.detach()).item()
            if predicted == tag:
                test_correct += 1

        # print model performance results
        log = (
            f"ITER: {ITER+1} | "
            f"train loss/sent: {total_loss/len(train_data):.4f} | "
            f"train accuracy: {train_correct/len(train_data):.4f} | "
            f"test accuracy: {test_correct/len(test_data):.4f}"
        )
        print(log)
