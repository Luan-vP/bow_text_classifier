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
    return torch.LongTensor([word_to_index[_word] for _word in sentence.split(" ")])
