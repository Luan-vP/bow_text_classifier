import random
from pathlib import Path

import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

model_dir = Path(__file__).parent.parent.parent / "models"


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


def perform_inference(model, sentence, word_to_index, tag_to_index, device):
    """
    Perform inference on the trained BoW model.

    Args:
        model (torch.nn.Module): The trained BoW model.
        sentence (str): The input sentence for inference.
        word_to_index (dict): A dictionary mapping words to their indices.
        tag_to_index (dict): A dictionary mapping tags to their indices.
        device (str): "cuda" or "cpu" based on availability.

    Returns:
        str: The predicted class/tag for the input sentence.
    """
    # Preprocess the input sentence to match the model's input format
    sentence_tensor = sentence_to_tensor(sentence, word_to_index)

    # Move the input tensor to the same device as the model
    sentence_tensor = sentence_tensor.to(device)

    # Make sure the model is in evaluation mode and on the correct device
    model.eval()
    model.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(sentence_tensor)

    # Move the output tensor to CPU if it's on CUDA
    if device == "cuda":
        output = output.cpu()

    # Convert the model's output to a predicted class/tag
    predicted_class = torch.argmax(output).item()

    # Reverse lookup to get the tag corresponding to the predicted class
    for tag, index in tag_to_index.items():
        if index == predicted_class:
            return tag

    # Return an error message if the tag is not found
    return "Tag not found"


def save_model(model, path):
    """
    Save the trained BoW model to a file.

    Args:
        model (torch.nn.Module): The trained BoW model.
        path (str): The file path to save the model to.
    """
    torch.save(model.state_dict(), path)


# TODO Integrate architecture params
def load_model(model, path):
    """
    Load a trained BoW model from a file.

    Args:
        model (torch.nn.Module): The BoW model architecture.
        path (str): The file path to load the model from.
    """
    model.load_state_dict(torch.load(path))
