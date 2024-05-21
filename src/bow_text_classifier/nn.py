import json
import random
from logging import Logger
from pathlib import Path

import torch
import torch.nn as nn

logger = Logger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_dir = Path(__file__).parent.parent.parent / "models"


class BoW_Classifier:
    def __init__(self):
        self._model = None
        self.word_to_index = None
        self.tag_to_index = None

    def __call__(self, input):
        return self.predict(input)

    def load_model(self, model_package_dir):
        if self._model is None:
            with open(model_package_dir / "word_to_index.json", "r") as f:
                self.word_to_index = json.load(f)
            with open(model_package_dir / "tag_to_index.json", "r") as f:
                self.tag_to_index = json.load(f)

            self._model = _BoW(len(self.word_to_index), len(self.tag_to_index))
            self._model.load_state_dict(torch.load(model_package_dir / "model.pth"))
            self._model.eval()

    def predict(self, input):
        if self._model is None:
            raise RuntimeError("You must load the model before making a prediction")
        output = perform_inference(
            self._model, input, self.word_to_index, self.tag_to_index, device
        )
        return output


# create a simple neural network with embedding layer, bias, and xavier initialization
class _BoW(torch.nn.Module):
    def __init__(self, nwords, ntags):
        super(_BoW, self).__init__()
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
def train_bow(
    model: _BoW,
    optimizer,
    criterion,
    train_data: list[tuple[str, str]],
    test_data: list[tuple[str, str]],
    type: torch.cuda.LongTensor | torch.LongTensor,
):
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


def save_model(model, word_to_index, tag_to_index, model_package_path: Path):
    """
    Save the trained BoW model to a file.

    Args:
        model (torch.nn.Module): The trained BoW model.
        path (str): The file path to save the model to.
    """
    model_package_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_package_path / "model.pth")

    with open(model_package_path / "word_to_index.json", "w") as f:
        json.dump(word_to_index, f)
    with open(model_package_path / "tag_to_index.json", "w") as f:
        json.dump(tag_to_index, f)

    logger.info(f"Model saved to {model_package_path}")


def load_model(model, model_package_dir):
    """
    Load a trained BoW model from a file.

    Args:
        model (torch.nn.Module): The BoW model architecture.
        path (str): The file path to load the model from.
    """

    model.load_state_dict(torch.load(model_package_dir / "model.pth"))
