import sys

from bow_text_classifier import model

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_sentence = sys.argv[1]
        print(model(input_sentence))
    else:
        print("Please provide an input sentence as a command line argument.")
