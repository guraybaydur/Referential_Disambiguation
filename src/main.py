# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import re
import numpy as np
import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

file_name = "./datasets/training_set.csv"
file_name2 = "./datasets/test.csv"
file_name3 = "./datasets/training_set.xlsx"



def find_pronouns(sentence):
    pronouns = set()
    regex = r">[^\s]*<"

    result = re.findall(regex, sentence)

    for res in result:
        result = res.replace('>', '')
        result = result.replace('<', '')
        #print(result)
        pronouns.add(result)

    pronouns = np.array(list(pronouns))

    return pronouns

def generate_two_word_pairs(sentence,pronouns):
    pairs = []
    tokens = tokenizer(sentence)
    sentence = [token.orth_ for token in tokens if not token.is_punct | token.is_space]

    for token in sentence.copy():
        if "referential" in token:
            sentence.remove(token)
        elif "<" in token:
            sentence.remove(token)
        elif ">" in token:
            sentence.remove(token)

    for word in sentence:
        if word not in pronouns:
            for pronoun in pronouns:
                if pronoun != word:
                    pairs.append([word,pronoun])


    return pairs


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #print_hi('PyCharm')
    data = pd.read_excel(file_name3)
    # data = np.genfromtxt(file_name,dtype=str,delimiter=',',skip_header=1)
    #print(type(data))

    data = data.to_numpy()
    data = data[:, 1]

    pronouns = set()
    regex = r">[^\s]*<"
    for sentence in data:
        result = re.findall(regex, sentence)
        #print(result)
        for res in result:
            result = res.replace('>', '')
            result = result.replace('<', '')
            #print(result)
            pronouns.add(result)
        # if result:
        #     result = result.group()
        #     print(result)
        #result = result.replace('<referential>', '')
        #result = result.replace('</referential>', '')
    pronouns = np.array(list(pronouns))
    #print(find_pronouns("In this case, the security device alerts the driver if the link has failed or if <referential>it</referential> is cancelled."))


    # Construction 2
    from spacy.lang.en import English
    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.Defaults.create_tokenizer(nlp)

    tokens = tokenizer("This is a , adsa:dsas sentence.")
    #print([token.orth_ for token in tokens if not token.is_punct | token.is_space] )
    sentence2 = "In this case, the security device alerts the driver if the link has failed or if <referential>it</referential> is cancelled."
    pronouns = find_pronouns(sentence2)

    print(generate_two_word_pairs(sentence2,pronouns))