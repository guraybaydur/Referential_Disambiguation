# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import re
import numpy as np
import pandas as pd
import inflect
import pyinflect
import spacy
import en_core_web_sm

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

file_name = "./datasets/training_set.csv"
file_name2 = "./datasets/test.csv"
file_name3 = "./datasets/training_set.xlsx"

# Construction 2
from spacy.lang.en import English

#nlp = en_core_web_sm.load()
# spacy.load('en_core_web_sm')

# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
#tokenizer = nlp.Defaults.create_tokenizer(nlp)




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

def get_plurality_vector(sentence):
    doc = nlp(sentence)
    words_with_tags = [(w.text, w.tag_) for w in doc]
    is_plural = []
    for word, tag in words_with_tags:
        if 'NNS' in tag:
            is_plural.append(1)
            print(word + " is plural")
        else:
            is_plural.append(0)

    # print(words_with_tags[:,0])
    print(is_plural)

    return np.array(is_plural)

def get_pos_vector(sentence):
    doc = nlp(sentence)

    words_with_tags = [(w.text, w.tag_) for w in doc]
    is_plural = []
    #for word, tag in words_with_tags:


    # print(words_with_tags[:,0])
    print(is_plural)

    return np.array(is_plural)

def generate_two_word_pairs(sentence,pronouns):
    pairs = []
    #tokenizer = Tokenizer(nlp.vocab)
    tokens = tokenizer(sentence)

    plural_arr = get_plurality_vector(sentence)
    print(plural_arr)
    # doc = nlp(sentence)
    # words_with_tags = [(w.text, w.tag_) for w in doc]
    # is_plural = []
    # for word, tag in words_with_tags:
    #     if 'NNS' in tag:
    #         is_plural.append(1)
    #         print(word + " is plural")
    #     else:
    #         is_plural.append(0)
    #
    # #print(words_with_tags[:,0])
    # print(is_plural)


    sentence = [token.orth_ for token in tokens if not token.is_punct | token.is_space]



    for token in sentence.copy():
        if "referential" in token:
            sentence.remove(token)
        elif "<" in token:
            sentence.remove(token)
        elif ">" in token:
            sentence.remove(token)

    for candidate_index in range(len(sentence)):
        candidate = sentence[candidate_index]
        if candidate not in pronouns:
            for pronoun in pronouns:
                if pronoun != candidate:
                    pairs.append([candidate,pronoun])

    return pairs


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    nlp = en_core_web_sm.load()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    data = pd.read_excel(file_name3)

    data = data.to_numpy()
    data = data[:, 1]

    pronouns = set()
    sentences = []
    regex = r">[^\s]*<"

    for sentence in data:
        result = re.findall(regex, sentence)
        #print(result)
        for res in result:
            result = res.replace('>', '')
            result = result.replace('<', '')
            #print(result)
            pronouns.add(result)

    pronouns = np.array(list(pronouns))
    print(pronouns)
    for sentence in data:
        pronouns = find_pronouns(sentence)
        print(generate_two_word_pairs(sentence, pronouns))

    #sentence2 = "In this case, the security device alerts the driver if the link has failed or if <referential>it</referential> is cancelled."
    #pronouns = find_pronouns(sentence2)

    #print(generate_two_word_pairs(sentence2,pronouns))

