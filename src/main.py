# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import re

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import inflect
import pyinflect
import spacy
import en_core_web_sm
import datetime
from sklearn.preprocessing import LabelEncoder

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

file_name = "./datasets/training_set.csv"
file_name2 = "./datasets/test.csv"
file_name3 = "./datasets/training_set.xlsx"

# Construction 2
from spacy.lang.en import English


# nlp = en_core_web_sm.load()
# spacy.load('en_core_web_sm')

# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
# tokenizer = nlp.Defaults.create_tokenizer(nlp)


def find_pronouns(sentence):
    pronouns = set()
    regex = r">[^\s]*<"

    result = re.findall(regex, sentence)

    for res in result:
        result = res.replace('>', '')
        result = result.replace('<', '')
        # print(result)
        pronouns.add(result)

    pronouns = np.array(list(pronouns))

    return pronouns


# takes sentence with referential tags as input, returns pronouns and their start indices in the sentence where referential tags are removed
def find_all_pronouns_in_sentence(sentence):
    pronouns = []
    soup = BeautifulSoup(sentence, features="html.parser")
    for elem in soup.findAll("referential"):
        pronoun = elem.renderContents().decode("utf-8")
        #print(pronoun)
        pronouns.append(pronoun)

    proper_sentence = convert_to_proper_sentence(sentence)

    pronouns_with_indices = []
    for pronoun_index in range(len(pronouns)):
        searched_pronoun = ' ' + pronouns[pronoun_index]
        found_index = find_nth(proper_sentence, searched_pronoun, pronoun_index + 1)
        pronouns_with_indices.append([pronouns[pronoun_index], found_index+1])

    return np.array(pronouns_with_indices)


def convert_to_proper_sentence(sentence):
    soup = BeautifulSoup(sentence, features="html.parser")

    for referential in soup.select('referential'):
        referential.unwrap()

    return str(soup)


# takes sentence with referential tags removed, returns word and its start indices
def get_words_with_start_indices(sentence):
    doc = nlp(sentence)
    words_with_tags = [(w.text, index) for index, w in enumerate(doc) if not w.is_punct | w.is_space]

    #x = [token.orth_ for token in tokens if not token.is_punct | token.is_space]

    words_with_start_indices = []
    for w in doc:
        if not w.is_punct | w.is_space:
            escaped = w.text.translate(str.maketrans({"-": r"\-",
                                                        "]": r"\]",
                                                        "[": r"\[",
                                                        "\\": r"\\",
                                                        "^": r"\^",
                                                        "$": r"\$",
                                                        "*": r"\*",
                                                        ".": r"\.",
                                                        "(": r"\(",
                                                        ")": r"\)",
                                                      }))

            search_regex = r"\b{0}\b".format(escaped)

            #word_with_start_indices = [m.start(0) for m in re.finditer('\b'+w.text+'\b', sentence)]
            word_with_start_indices = [m.start(0) for m in re.finditer(search_regex, sentence)]
            words_with_start_indices.append([[w.text, int(start_index)] for start_index in word_with_start_indices])

    final_words_with_start_indices = []
    for elem in words_with_start_indices:
        for item in elem:
            if not item in final_words_with_start_indices:
                final_words_with_start_indices.append(item)

    # regex =
    # urls = [(m.start(0), m.end(0)) for m in re.finditer(regex, document)]
    final_words_with_start_indices = np.array(final_words_with_start_indices)
    final_words_with_start_indices = final_words_with_start_indices[final_words_with_start_indices[:, 1].astype('int').argsort()]
    return final_words_with_start_indices


# takes sentence with referential tags removed, returns plurality of each word (singular 0, plural 1)
def get_plurality_of_words_in_sentence(sentence):
    doc = nlp(sentence)
    words_with_tags = [(w.text, w.tag_) for w in doc]
    is_plural = []
    for word, tag in words_with_tags:
        if 'NNS' in tag:
            is_plural.append(1)
            # print(word + " is plural")
        else:
            is_plural.append(0)

    # print(words_with_tags[:,0])
    # print(is_plural)

    return np.array(is_plural)


# takes sentence with referential tags removed, returns pos tags of each word
def get_pos_tags_of_words_in_sentence(sentence):
    doc = nlp(sentence)

    words_with_tags = [(w.text, w.tag_, index) for index, w in enumerate(doc)]

    pos_vector = [i[1] for i in words_with_tags]

    return np.array(pos_vector)


def generate_two_word_pairs(sentence, pronouns):
    pairs = []
    # tokenizer = Tokenizer(nlp.vocab)
    tokens = tokenizer(sentence)

    plural_arr = get_plurality_of_words_in_sentence(sentence)
    # print(plural_arr)
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
                    pairs.append([candidate, pronoun])

    return pairs


def generate_two_word_pairs2(sentence, pronouns):
    doc = nlp(sentence)

    words_with_tags = [(w.text, w.tag_, index) for index, w in enumerate(doc)]

    pos_vector = [i[1] for i in words_with_tags]

    return np.array(pos_vector)


def remove_referential_tags_from_sentence(sentence):
    print(sentence)
    # regex = r"<.*>"
    # test = re.match(regex, sentence)
    # print(test)
    m = re.search(r"<.*>", sentence)
    print(m.group())
    # result = re.sub('<.*>', '', sentence)
    # print(result)

    return ""


def get_closeness_vector(sentence):
    return ""


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nlp = en_core_web_sm.load()
    '''
        tokenizer = nlp.Defaults.create_tokenizer(nlp)
        data = pd.read_excel(file_name3)
    
        data = data.to_numpy()
        data = data[:, 1]
    
        pronouns = set()
        sentences = []
        regex = r">[^\s]*<"
    
        for sentence in data:
            result = re.findall(regex, sentence)
            # print(result)
            for res in result:
                result = res.replace('>', '')
                result = result.replace('<', '')
                # print(result)
                pronouns.add(result)
    
        pronouns = np.array(list(pronouns))
        #print(pronouns)
        for sentence in data:
            pronouns = find_pronouns(sentence)
            #print(generate_two_word_pairs(sentence, pronouns))
            result = generate_two_word_pairs(sentence, pronouns)
    '''

    #sentence2 = "Init this case, the security device alerts the driver if the link has failed or if <referential id=a>it</referential> is cancelled."
    #sentence3 = 'This function receives an AIP request that identifies the requested AIP(s) and provides <referential id=a">them</referential> on the requested media type or transfers <referential id="b">them</referential> to a staging area."'
    # soup = BeautifulSoup(sentence3, features="html.parser")

    # for referential in soup.select('referential'):
    #    referential.unwrap()
    # print(sentence3)
    # print(convert_to_proper_sentence(sentence3))

    # print(soup)

    # print(str(soup))

    # for elem in soup.findAll("referential"):
    #    t = elem.renderContents().decode("utf-8")
    #    print(t)

    # remove_referential_tags_from_sentence(sentence2)

    # pronouns = find_pronouns(sentence2)

    # print(generate_two_word_pairs(sentence2,pronouns))

    # columns = ['POS_TAG', 'NUMERIC']
    #
    # df_ = pd.DataFrame(columns=columns)
    #
    # all_pos_tags_serie = pd.Series(all_pos_tags)
    # df_['POS_TAG'] = all_pos_tags_serie
    #
    # LE = LabelEncoder()
    # print(LE.fit_transform(all_pos_tags_serie))
    # df_['NUMERIC'] = LE.fit_transform(all_pos_tags_serie)
    # print(df_)

    # proper_sentence = convert_to_proper_sentence(sentence3)
    # print(get_pos_tags_of_words_in_sentence(proper_sentence))
    # print(len(get_pos_tags_of_words_in_sentence(proper_sentence)))
    # print(get_plurality_of_words_in_sentence(proper_sentence))
    # print(len(get_plurality_of_words_in_sentence(proper_sentence)))
    # # print(list(nlp.tokenizer.vocab.morphology.tag_map.keys()))
    # all_pos_tags = list(nlp.tokenizer.vocab.morphology.tag_map.keys())
    #
    # proper_sentence2 = convert_to_proper_sentence(sentence2)
    # print(proper_sentence2)
    # print(find_nth(proper_sentence2, ' it', 0))
    # print(find_all_pronouns_in_sentence(sentence3))

    #sentence4 = "<referential>It</referential> may also receive a report request from Access and provides descriptive information for a specific AIP."
    #proper_sentence4 = convert_to_proper_sentence(sentence4)

    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    data = pd.read_excel(file_name3)

    data = data.to_numpy()
    data = data[:, 1]
    for sentence in data:
        print(sentence)
        proper_sentence = convert_to_proper_sentence(sentence)
        print(get_words_with_start_indices(proper_sentence))
        print(find_all_pronouns_in_sentence(sentence))

