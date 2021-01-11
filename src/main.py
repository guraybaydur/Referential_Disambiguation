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
file_name4 = "./datasets/detection_answers_file.xlsx"


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

    pronouns_with_indices = np.array(pronouns_with_indices)

    df = pd.DataFrame(data=pronouns_with_indices, columns=['WORD', 'START_INDEX'])
    return df

# # takes sentence with referential tags as input, returns sentence without referential tags
def convert_to_proper_sentence(sentence):
    soup = BeautifulSoup(sentence, features="html.parser")

    for referential in soup.select('referential'):
        referential.unwrap()

    return str(soup)


# takes sentence with referential tags removed, returns word and its start indices
def get_words_with_start_indices(sentence):
    doc = nlp(sentence)
    #words_with_tags = [(w.text, index) for index, w in enumerate(doc) if not w.is_punct | w.is_space]

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

            word_with_start_indices = [m.start(0) for m in re.finditer(search_regex, sentence)]
            words_with_start_indices.append([[w.text, int(start_index)] for start_index in word_with_start_indices])

    final_words_with_start_indices = []
    for elem in words_with_start_indices:
        for item in elem:
            if not item in final_words_with_start_indices:
                final_words_with_start_indices.append(item)

    final_words_with_start_indices = np.array(final_words_with_start_indices)
    final_words_with_start_indices = final_words_with_start_indices[final_words_with_start_indices[:, 1].astype('int').argsort()]

    df = pd.DataFrame(data=final_words_with_start_indices,columns=['WORD','START_INDEX'])
    #print(df)
    return df


# takes sentence with referential tags removed, returns plurality of each word (singular 0, plural 1)
def get_plurality_of_words_in_sentence(sentence):
    doc = nlp(sentence)
    words_with_tags = [(w.text, w.tag_) for w in doc]
    is_plural = []
    for word, tag in words_with_tags:
        if 'NNS' in tag:
            is_plural.append(1)
            # print(word + " is plural")
        elif 'PRP' in tag:
            if word in ['they','them','theirs','their']:
                is_plural.append(1)
        else:
            is_plural.append(0)

    # print(words_with_tags[:,0])
    # print(is_plural)
    is_plural = np.array(is_plural)

    df = pd.DataFrame(data=is_plural, columns=['IS_PLURAL'])
    return df


# takes sentence with referential tags removed, returns pos tags of each word
def get_pos_tags_of_words_in_sentence(sentence):
    doc = nlp(sentence)
    # sentence = [token.orth_ for token in tokens if not token.is_punct | token.is_space]
    words_with_tags = [(w.text, w.tag_, index,w.dep_) for index, w in enumerate(doc) if not w.is_punct | w.is_space]

    pos_vector = [[i[1],i[3]] for i in words_with_tags]
    pos_vector = np.array(pos_vector)

    df = pd.DataFrame(data=pos_vector, columns=['POS_TAG','DEP'])
    return df


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

    words_with_tags = [(w.text, w.pos_, index) for index, w in enumerate(doc)]

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



def rule_based_ambiguity_detection(sentence):
    doc = nlp(sentence)
    words_with_tags = [(w.text, w.tag_, index) for index, w in enumerate(doc)]
    if words_with_tags[0][1] == 'PRP':
        return "ambigous"

    return words_with_tags

def combine_dataframes(sentence):
    proper_sentence = convert_to_proper_sentence(sentence)

    df = get_words_with_start_indices(proper_sentence)
    df = df.join(get_pos_tags_of_words_in_sentence(proper_sentence))
    df = df.join(get_plurality_of_words_in_sentence(proper_sentence))



    return df


def calculate_feature_vector(pronoun_df,candidate_df):
    print(pronoun_df)
    x1_both_plural = (pronoun_df.iloc[0]['IS_PLURAL'] == candidate_df.iloc[0]['IS_PLURAL'])
    x2_both_same_tag = (pronoun_df.iloc[0]['DEP'] == candidate_df.iloc[0]['DEP'])
    x3_closeness =  abs(int(pronoun_df.iloc[0]['START_INDEX']) - int(candidate_df.iloc[0]['START_INDEX']))

    return ""

    #df.loc[(df['WORD'] == 'it') & (df['START_INDEX'] == '57')]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nlp = en_core_web_sm.load()
    merge_nps = nlp.create_pipe("merge_noun_chunks")
    nlp.add_pipe(merge_nps)
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

    #tokenizer = nlp.Defaults.create_tokenizer(nlp)
    data = pd.read_excel(file_name3)

    data = data.to_numpy()
    data = data[:, 1]
    for sentence in data:
        print(sentence)
        result = combine_dataframes(sentence)
        print(result)

        candidate = res = result.iloc[[0]]
        pronoun = res = result.iloc[[8]]
        calculate_feature_vector(candidate,pronoun)
        print(res)
        #print(find_all_pronouns_in_sentence(sentence))
        #proper_sentence = convert_to_proper_sentence(sentence)
        #print(get_words_with_start_indices(proper_sentence))
        ##print(len(get_words_with_start_indices(proper_sentence)))
        #print(find_all_pronouns_in_sentence(sentence))
        ##print(len(find_all_pronouns_in_sentence(sentence)))
        #print(get_pos_tags_of_words_in_sentence(proper_sentence))
        ##print(len(get_pos_tags_of_words_in_sentence(proper_sentence)))
        #print(get_plurality_of_words_in_sentence(proper_sentence))




################# ATTEMPT TO DETECT DISAMBIGUITY START ####################################
    '''
    test = "Only <referential>They</referential> shall display and allow modification of all database tables with the exception of log tables."
    proper_test = convert_to_proper_sentence(test)
    print(rule_based_ambiguity_detection(proper_test))


    data2 = pd.read_excel(file_name4)
    data2 = data2.to_numpy()
    #print(data2)
    #print(data)

    result = []
    sum_of_ambigous = 0
    sum_of_antecedent_ambiguos = 0
    sum_of_pronoun_ambigous = 0
    sum_of_unambigous = 0
    sum_of_antecedent_unambiguos = 0
    sum_of_pronoun_unambigous = 0
    or_and_counts_ambigous = 0
    or_and_counts_unambigous = 0
    for index in range(len(data)):

        if data2[index][1] == "AMBIGUOUS":

            sentence = data[index]
            proper_sentence = convert_to_proper_sentence(sentence)
            pronoun_count = len(find_all_pronouns_in_sentence(sentence))
            doc = nlp(sentence)
            noun_count = 0
            pronounList = []
            pronounIndexList = []
            stringToken = []
            pronounIndex = 0
            for token in doc:
                stringToken.append(token.text)
                if token.pos_ == 'PRON':
                    pronounList.append(token.text)
                    pronounIndexList.append(pronounIndex)
                pronounIndex = pronounIndex + 1

            last_index = 0
            for i in pronounIndexList:
                sum_of_pronoun_ambigous += 1
                string = " "
                for j in range(0, i - 1):
                    string = string + stringToken[j] + " "
                docTest = nlp(string)
                #print("------Noun Chunks------")
                for token in docTest.noun_chunks:
                    sum_of_antecedent_ambiguos += 1
                    #print(token.text)
                #print("------Pronoun Chunks------")
                #print(stringToken[i])
                last_index = j

            pronounList.clear()
            pronounIndexList.clear()
            stringToken.clear()

            for token in doc.noun_chunks:
                noun_count+=1
            #print(proper_sentence)
            #print('noun_count: ' +  str(noun_count))
            #print('pronoun_count: ' +  str(pronoun_count))

            ratio = pronoun_count/noun_count

            sum_of_ambigous += ratio
            #print(ratio)

            result.append([data2[index][1], data[index]])

            if proper_sentence.find("and", 0, len(proper_sentence)) != -1: #or proper_sentence.find("and", 0, len(proper_sentence)) != -1:
                or_and_counts_ambigous += 1
        else:
            sentence = data[index]
            proper_sentence = convert_to_proper_sentence(sentence)
            pronoun_count = len(find_all_pronouns_in_sentence(sentence))
            doc = nlp(proper_sentence)
            noun_count = 0
            pronounList = []
            pronounIndexList = []
            stringToken = []
            pronounIndex = 0
            for token in doc:
                stringToken.append(token.text)
                if token.pos_ == 'PRON':
                    pronounList.append(token.text)
                    pronounIndexList.append(pronounIndex)
                pronounIndex = pronounIndex + 1

            last_index = 0
            for i in pronounIndexList:
                sum_of_pronoun_unambigous += 1
                string = " "
                for j in range(0, i - 1):
                    string = string + stringToken[j] + " "
                docTest = nlp(string)
                #print("------Noun Chunks------")
                for token in docTest.noun_chunks:
                    sum_of_antecedent_unambiguos += 1
                    #print(token.text)
                #print("------Pronoun Chunks------")
                #print(stringToken[i])
                last_index = j

            pronounList.clear()
            pronounIndexList.clear()
            stringToken.clear()

            for token in doc.noun_chunks:
                noun_count += 1
            print(proper_sentence)
            print('noun_count: ' + str(noun_count))
            print('pronoun_count: ' + str(pronoun_count))

            ratio = pronoun_count / noun_count

            sum_of_unambigous += ratio
            print(ratio)

            result.append([data2[index][1], data[index]])

            if proper_sentence.find("and",0,len(proper_sentence)) != -1:
                    #or proper_sentence.find("and",0,len(proper_sentence)) != -1:
                or_and_counts_unambigous += 1

    print("average ratio of ambigous: " + str(sum_of_ambigous/64))
    print("average ratio of unambigous: " + str(sum_of_unambigous/66))

    print("or_and_counts_ambigous: " + str(or_and_counts_ambigous))
    print("or_and_counts_unambigous:  " + str(or_and_counts_unambigous))

    print("potential antecedent ratio ambigous/unambigous: " + str(sum_of_antecedent_ambiguos / sum_of_antecedent_unambiguos))
    print("pronoun ratio ambigous/unambigous: " + str(sum_of_pronoun_ambigous / sum_of_pronoun_unambigous))
    print("sum_of_pronoun_ambigous: " + str(sum_of_pronoun_ambigous))
    print("sum_of_pronoun_unambigous: " + str(sum_of_pronoun_unambigous))
    print("sum_of_antecedent_ambiguos: " + str(sum_of_antecedent_ambiguos))
    print("sum_of_antecedent_unambiguos: " + str(sum_of_antecedent_unambiguos))
    # print(np.array(result))
    # print(len(result))
    # print(len(data2))
    # print(combined)
