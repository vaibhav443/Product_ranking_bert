import deep_translator
from deep_translator import GoogleTranslator
import langdetect
from langdetect import detect
import pandas as pd
import nltk

nltk.download('punkt')


# code for translating the dataset into english if needed
def translation(data):
    """

    :param data: data
    :return: transated data
    """
    df_en = pd.DataFrame(data[['description', 'title']])
    for column in df_en.columns:
        for index in df_en.index:
            try:
                if (detect(df_en[column][index]) == 'en'):
                    continue
                else:
                    df_en[column][index] = GoogleTranslator(source='french', target='english').translate(
                        df_en[column][index])
            except:
                print(f'Unable to translate for index: {index}')
                continue
    return df_en


def translation_final(data):
    """

    :param data: data with some values not translated
    :return: fully translated data
    """
    df_en = pd.DataFrame(data[['description', 'title']])
    for column in df_en.columns:
        for index in df_en.index:
            try:
                if (detect(df_en[column][index]) == 'en'):
                    continue
                else:
                    a_list = nltk.tokenize.sent_tokenize(df_en[column][index])
                    for i in range(len(a_list)):
                        a_list[i] = GoogleTranslator(source='french', target='english').translate(a_list[i])
                    df_en[column][index] = " ".join(a_list)
            except:
                print(f'Unable to translate for index: {index}')
                continue
    return df_en


from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import spacy
import unidecode
import string
import nltk
import re
from nltk.corpus import stopwords
import pandas as pd

nltk.download('stopwords')
stop = stopwords.words('english')


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


def basic_preprocessing_exact(Df):
    # removing accented characters
    """

    :param Df: data for exact match
    :return: preprocessed data for exact match
    """
    df = pd.DataFrame(Df, copy=True)
    df['description'] = df['description'].apply(remove_accented_chars)
    df['title'] = df['title'].apply(remove_accented_chars)

    # removing digits and converting into lower case
    df['description'] = df['description'].str.lower().str.replace('\d+', '')
    df['title'] = df['title'].str.lower().str.replace('\d+', '')

    # removing extrawhite spacing
    df['description'] = df['description'].str.replace('\s\s+', ' ')
    df['title'] = df['title'].str.replace('\s\s+', ' ')

    # removing punctuations
    df['description'] = df['description'].str.replace('[^\w\s]', '')
    df['title'] = df['title'].str.replace('[^\w\s]', '')

    # removing stopwords
    df['description'] = df['description'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df['title'] = df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    return df


def basic_preprocessing_ST(Df):
    """

    :param Df: data
    :return: preprocessed data for semantic match
    """
    df = pd.DataFrame(Df, copy=True)
    df['description'] = df['description'].apply(remove_accented_chars)
    df['title'] = df['title'].apply(remove_accented_chars)

    df['description'] = df['description'].str.lower().str.replace('\d+', '')
    df['title'] = df['title'].str.lower().str.replace('\d+', '')

    df['description'] = df['description'].str.replace('\s\s+', ' ')
    df['title'] = df['title'].str.replace('\s\s+', ' ')

    return df


def basic_preprocessing_keyword_exact(keywords):
    """

    :param keywords: keywords list
    :return: preprocessed keywords for exact match
    """
    for i in range(len(keywords)):
        keywords[i] = remove_accented_chars(keywords[i])
        keywords[i] = keywords[i].lower()
        keywords[i] = re.sub(r'\d+', '', keywords[i])
        keywords[i] = re.sub(r'\s\s+', ' ', keywords[i])
        keywords[i] = re.sub(r'[^\w\s]', '', keywords[i])
        keywords[i] = ' '.join([word for word in keywords[i].split() if word not in (stop)])
    return keywords


def basic_preprocessing_keywords_ST(keywords):
    """

    :param keywords: keyword list
    :return: preprocessed keywords list for semantic match
    """
    for i in range(len(keywords)):
        keywords[i] = remove_accented_chars(keywords[i])
        keywords[i] = keywords[i].lower()
        keywords[i] = re.sub(r'\d+', '', keywords[i])
        keywords[i] = re.sub(r'\s\s+', ' ', keywords[i])

    return keywords


import spacy
import rank_bm25
from rank_bm25 import BM25Okapi
import nltk
import operator

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
nltk.download('wordnet')


def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


def exact_match(keyword, data):
    """

    :param keyword: preprocessed keywords for exact match
    :param data: preprocessed data for exact match
    :return: dictionary of scores from exact match
    """
    dict_scores = {}
    scores_title = []
    scores_descrip = []
    lemmatized_descrip = data['description'].apply(lemmatize_text)
    lemmatized_title = data['title'].apply(lemmatize_text)
    lemmatized_descrip = lemmatized_descrip.to_list()
    lemmatized_title = lemmatized_title.to_list()
    lemmatized_keyword = []
    for w in w_tokenizer.tokenize(keyword):
        lemmatized_keyword.append(lemmatizer.lemmatize(w))
    s1 = BM25Okapi(lemmatized_title)
    s2 = BM25Okapi(lemmatized_descrip)
    scores_title = s1.get_scores(lemmatized_keyword)
    scores_descrip = s2.get_scores(lemmatized_keyword)
    weighted_scores = (scores_title * 60 + scores_descrip * 40) / 100

    for i in range(len(weighted_scores)):
        dict_scores[i] = weighted_scores[i]

    return dict_scores


import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer, util
import operator
import pandas as pd

import numpy as np

model = SentenceTransformer('bert-base-nli-mean-tokens')


def embeddings_bert(Df):
    """

    :param Df: preprocessed data for semantic match
    :return: embeddings of title and description using local bert model
    """
    title_text = []
    description_text = []
    for index in Df.index:
        title_text.append(Df['title'][index])
        description_text.append(Df['description'][index])

    embeddings_title = model.encode(title_text)
    embeddings_description = model.encode(description_text)

    return [embeddings_title, embeddings_description]


def get_embeddings_bert(Df, lang):
    """

    :param Df: preprocessed data for semantic match
    :return: embeddings of title and description using server bert model
    """
    title_text = []
    description_text = []
    for index in Df.index:
        title_text.append(Df['title'][index])
        description_text.append(Df['description'][index])

    embeddings_title = embeddings(title_text, True, lang)
    embeddings_description = embeddings(description_text, True, lang)

    return [embeddings_title, embeddings_description]


def scores_bert_server(embeddings_title, embeddings_description, data, keyword, lang):
    """

    :param embeddings_title: embeddings of title from server
    :param embeddings_description: embeddings of description from server
    :param data: preprossed data for semantic match
    :param keyword: preprossed keyword for semantic match
    :return: dictionary of scores by bert using server
    """
    keyword_list = [keyword]
    embeddings_keyword = embeddings(keyword_list, True, lang)
    scores_description = util.pytorch_cos_sim(torch.from_numpy(embeddings_keyword),
                                              torch.from_numpy(embeddings_description))
    scores_title = util.pytorch_cos_sim(torch.from_numpy(embeddings_keyword), torch.from_numpy(embeddings_title))

    Dict_bert = {}
    for i in range(data.shape[0]):
        Dict_bert[i] = ((scores_description[0][i].item()) * 40 + (scores_title[0][i].item()) * 60)

    return Dict_bert


def scores_bert_local(embeddings_title, embeddings_description, data, keyword):
    """

    :param embeddings_title: embeddings of title from local model
    :param embeddings_description: embeddings of descrip from local model
    :param data: preprocessed data for semantic match
    :param keyword: preprocessed keyword for semantic match
    :return: dictionary of scores by bert using local model
    """
    keyword_list = [keyword]
    embeddings_keyword = model.encode(keyword_list)
    scores_description = util.pytorch_cos_sim(embeddings_keyword,
                                              embeddings_description)
    scores_title = util.pytorch_cos_sim(embeddings_keyword, embeddings_title)

    Dict_bert = {}
    for i in range(data.shape[0]):
        Dict_bert[i] = ((scores_description[0][i].item()) * 40 + (scores_title[0][i].item()) * 60)

    return Dict_bert


import sklearn
import operator
import sentence_transformers
from sentence_transformers import util
import torch
import numpy
import tensorflow as tf
import tensorflow_hub as hub

import pandas as pd
import os

os.environ['TFHUB_CACHE_DIR'] = r'C:\Users\vaibh\Downloads\PM1 Internship\Experiments\TFHUB_CACHE_DIR'
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model1 = hub.load(module_url)
print("module %s loaded" % module_url)


def get_embeddings_USE(data, lang):
    """

    :param data: preprossed data for semantic match
    :return: embeddings of title and description using server USE model
    """
    title_text = []
    description_text = []
    for i in data.index:
        title_text.append(data["title"][i])
        description_text.append(data["description"][i])

    embeddings_title_USE = embeddings(title_text, is_bert=False, lang=lang)
    embeddings_description_USE = embeddings(description_text, is_bert=False, lang=lang)
    return [embeddings_title_USE, embeddings_description_USE]


def embeddings_USE(data):
    """

    :param Df: preprossed data for semantic match
    :return: embeddings of title and description using local USE model
    """
    title_text = []
    description_text = []
    for i in data.index:
        title_text.append(data['title'][i])
        description_text.append(data['description'][i])

    embeddings_USE_title = model1(title_text)
    embeddings_USE_description = model1(description_text)

    return [embeddings_USE_title, embeddings_USE_description]


def scores_USE_local(embeddings_title, embeddings_description, data, keyword):
    """

    :param embeddings_title: embeddings of title from local model
    :param embeddings_description: embeddings of description from local model
    :param data: preprocessed data for semantic match
    :param keyword: preprocessed keyword for semantic match
    :return: dictionary of scores by USE using local model
    """
    keyword_list = [keyword]
    # embeddings_keyword = embeddings(keyword_list,False)
    embeddings_keyword = model1(keyword_list)
    scores_description = util.pytorch_cos_sim(embeddings_keyword.numpy(), embeddings_description.numpy())
    scores_title = util.pytorch_cos_sim(embeddings_keyword.numpy(), embeddings_title.numpy())

    Dict_USE = {}
    for i in range(data.shape[0]):
        Dict_USE[i] = ((scores_description[0][i].item()) * 40 + (scores_title[0][i].item()) * 60)

    return Dict_USE


def scores_USE_server(embeddings_title, embeddings_description, data, keyword, lang):
    """

    :param embeddings_title: embeddings of title from server
    :param embeddings_description: embeddings of description from server
    :param data: preprocessed data for semantic match
    :param keyword: preprocessed keyword for semantic match
    :return: dictionary of scores by USE using server
    """
    keyword_list = [keyword]
    embeddings_keyword = embeddings(keyword_list, False, lang=lang)
    # embeddings_keyword = model1(keyword_list)
    scores_description = util.pytorch_cos_sim(torch.from_numpy(embeddings_keyword),
                                              torch.from_numpy(embeddings_description))
    scores_title = util.pytorch_cos_sim(torch.from_numpy(embeddings_keyword), torch.from_numpy(embeddings_title))

    Dict_USE = {}
    for i in range(data.shape[0]):
        Dict_USE[i] = ((scores_description[0][i].item()) * 40 + (scores_title[0][i].item()) * 60)

    return Dict_USE


import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def final_results(dict_exact, dict_bert, dict_USE, N_prod, Df):
    """
    For final results  -
    :param dict_exact:dictionary of scores by exact match
    :param dict_bert: dictionary of scores by bert
    :param dict_USE:dictionary of scores by USE
    :param N_prod: no of products to be shown
    :param Df: original data frame
    :return: DataFrame of results
    """
    dict_res = {}
    data = pd.DataFrame(columns=['bert_scores', 'USE_scores', 'exact_scores'])
    list_bert = [dict_bert[key] for key in dict_bert]
    list_exact = [dict_exact[key] for key in dict_exact]
    list_USE = [dict_USE[key] for key in dict_USE]
    # scaling of data using minmax scaler for comparison of all the scores
    data['bert_scores'] = list_bert
    data['USE_scores'] = list_USE
    data['exact_scores'] = list_exact
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=['bert_scores', 'USE_scores', "exact_scores"])

    data['final_res'] = ((data[['bert_scores', 'USE_scores']].max(axis=1)) * 80) + data['exact_scores'] * 20
    for index in data.index:
        dict_res[index] = data['final_res'][index]

    sorted_dict = dict(sorted(dict_res.items(), key=operator.itemgetter(1), reverse=True))
    dict_items = sorted_dict.items()
    top_n_prods = list(dict_items)[:N_prod]
    results1 = pd.DataFrame(data=top_n_prods, columns=['index', 'score'])
    title1 = [Df['title'][results1['index'][i]] for i in results1.index]
    description1 = [Df['description'][results1['index'][i]] for i in results1.index]
    brand = [Df['brand'][results1['index'][i]] for i in results1.index]

    results1['title'] = title1
    results1['description'] = description1
    results1['brand'] = brand

    return results1


import json
import requests
import numpy as np


def embeddings(text, is_bert, lang):
    """

    :param text: list of texts
    :param is_bert: True for bert , false for USE
    :return: array of embeddings
    """
    bert_endpoint = 'http://35.180.247.177:8001/bert'
    use_endpoint = 'http://35.180.247.177:8001/use'
    params = {
        'sentences': text,
        'lang': lang
    }
    params = json.dumps(params)
    if is_bert:
        query_emb = requests.post(bert_endpoint, data=params)
        if query_emb.ok:
            query_emb = query_emb.json()
            query_emb = query_emb.get('bert_embeddings')
        else:
            raise ValueError(query_emb.text)
        query_emb = np.array(query_emb)
    else:
        query_emb = requests.post(use_endpoint, data=params)
        if query_emb.ok:
            query_emb = query_emb.json()
            query_emb = query_emb.get('use_embeddings')
        else:
            raise ValueError(query_emb.text)
        query_emb = np.array(query_emb)

    return query_emb


