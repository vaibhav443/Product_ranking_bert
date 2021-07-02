from Utils import *
from itertools import product
import pandas as pd
import json
import time

final_df = pd.read_csv(r"C:\Users\vaibh\Downloads\PM1 Internship\trans_data.csv")
df = final_df[:100]
data_dict = df.to_dict()

prod_params = {
    "kwds" : ["huile bebe"],
    "lang" : "fr",
    "data" : data_dict,
    "N_prod" : 5
}

def product_ranker(prod_params):
    """

    :param prod_params: Product parameters
    :return: list of results
    """

    list_res = []
    Df = pd.DataFrame(prod_params.get("data"))
    #trans_data_pre = translation(Df)
    #trans_df = translation_final(trans_data_pre)
    #Df['description_eng'] = trans_df['description']
    #Df['title_eng'] = trans_df['title']
    Df_exact = basic_preprocessing_exact(Df)
    Df_ST = basic_preprocessing_ST(Df)
    keyword_exact = basic_preprocessing_keyword_exact(prod_params.get("kwds").copy())
    keyword_ST = basic_preprocessing_keywords_ST(prod_params.get("kwds").copy())
    for index in range(len(keyword_exact)):
        dict_res = {}
        dict_exact = exact_match(keyword_exact[index], Df_exact)
        embeddings_title_bert, embeddings_descrip_bert = get_embeddings_bert(Df_ST,prod_params.get("lang","en"))
        dict_bert = scores_bert_server(embeddings_title_bert, embeddings_descrip_bert, Df_ST, keyword_ST[index],prod_params.get("lang","en"))
        embeddings_title_USE, embeddings_descrip_USE = get_embeddings_USE(Df_ST,prod_params.get("lang","en"))
        dict_USE = scores_USE_server(embeddings_title_USE, embeddings_descrip_USE, Df_ST, keyword_ST[index],prod_params.get("lang","en"))
        results = final_results(dict_exact, dict_bert, dict_USE, prod_params.get("N_prod", 3), Df)
        dict_res["kwd_value"] = prod_params.get("kwds")[index]
        dict_res["top_n_results"] = results[['score','title','description','brand']].to_dict()

        list_res.append(dict_res)

    return list_res


start = time.time()
list_ = product_ranker(prod_params)
print(list_)
end = time.time()
print(end - start)


