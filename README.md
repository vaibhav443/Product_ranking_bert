# Product_ranking_bert
Product ranking algorithm based on rank_bm25,Bert and Universal sentence encoders.


Main function-
```python
def product_ranker(prod_params):
    """

    :param prod_params: Product parameters
    :return: list of results
    """

    list_res = []
    Df = pd.DataFrame(prod_params.get("data"))
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

```
Input Format-

Input is a dictionary containing list of keywords,lang, data and No of top products to search.
```python
prod_params = {
    "kwds": ["gélules de vitamine"],
    "lang": "fr",
    "data": data_dict,
    "N_prod": 5
}
```
data_dict is a dictionary containing title, description, brand and price of products.

sample format of data_dict-
```python
{
    "title": {"0":"Alphanova - Lait solaire Bébé bio SPF 50+ - 50 ml - Solaires",
      "1":"Phyto-Actif - Acérola Plus 500 - 2 x 15 comprimés - Vitamines et minéraux",
      },
      "description":{"0":"text of description",
      "1":"text of description"
      },
      "brand":{"0":"Alphanova",
      "1":"Phyto-Actif"},
      "price":{"0":13.25,
      "1":8.9}
}
```

Output format-
Output is a list of dictionaries containing keyword and top_n_results.
```python
[
   {
      "kwd_value":"gélules de vitamine",
      "top_n_results":
         {
            "score":{
               "0":94.7544252872467,
               "1":86.8496572971344,
               "2":62.51157820224762,
               "3":60.15981961041689,
               "4":53.921242356300354
            },
            "title":{
               "0":"Biotechnie - Calcium Marin - 40 gélules - Vitamines et minéraux",
               "1":"Alphanova - Après-soleil Aloé vera et Grenade bio - 125 ml - Solaires",
               "2":"Alphanova - Lait solaire Bébé bio SPF 50+ - 50 ml - Solaires",
               "3":"Phyto-Actif - Acérola Plus 500 - 2 x 15 comprimés - Vitamines et minéraux",
               "4":"Nature's Plus - SOURCE DE VIE Adulte 60 Comprimés - Vitamines et minéraux"
            },
            "description":{
               "0":"text of description",
               "1":"text of description",
               "2":"text of description",
               "3":"text of description",
               "4":"text of description"
            },
            "brand":{
            "0":"Biotechnie",
            "1":"Alphanova",
            "2":"Alphanova",
            "3":"Phyto-Actif",
            "4":"Nature's Plus"
            }
         }
     }
]
```
