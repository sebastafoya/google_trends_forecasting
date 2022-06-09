from turtle import down
import pandas as pd
from pytrends.request import TrendReq

KEYWORDS = [
    "sintomas covid",
    "trabajos",
    "carros",
    "reclutamiento",
    "real estate",
    "casas",
    "tenis",
    "botas",
    "sandalias",
    "adidas",
    "nike",
    "converse",
    "desempleo"
]

def download_keywords(keywords):

    pytrends = TrendReq(hl='es-MX', tz=360)

    df_list = []

    for kw in keywords:
        print(kw)

        pytrends.build_payload([kw], cat=0, timeframe='2019-05-01 2022-05-15', geo='MX')

        time_df = pytrends.interest_over_time()
        df_list.append(time_df[[kw]])

    df = pd.concat(df_list, axis=1)

    return df.reset_index()
    # breakpoint()
    # df.reset_index().to_csv('data/keyword_trends_tests.csv', index=False)
    # print(time_df)
