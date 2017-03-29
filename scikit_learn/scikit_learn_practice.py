from sklearn.feature_extraction import DictVectorizer
from functools import reduce
import pandas as pd
import pickle
import os
import numpy as np

def test_dict_vectorizer(data):
    # data = [{'BKG': 'AAM'}, {'BKG': 'ALM'}, {'BKG': 'LLK'}, {'TLD': 'AAA'}]
    vec = DictVectorizer()
    vec.fit(data)
    # matrix = vec.fit_transform(data)
    # t = vec.transform([{'BKG': 'AAM'}])
    # print('this is t: ')
    # print(t.toarray())
    # print(matrix)
    # a = matrix.toarray()
    # print(a)
    # names = vec.get_feature_names()
    # print(names)
    # map = vec.inverse_transform(matrix)
    # print(map)
    return vec

def data_transform():
    df = pd.read_csv('../data_set/test20170314060000.csv')
    column_value = df.iloc[0:, 1]
    column_value_list = column_value.tolist()
    func = lambda x, y: x if y in x else x + [y]
    distinct_column_value_list = reduce(func, [[], ] + column_value_list)

    training_data = []
    for el in distinct_column_value_list:
        r = {column_value.name: str(el)}
        training_data.append(r)
    print(training_data)
    return training_data

if __name__ == "__main__":
    data = data_transform()
    vec_model = None
    vec_file_exists = os.path.exists('../model/vectorizer/vec1.pkl')
    if(vec_file_exists):
        vec_file = open('../model/vectorizer/vec1.pkl', 'rb')
        vec_model = pickle.load(vec_file)
    else:
        vec_model = test_dict_vectorizer(data)
        vec_file = open('../model/vectorizer/vec1.pkl', 'wb')
        pickle.dump(vec_model, vec_file)
    t = vec_model.transform({'releaseOffice': 'KHI'})
    print(t)
    print(vec_model.get_feature_names())