from sklearn.feature_extraction import DictVectorizer

def test_dict_vectorizer():
    data = [{'BKG': 'AAM'}, {'BKG': 'ALM'}, {'BKG': 'LLK'}, {'TLD': 'AAA'}]
    vec = DictVectorizer()
    vec.fit(data)
    # matrix = vec.fit_transform(data)
    t = vec.transform([{'BKG': 'AAM'}])
    print('this is t: ')
    print(t.toarray())
    # print(matrix)
    # a = matrix.toarray()
    # print(a)
    names = vec.get_feature_names()
    print(names)
    # map = vec.inverse_transform(matrix)
    print(map)


test_dict_vectorizer()