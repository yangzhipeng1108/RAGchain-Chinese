from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    "what is the weather like today",
    "what is for dinner tonight",
    "this is a question worth pondering",
    "it is a beautiful day today"
]
tfidf_vec = TfidfVectorizer()
# 利用fit_transform得到TF-IDF矩阵
tfidf_matrix = tfidf_vec.fit_transform(corpus)
# 利用get_feature_names得到不重复的单词
print(tfidf_vec.get_feature_names())
# 得到每个单词所对应的ID
print(tfidf_vec.vocabulary_)
# 输出TF-IDF矩阵
print(tfidf_matrix)