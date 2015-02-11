import gensim

# load data
dictionary = gensim.corpora.Dictionary.loadFromText('jawiki_wordids.txt')
tfidf_corpus = gensim.corpora.MmCorpus('jawiki_tfidf.mm')

# calculate TF-IDF similarity index
tfidf_index = gensim.similarities.SparseMatrixSimilarity(tfidf_corpus)

# save index
tfidf_index.save('jawiki_tfidf_similarity.index')

# usage of this index:
# tfidf_index = gensim.similarities.SparseMatrixSimilarity.load('jawiki_tfidf_wimilarity.index')
