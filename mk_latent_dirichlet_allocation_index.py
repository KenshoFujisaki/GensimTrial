import gensim

# load data
dictionary = gensim.corpora.Dictionary.loadFromText('jawiki_wordids.txt')
tfidf_corpus = gensim.corpora.MmCorpus('jawiki_tfidf.mm')

# calculate LSI(latent semantic index)
## "nomTopics" is number of topics, in other words, dimension of semantic space.
lda = gensim.models.LdaModel(
        corpus = tfidf_corpus, 
        id2word = dictionary, 
        numTopics = 300)

# save index
lda.save('jawiki_lda_topics300.model')

# usage of this index:
# lda = gensim.models.LdaModel.load('jawiki_lda_topics300.model')
