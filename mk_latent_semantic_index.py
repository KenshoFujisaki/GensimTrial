import gensim

# load data
dictionary = gensim.corpora.Dictionary.loadFromText('jawiki_wordids.txt')
tfidf_corpus = gensim.corpora.MmCorpus('jawiki_tfidf.mm')

# calculate LSI(latent semantic index)
## "nomTopics" is number of topics, in other words, dimension of semantic space.
lsi = gensim.models.LsiModel(
        corpus = tfidf_corpus, 
        id2word = dictionary, 
        numTopics = 300)

# save index
lsi.save('jawiki_lsi_topics300.model')

# usage of this index:
# lsi = gensim.models.LsiModel.load('jawiki_lsi_topics300.model')
