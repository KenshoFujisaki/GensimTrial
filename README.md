## GensimTrial  

本内容は，こちらの記事（[LSIやLDAを手軽に試せるGensimを使った自然言語処理入門](http://yuku-tech.hatenablog.com/entry/20110623/1308810518)）に従って，作業手順のみをピックアップしたものです.
簡単には，[Gensim](https://radimrehurek.com/gensim/)を用いて，Wikipediaダンプデータに対して，1)TF-IDFインデックス作成，2)LSI作成，および，3)LDAを実行します．  
[!] 環境はMacOSX 10.10 Yosemiteとします．

### Wikipediaコーパス作成手順  
1. 各種モジュールのインストール  
    ```bash
    $ pip install numpy
    $ pip install scipy
    $ pip install gensim==0.7.8  # important
    ```
    [!] gensimのバージョンは0.7.8としてください．ほかバージョンでは適切に動作しません．  

2. 日本語Wikipediaダンプデータの取得  
    ```bash
    $ wget http://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2
    ```

3. 日本語Wikipediaコーパスの生成  
    ```bash
    $ python jawikicorpus.py jawiki-latest-pages-articles.xml.bz2 jawiki 
    ```
    数時間待つ...  
    [!] もし，実行時にエラー`ImportError: No module named maxentropy`が出る場合，以下ファイルを修正し，上コマンドを再度実行します．
    また，ファイルパスは環境に応じて読み替えてください．
    * matutils.py に以下を追記  
        ```bash
        $ vim ~/.anyenv/envs/pyenv/versions/2.7.1/lib/python2.7/site-packages/gensim/matutils.py
        ```
        ```python
        # append this
        def logsumexp(a):
            a = asarray(a)
            a_max = a.max()
            return a_max + log((exp(a-a_max)).sum())
        ```
    * ldamodel.py を修正  
        ```bash
        $ vim ~/.anyenv/envs/pyenv/versions/2.7.1/lib/python2.7/site-packages/gensim/models/ldamodel.py
        ```
        ```diff
        -from scipy.maxentropy import logsumexp # log(sum(exp(x))) that tries to avoid overflow
        +from gensim.matutils import logsumexp
        ```

結果，以下ファイルが生成されます．  
> ［[LSIやLDAを手軽に試せるGensimを使った自然言語処理入門](http://yuku-tech.hatenablog.com/entry/20110623/1308810518)より引用］  
> 　jawiki_bow.mm  
> 　　　単語頻度(term frequency, tf)を次元の値とする文書単語行列  
> 　jawiki_tfidf.mm  
> 　　　単語頻度からtfidfを計算した文書単語行列  
> 　jawiki_wordids.txt  
> 　　次元と単語の関係。例えば、1次元目は"python"を表す、など。Gensimの中ではDictionaryと呼ばれる。  

### インデックス作成手順  
用途に合わせて各種インデックス作成します．  
* TF-IDFインデックス作成  
    インデックス作成  
    ```bash
    $ python ./mk_tfidf_index.py
    ```
    pythonからインデックスをロード  
    ```python
    import gensim
    dictionary = gensim.corpora.Dictionary.loadFromText('jawiki_wordids.txt')
    tfidf_index = gensim.similarities.SparseMatrixSimilarity.load('jawiki_tfidf_wimilarity.index')
    ```
* LSI作成  
    インデックス作成  
    [!] トピック数（潜在空間における次元数）は300としています．  
    ```bash
    $ python ./mk_latent_semantic_index.py
    ```
    pythonからインデックスをロード  
    ```python
    import gensim
    dictionary = gensim.corpora.Dictionary.loadFromText('jawiki_wordids.txt')
    lsi = gensim.models.LsiModel.load('jawiki_lsi_topics300.model')
    ```
* LDAモデル作成  
    インデックス作成  
    [!] トピック数は300としています．  
    ```bash
    $ python ./mk_latent_dirichlet_allocation_index.py
    ```
    pythonからモデルをロード  
    ```python
    import gensim
    dictionary = gensim.corpora.Dictionary.loadFromText('jawiki_wordids.txt')
    lda = gensim.models.LdaModel.load('jawiki_lda_topics300.model')
    ```
