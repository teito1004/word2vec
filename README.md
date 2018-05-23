### word2vec and t-sne for livedoor news data

### Basic Installation
http://hirotaka-hachiya.hatenablog.com/entry/2017/10/12/101858

#### word2vec & Mecab
1. pip install gensim

#### livedoor-news-data
1. cd livedoor-news-data
2. download https://www.rondhuit.com/download/livedoor-news-data.tar.gz
3. tar -xvzf livedoor-news-data.tar.gz

#### Generate mean 
1. python word2vec_livedoor.py T F T
