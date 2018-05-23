### word2vec and t-sne for livedoor news data

### Requirements
* Ubuntu 16.04
* python 3.6

### Basic Installation

#### Mecab
Please refer to http://hirotaka-hachiya.hatenablog.com/entry/2017/10/05/130026

#### word2vec
1. pip install gensim

#### livedoor-news-data
1. cd livedoor-news-data
2. wget https://www.rondhuit.com/download/livedoor-news-data.tar.gz
3. tar -xvzf livedoor-news-data.tar.gz
4. cd ..

#### Generate mean 
1. python word2vec_livedoor.py T F T
