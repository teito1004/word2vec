### word2vecとT-SNEを用いたライブドアのニュース記事の可視化

### 概要
ライブドアのニュース記事ごとに、ドキュメント特徴量を抽出します。
ドキュメント特徴量の抽出方法：
1. 記事に出現する単語のword2vec特徴量の平均
2. 記事に出現する単語のword2vec特徴量のTF-IDFの重み付き平均
3. 記事のseq2seqのembbeding特徴量

現在は、１）のみ実装済み。

### 要件
* Ubuntu 16.04
* python 3.6

### インストール方法

#### Mecab
以下を参照して、インストールします。
http://hirotaka-hachiya.hatenablog.com/entry/2017/10/05/130026

#### word2vec
1. pip install gensim

#### livedoor-news-data
ライブドアのニュース記事データのtgzファイルをダウンロードし、livedoor-news-dataに置き、解凍します。
ライブドアのニュース記事は、カテゴリごとにxmlで記載されています。
1. cd livedoor-news-data
2. wget https://www.rondhuit.com/download/livedoor-news-data.tar.gz
3. tar -xvzf livedoor-news-data.tar.gz
4. cd ..

### 使い方
上記のインストールが完了した状態で、以下のようにword2vec_livedoor.pyを実行します。 

python word2vec_livedoor.py T F T

word2vec_livedoor.pyの引数は、
* 記事データへの処理を行うフラグ
* 辞書データの作成・更新を行うフラグ
* word2vecモデルの学習を行うフラグ
となっています。

### 各ニュースカテゴリに対する処理（方式１の場合）の説明
1. livedoor-news-dataからxmlの記事データを読み込む。
2. titleとbodyアイテムの値を抽出したテキストデータを、記事ごとに１行ずつlivedoor-news-data-txtに書き出す。
3. テキストデータにMecabで形態素解析し、形態素ごとに半角スペースで区切った形態素データをlivedoor-news-wakatiに書き出す。
4. 形態素データに対し、word2vecをかけて、記事ごとに平均を取ったベクトル（100次元）をlivedoor-news-pklに書き出す。つまり、pklには、記事の数 X 100次元の行列が保存される。

### To Do
1. word2vec特徴量の次元を引数またはmainの変数として設定できるようにする
2. 特徴量の種類を引数で設定できるようにする。また、特徴量の出力先を特徴量の種類ごとに用意する
3. 特徴量2と3を実装する
4. 形態素の区切り文字として半角スペースを用いているため、元々文字列含まれる半角スペース（例えば、株式会社 XXX）と区切り文字を、word2vecが混同してしまう問題がある。そのため、前処理として、テキストの段階で半角スペースをハイフン「-」置き換えるなどの処理を入れる。
