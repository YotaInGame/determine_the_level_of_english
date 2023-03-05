import streamlit as st
import pandas as pd
import numpy as np
import string
import pysrt
import spacy
import os 

from joblib import dump, load
from os.path import abspath
from difflib import get_close_matches
from re import sub, compile, search
from difflib import get_close_matches
from nltk.stem import WordNetLemmatizer
from io import StringIO
from tempfile import NamedTemporaryFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from catboost import CatBoostClassifier, Pool

# константы
RANDOM_STATE = 42

def execute():

    st.title('Уровень сложности текста');

    uploaded_file = st.file_uploader("Загрузите файл с субтитрами .str")

    if uploaded_file is not None:
     
        file_url = abspath(__file__).replace('\index.py', '')

        save_uploadedfile(uploaded_file, file_url) 

        nlp = spacy.load('en_core_web_sm')
        lemmatizer = WordNetLemmatizer()
    
        text = process_text(read_text(file_url + "/temp/temp.srt"), nlp, lemmatizer)

        model = CatBoostClassifier()
        model.load_model(file_url + "/model/catboost.cbm")

        tfidf = load(file_url + '/model/tfidf.pkl')
        fid_features = tfidf.transform([text])

        predicted = model.predict_proba(Pool(data=fid_features))
        predicted = pd.DataFrame((predicted.T * 100).round(2), index=model.classes_, columns=['Вероятность'])

        st.bar_chart(predicted, use_container_width=True)
        

def save_uploadedfile(uploadedfile, file_url):
    with open(os.path.join(file_url + "/temp", 'temp.srt'),"wb") as f:
        f.write(uploadedfile.getbuffer())
    

# функция сопостовления 
# название файла с названием фильма в датасете
def set_filename(x, file_list):

    finded_name = get_close_matches(x, file_list)
    if finded_name:
        return finded_name[0] + '.srt'
    
    finded_name = list(filter(compile(x).match, file_list))
    if finded_name:
        return finded_name[0] + '.srt'
        
    return 0  

# чтение файла и добавление в датасет
def read_text(file):
    
    reader = pysrt.open(file, encoding='windows-1251')
    return reader.text

# функция предобработки
def process_text(text, nlp, lemmatizer):

    text = text.lower()
    text = sub(compile('<.*?>'), '', text) 
    text = sub(compile('™'), '', text) 
    text = sub(compile('вє'), '', text) 
    
    regular_url = r'(http\S+)|(www\S+)|([\w\d]+www\S+)|([\w\d]+http\S+)'
    text = sub(regular_url, '', text)

    text = sub('720p', '', text)
    text = sub('1080p', '', text)
    text = sub('3d', '', text) 

    text = sub('\'', ' ', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = sub('\d', '', text)
    text = sub('–', '', text)
      
    text_clean = ""
    text_clean = sub(' +', ' ', text)
    text_clean = text_clean.strip()
    text_clean = sub('\n', ' ', text_clean)
    
    document = nlp(text_clean) 
    cleared_text = [ent.text for ent in document if not ent.ent_type_]
    lemmatizer_text = [lemmatizer.lemmatize(word) for word in cleared_text]
 
    return ' '.join(lemmatizer_text)


def process_dataset():

     # собираем список файлов
    file_list = []
    file_path = abspath(__file__).replace('\index.py', '')

    for root, dirs, files in os.walk(file_path + "/data/subtitles/"):  
        for filename in files:
            names = filename.split('.')   
            file_list.append(names[0].lower())    
   
    # подготавливаем датасет для обучаения
    labels = pd.read_csv(file_path + '/data/labels_all.csv')
    labels['Movie'] = labels['Movie'].apply(lambda x: sub('[^a-zA-Z| ]', '', x).strip().replace(' ', '_'))
 
    # записываем название файлов в столбец
    labels.loc[~labels['Movie'].isna(), 'file_name'] = labels[~labels['Movie'].isna()]['Movie'].apply(lambda x: set_filename(x.lower(), file_list))

    # чистим то, что не удалось обьеденить
    labels = labels[~labels['file_name'].isna()]

    # считываем все файлы субтитров
    labels.loc[labels['file_name'] != 0, 'text'] = labels.query('file_name != 0')['file_name'].apply(lambda x: read_text(file_path + '/data/subtitles/' + x))
    labels = labels.query('file_name != 0')

    # считываем второй датасет
    movies = pd.read_csv(file_path +  '/data/cefr_leveled_texts.csv')
    movies.info()
    movies = movies.rename(columns={'label':'Level'})

    tmp = pd.concat([labels, movies], axis=0)

    labels = tmp

    nlp = spacy.load('en_core_web_sm')
    lemmatizer = WordNetLemmatizer()

    labels['text'] = labels['text'].apply(lambda x: process_text(x, nlp, lemmatizer))

    labels['split_level'] = labels['Level'].apply(lambda x: x.split('/')[0])
    labels['split_level'] = labels['split_level'].apply(lambda x: x.split(',')[0])

    label_list = labels['split_level'].unique().tolist()
    label_dict = {label_list[index] : index + 1 for index in range(len(label_list))}

    labels['code_level'] = labels['split_level'].apply(lambda x: label_dict[x])

    return labels


def get_features(data):

    features_train, features_valid, target_train, target_valid = train_test_split(
        data['text'].values, data['code_level'].values, 
        stratify=data['code_level'].values, 
        random_state=RANDOM_STATE, 
        test_size=0.4, 
        shuffle=True)

    return features_train, features_valid, target_train, target_valid


def tfid(features_train, features_valid, min_df, max_df):

    tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df)

    tfid_features_train = tfidf.fit_transform(features_train)
    tfid_features_valid = tfidf.transform(features_valid)

    return tfid_features_train, tfid_features_valid 

def model_logistic_reg(features_train, target_train, feature_valid, target_valid):

    model = LogisticRegression(class_weight='balanced')
    model.fit(features_train, target_train)

    #predictions = model.predict_proba(feature_valid)

    #print('accuracy_score:', accuracy_score(target_valid, predictions))

    return model

def mode_catboost_classifier(features_train, target_train):
    params = {
        'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'iterations':[100]
    }
    model = CatBoostClassifier(random_state=RANDOM_STATE)
    grid_search_result = model.grid_search(params, X=features_train, y=target_train, plot=True)
    grid_search_result.save_model('catboost')
    return grid_search_result

execute()  