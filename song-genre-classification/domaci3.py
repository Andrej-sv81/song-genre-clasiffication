import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from sklearn.naive_bayes import MultinomialNB

#funckija koja uklanja znakove, dodatne razmake, rijeci krace od 3 slova i  pretvara sva slova u mala
def preprocess_text(text):
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    text = ' '.join(text.split())
    text = ' '.join(word for word in text.split() if len(word) > 2)
    text = text.lower()
    return text
    
def main(train, test):
    data_train = pd.read_json(train)
    data_test = pd.read_json(test)
    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)
    df_train['strofa'] = df_train['strofa'].apply(preprocess_text)
    df_test['strofa'] = df_test['strofa'].apply(preprocess_text)

    #rijeci nepotpunog znacenja
    stopwords = ['ali', 'kroz', 'samo', 'sto',
                'dok', 'kad', 'jer', 'hej',
                'halo', 'mozda', 'verovatno', 'evo',
                'eno', 'eto']

    vectorizer = TfidfVectorizer(stop_words=stopwords)
    X_train = vectorizer.fit_transform(df_train['strofa'])
    X_test = vectorizer.transform(df_test['strofa'])

    le = LabelEncoder()
    y_train = le.fit_transform(df_train['zanr'])
    y_test = le.transform(df_test['zanr'])

    model = SVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f1_score(y_test, y_pred, average='micro'))

if __name__ == "__main__":
    path_train = sys.argv[1]
    path_test = sys.argv[2]
    main(path_train, path_test)