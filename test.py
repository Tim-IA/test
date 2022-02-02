import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly_express as px
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold

url1='https://raw.githubusercontent.com/Tim-IA/test/main/Filtred_spam.csv'
flitred_file=pd.read_csv(url1, encoding='latin-1')  
url2='https://raw.githubusercontent.com/Tim-IA/test/main/spam.csv'
orininal_csv=pd.read_csv(url2, encoding='latin-1')

st.set_page_config(page_title='Réglage des hyperparamètres', layout='wide')


genre = st.radio(
     "Dites Bonjour",
     ('Bonjour', 'Quoi'))
if genre =='Bonjour':
    st.write('On commence notre projet')
else:
     st.write("Vous n'avez pas dit Bonjour.")
     st.write('On commence notre projet')
     
col1, col2 =st.columns(2)
about_expander = col1.expander('à propos',expanded=False)
with about_expander:
    st.info("""
             Cette application Web n'est qu'une simple démonstration du réglage des hypermètres 
             avec **GridSearchCV**. Les données spam originales et filtrées par Eric DOMAS sont 
             importées pour détecter les éventuels spams. Pour faire cette étude, les algorithmes 
             tels que SVM, KNN, Random Forest sont importés depuis la librairie Scikit learn.  
             """)
info_expander = col2.expander('Pourquoi régler les hyperparamètres?',expanded=False)
with info_expander:
    st.info("""
             **Hyperparamètres** décrivent les bases du modèle.
             **Réglage des hyperparamètres** nous permet d'avoir une vue optimale de notre modèle
             """)
 
st.title('spam')
st.write("""#Exploration des données""")


data_name=st.sidebar.selectbox("Selectionner les données", ('filtred csv', 'original csv'))

classifier=st.sidebar.selectbox("Selectionner un classifier", ("KNN", "SVM", "Random Forest"))

cv_count = st.sidebar.slider('Compter Cross-validation', 2, 5, 3)

def analyse(data_name):
    st.title(data_name)
    if data_name=='filtred csv':
        df=flitred_file
        st.write(df)
    else:
       df=orininal_csv
       df=df.rename(columns={'v1':'type','v2': 'text'})
       st.write(df)
    plt.figure(figsize=(6,6))
    df_count_type = pd.DataFrame(df.type.value_counts()).reset_index().rename(
    columns={'index': 'type', 'type': 'count'})
    pourcentage = px.pie(df_count_type, names='type', color='type',
                     values='count', title='Repartition des types de messages')
    pie = st.write(pourcentage)
    y = df['type'] = df['type'].map({'ham': 0, 'spam': 1})
    X = df.drop(columns='type')
    return y, X
y, X=analyse(data_name)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

st.write('Shape:', X.shape)
def get_classifier(classifier):
    model=None
    parameters=None
    if classifier=='KNN':
        model=KNeighborsClassifier()
        pipe = Pipeline(steps=[('scaler', StandardScaler()),('model', model)])
        st.sidebar.write("**Nombre de Neighbors**")
        K=st.sidebar.slider("", 1, 15)
        parameters = {'model__n_neighbors':[K]} 
    elif classifier == 'SVM':
        model=SVC()
        pipe = Pipeline(steps=[('scaler', StandardScaler()),('model', model)])
        st.sidebar.write("**Type de noyau**")
        type_noyau = st.sidebar.multiselect('', options=['linear', 'rbf', 'poly'], default=['linear', 'rbf', 'poly'])
        st.sidebar.write('**Réglage des paramétres C**')
        C = st.sidebar.slider('', 1, 10)
        parameters = {'model__C':[C], 'model__kernel':type_noyau}
    else:
        model = RandomForestClassifier()
        pipe = Pipeline(steps=[('scaler', StandardScaler()),('model', model)])
        st.sidebar.write('**Nombre des arbres**')
        n = st.sidebar.slider('', 1, 220, 20)
        st.sidebar.write('La profondeur maximale arbre')
        md = st.sidebar.slider('', 2, 15, 1)
        parameters = {'model__n_estimators':range(n), 
                      'model__max_depth':range(md)}
    grid = GridSearchCV(pipe, parameters, cv = cv_count, return_train_score=False)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    st.subheader('**Paramètres et Mean test score**')
    st.write('Best Score:', round(grid.best_score_, 2))
    st.write("Coefficient de determination :", round(r2_score(y_test, y_pred), 3))
    st.write('Precision et Rappel:', precision_recall_curve(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=grid.classes_)
    #st.write("Matrice de confusion", ConfusionMatrixDisplay(confusion_matrix=cm,
                                 #display_labels=['ham', 'spam']))
    st.write("Matrice de confusion", confusion_matrix(y_test, y_pred))
    
    return model, parameters
          
model, parameters = get_classifier(classifier)


