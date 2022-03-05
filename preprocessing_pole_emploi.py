import pandas as pd
import re
import numpy as np
import string

df_11=pd.read_csv('C:\\Users\\Tim_secure\\Documents\\GitHub\\test\\data_11.csv')

df_24=pd.read_csv('C:\\Users\\Tim_secure\\Documents\\GitHub\\test\\data_24.csv')

df_27=pd.read_csv('C:\\Users\\Tim_secure\\Documents\\GitHub\\test\\data_27.csv')

df_28=pd.read_csv('C:\\Users\\Tim_secure\\Documents\\GitHub\\test\\data_28.csv')

df_32=pd.read_csv('C:\\Users\\Tim_secure\\Documents\\GitHub\\test\\data_32.csv')

df_44=pd.read_csv('C:\\Users\\Tim_secure\\Documents\\GitHub\\test\\data_44.csv')

df_52=pd.read_csv('C:\\Users\\Tim_secure\\Documents\\GitHub\\test\\data_52.csv')

df_75=pd.read_csv('C:\\Users\\Tim_secure\\Documents\\GitHub\\test\\data_75.csv')

df_76=pd.read_csv('C:\\Users\\Tim_secure\\Documents\\GitHub\\test\\data_76.csv')

df_84=pd.read_csv('C:\\Users\\Tim_secure\\Documents\\GitHub\\test\\data_84.csv')

df_93=pd.read_csv('C:\\Users\\Tim_secure\\Documents\\GitHub\\test\\data_93.csv')

df_region=pd.concat([df_11, df_24,df_27, df_28, df_32, df_44, df_52, df_75, df_76, df_84, df_93])


df_region=df_region.dropna(subset=['salaire.libelle', 'salaire.commentaire','salaire.complement1', 'salaire.complement2'], how='all')

df_region=df_region.drop(['id', 'Unnamed: 0', 'dateCreation', 'dateActualisation', 'accessibleTH', 'entreprise.entrepriseAdaptee', 'contact.nom', 
                          'contact.coordonnees1','agence.courriel','contact.commentaire', 'contact.urlPostulation', 'permis', 
                          'experienceCommentaire','entreprise.logo', 'lieuTravail.latitude', 'lieuTravail.longitude',
                          'entreprise.url', 'nombrePostes', 'deplacementCode',
                          'deplacementLibelle', 'offresManqueCandidats', 'alternance', 'contact.courriel',
                          'origineOffre.origine', 'origineOffre.urlOrigine', 'complementExercice', 
                          'origineOffre.partenaires', 'contact.coordonnees3'],axis=1)


df_region['salaire.libelle'] = df_region['salaire.libelle'].fillna(df_region['salaire.commentaire'])

remove_words = ['A négocier', 'De', 'FD', 'compétitif', 'SELON LA GRILLE DE FHF', 'selon profil', 'Selon grille FPH', 'Fixe + variable', '12 mois + prime décentralisée ', 'Selon convention FNCL', 'Selon profil', 'Selon expérience', 'A définir selon profil', 'suivant grille indiciaire FPH', 
                'Varie selon', 'Selon le profit d', 'fixe + variable','12 mois + prime décentralisée', 'à négocier', 'Interessement', 'A partirr de', 'Selon le profil d']
pat = r'\b(?:{})\b'.format('|'.join(remove_words))
df_region['salaire.libelle'] = df_region['salaire.libelle'].str.replace(pat, '')
df_region=df_region.drop(['salaire.commentaire'],axis=1)


df_region= df_region.applymap(lambda s:s.lower() if type(s) == str else s)

df_region['data_scientist']=np.where(df_region[['intitule', 'appellationlibelle', 'description']].apply(lambda x: x.str.contains('data scientist')).any(1), 'data scientist', None)
df_region['data_analyst'] = np.where(df_region[['intitule', 'appellationlibelle', 'description']].apply(lambda x: x.str.contains('data analyst')).any(1), 'data analyst', None)
df_region['big_data'] = np.where(df_region[['intitule', 'appellationlibelle', 'description']].apply(lambda x: x.str.contains('big data')).any(1), 'developpeur big data', None)
df_region['data_manager'] = np.where(df_region[['intitule', 'appellationlibelle', 'description']].apply(lambda x: x.str.contains('data manager|chief', case=False, regex=True)).any(1), 'data manager', None)
df_region['consultant'] = np.where(df_region[['intitule', 'appellationlibelle', 'description']].apply(lambda x: x.str.contains('consultant')).any(1), 'consultant data', None)
df_region['data engineer'] = np.where(df_region[['intitule', 'appellationlibelle', 'description']].apply(lambda x: x.str.contains('data engineer|ingénieur data|architecte|architect',
                                                 case=False, regex=True)).any(1), 'data engineer', None)


df_adaptif=df_region.dropna(subset=['data_scientist', 'data_analyst','big_data', 'data_manager', 'consultant', 'data engineer',  
                                    'data engineer'], how='all')

df_adaptif['big_data'] = df_adaptif['big_data'].fillna(df_adaptif['data_analyst'])
df_adaptif['big_data'] = df_adaptif['big_data'].fillna(df_adaptif['data_scientist'])
df_adaptif['big_data'] = df_adaptif['big_data'].fillna(df_adaptif['data_manager'])
df_adaptif['big_data'] = df_adaptif['big_data'].fillna(df_adaptif['data engineer'])
df_adaptif['big_data'] = df_adaptif['big_data'].fillna(df_adaptif['consultant'])

df_adaptif=df_adaptif.drop(['data_scientist',
                          'data_analyst','data_manager','data engineer', 
                          'consultant',],axis=1)
df_adaptif.rename(columns={'big_data': 'poste'}, inplace=True)


df_adaptif['experienceLibelle']= df_adaptif['experienceLibelle'].str.replace(r'(^.*débutant accepté.*$)', '0')

df_adaptif = df_adaptif.assign(experienceLibelle = lambda x: x['experienceLibelle'].str.extract('(\d+)'))

df_adaptif['bac+'] = df_adaptif['formations'].str.extract('([a-zA-Z+]+[0-9-])',expand= True)

df_adaptif['bac+1']=df_adaptif['description'].str.extract('(\+[0-9.])',expand= True)
df_adaptif['bac+1'] = np.where(df_adaptif[['intitule', 'description',
                                            'entreprise.description', 'competences']].apply(lambda x: x.str.contains('bac+',
                                                 case=False, regex=True)).any(1),df_adaptif['bac+1'] , None)

df_adaptif['bac+1']=df_adaptif['bac+1'].str.extract("(\d+)", expand=True)                                                                                                                                                                                                       
df_adaptif['bac+']=df_adaptif['bac+'].str.extract("(\d+)", expand=True)
df_adaptif['bac+'] = df_adaptif['bac+'].astype(float)
df_adaptif['bac+1'] = df_adaptif['bac+1'].astype(float)
df_adaptif['bac+'] = df_adaptif['bac+'].fillna(df_adaptif['bac+1'])
df_adaptif['bac+']=np.where(df_adaptif[['formations']].apply(lambda x: x.str.contains('bac ou équivalent',
                                                 case=False, regex=True)).any(1), 0, df_adaptif['bac+'])
df_adaptif['bac+'] = df_adaptif['bac+'].fillna(df_adaptif.groupby('poste')['bac+'].transform('mean'))
df_adaptif['bac+'] = df_adaptif['bac+'].astype(int)


df_adaptif['dureeTravail'] = df_adaptif['dureeTravailLibelle'].str.extract("(\d+\w+? |\d+\w+?\d+)", expand=True)

df_adaptif['dureeTravail1'] = df_adaptif['dureeTravail'] .str.extract("([a-zA-Z+]+[0-9+])", expand=True)

df_adaptif['dureeTravail'] = df_adaptif['dureeTravail'].str.extract("(\d+)", expand=True)
df_adaptif['dureeTravail1']=df_adaptif['dureeTravail1'].str.extract("(\d+)", expand=True) 
df_adaptif['dureeTravail1']=(df_adaptif['dureeTravail1'].astype(float))/6
df_adaptif['dureeTravail1'] = df_adaptif['dureeTravail1'].fillna(0) 
df_adaptif['dureeTravail']=(df_adaptif['dureeTravail'].astype(float))+(df_adaptif['dureeTravail1'].astype(float))
df_adaptif['dureeTravail'] = df_adaptif['dureeTravail'].fillna(35)
df_adaptif['dureeTravail']=df_adaptif['dureeTravail'].round(decimals = 2)                        


df_adaptif['Anglais']=np.where(df_adaptif[['langues', 'description']].apply(lambda x: x.str.contains('anglais')).any(1), 1, 0)
df_adaptif['lieuTravail.libelle']=df_adaptif['lieuTravail.libelle'].str.rstrip(string.digits)
df_adaptif['lieuTravail.libelle']=df_adaptif['lieuTravail.libelle'].replace({'nouvelle-aquitaine':'87 - limoges'})
df_adaptif['Departement']=df_adaptif['lieuTravail.libelle'].str.extract("(\d+)", expand=True) 
df_adaptif['Departement'] = df_adaptif['Departement'].fillna(df_adaptif['lieuTravail.commune'].astype(str).str[:2])


df_adaptif['Contrat']=np.where(df_adaptif[['natureContrat']].apply(lambda x: x.str.contains('apprentissage')).any(1), 'apprentissage', None)
df_adaptif['contrat1']=np.where(df_adaptif[['natureContrat']].apply(lambda x: x.str.contains('professionnalisation')).any(1), 'professionnalisation', None)
df_adaptif['contrat2']=np.where(df_adaptif[['natureContrat', 'intitule']].apply(lambda x: x.str.contains('stage')).any(1), 'stage', None)
df_adaptif['Contrat'] = df_adaptif['Contrat'].fillna(df_adaptif['contrat1'])
df_adaptif['Contrat'] = df_adaptif['Contrat'].fillna(df_adaptif['contrat2'])
df_adaptif['Contrat'] = df_adaptif['Contrat'].fillna(df_adaptif['typeContrat'])


df_adaptif['Statut2']=np.where(df_adaptif[['Contrat']].apply(lambda x: x.str.contains('apprentissage|professionnalisation')).any(1), 6, None)
df_adaptif['Statut1']=np.where(df_adaptif[['Contrat']].apply(lambda x: x.str.contains('stage')).any(1), 0, None)
#df_adaptif['Statut'] = df_adaptif['Statut'].fillna(df_adaptif['qualificationCode'].astype(float))
df_adaptif['Statut']=df_adaptif['qualificationCode'].astype(float)
df_adaptif['Statut'] = df_adaptif['Statut'].fillna(df_adaptif['Statut1'])
df_adaptif['Statut'] = df_adaptif['Statut'].fillna(df_adaptif['Statut2'])
df_adaptif['Statut'] = np.where(df_adaptif['Statut']> 8, 9, df_adaptif['Statut'])
df_adaptif['Statut'] = df_adaptif['Statut'].fillna(df_adaptif.groupby(['poste', 'Contrat'])['Statut'].transform('mean'))

df_adaptif['Statut'] = np.where(df_adaptif['Statut']> 8, 9, df_adaptif['Statut'])

df_adaptif['Statut'] = np.where((df_adaptif['Statut']> 7) & (df_adaptif['Statut']<8), 8, df_adaptif['Statut'])

to_replace = {'ï¿½': '', 'à': '', 'mois': '', 'horaire': '','euros': '', 
       'annuel': '', 'autre': '', 'de': '', 'partir': '', 'a': '', 'mensuel': '',  
       'brut': '', 'sur 12.5': '', 'sur 12': '', 'sur 13.3' :'', 'sur 13': '', 'k€': '', ',00': '', ',': '.'
       }


df_adaptif['salaire'] =df_adaptif['salaire.libelle'] .replace(to_replace, regex=True)

df_adaptif['salaire'] = df_adaptif['salaire'].str.strip()

df_adaptif['salaire_min'] = df_adaptif['salaire'].str.extract('(\d+\.? |\d+\.\d+)',expand= True)
df_adaptif['salaire_min'] = df_adaptif['salaire_min'].fillna(df_adaptif['salaire'])

df_adaptif['salaire_min'] = pd.to_numeric(df_adaptif['salaire_min'],errors='coerce')

df_adaptif['salaire_max'] = df_adaptif['salaire'].str.extract('.*?([0-9.]+[0-9]+)$')

df_adaptif['salaire_max'] = pd.to_numeric(df_adaptif['salaire_max'],errors='coerce')

df_adaptif['salaire_moyen'] = df_adaptif[['salaire_min', 'salaire_max']].mean(axis=1)

df_adaptif['salaire_moyen']=np.where(df_adaptif[['salaire.libelle']].apply(lambda x: x.str.contains('mensuel')).any(1), df_adaptif['salaire_moyen']*12, df_adaptif['salaire_moyen'])

df_adaptif['salaire_moyen']=np.where(df_adaptif[['salaire.libelle']].apply(lambda x: x.str.contains('k€')).any(1), df_adaptif['salaire_moyen']*1000, df_adaptif['salaire_moyen'])

df_adaptif['salaire_moyen']=np.where(df_adaptif[['salaire.libelle']].apply(lambda x: x.str.contains('horaire')).any(1), df_adaptif['salaire_moyen']*df_adaptif['dureeTravail']*52.143, df_adaptif['salaire_moyen'])

df_adaptif['salaire_moyen']=df_adaptif['salaire_moyen'].astype(float)

df_adaptif=df_adaptif.drop_duplicates(subset=['poste', 'Departement', 'Statut', 'dureeTravail', 'salaire_moyen'])
df_adaptif['salaire_moyen'] = df_adaptif['salaire_moyen'].fillna(df_adaptif.groupby(['poste', 'Contrat','Statut'])['salaire_moyen'].transform('median'))
                                                                                    
df_adaptif=df_adaptif.dropna(subset=['salaire_moyen'], how='all')

nom = df_adaptif['entreprise.description'].str.split().str.get(0)
df_adaptif['entreprise.nom']=df_adaptif['entreprise.nom'].fillna(nom)
df_adaptif['entreprise.nom']=df_adaptif['entreprise.nom'].str.replace(',','')
df_adaptif['entreprise.nom']=df_adaptif['entreprise.nom'].str.replace('restaurant','karpos')

secteurActivite=df_adaptif[df_adaptif['secteurActivite'].isna() ]

secteurActivite=secteurActivite[['entreprise.nom']]


df_adaptif['secteurActivite62'] = np.where(df_adaptif[['entreprise.nom']].apply(lambda x: x.str.contains('viseo|conserto|prestinfo services|agilicio|celad',
                                                 case=False, regex=True)).any(1), 62, None)

df_adaptif['secteurActivite78']=np.where(df_adaptif[['entreprise.nom']].apply(lambda x: x.str.contains('expectra|manpower|karpos',
                                                 case=False, regex=True)).any(1), 78, None)

df_adaptif['secteurActivite70']=np.where(df_adaptif[['entreprise.nom']].apply(lambda x: x.str.contains('fed it|talexim',
                                                 case=False, regex=True)).any(1), 70, None)

df_adaptif['secteurActivite10']=np.where(df_adaptif[['entreprise.nom']].apply(lambda x: x.str.contains('groupe psa',
                                                 case=False, regex=True)).any(1), 10, None)

df_adaptif['secteurActivite35']=np.where(df_adaptif[['entreprise.nom']].apply(lambda x: x.str.contains('enedis|total energies',
                                                 case=False, regex=True)).any(1), 35, None)

df_adaptif['secteurActivite45']=np.where(df_adaptif[['entreprise.nom']].apply(lambda x: x.str.contains('tressol-chabrier',
                                                 case=False, regex=True)).any(1), 45, None)


df_adaptif['secteurActivite'] = df_adaptif['secteurActivite'].fillna(df_adaptif['secteurActivite62'])
df_adaptif['secteurActivite'] = df_adaptif['secteurActivite'].fillna(df_adaptif['secteurActivite78'])
df_adaptif['secteurActivite'] = df_adaptif['secteurActivite'].fillna(df_adaptif['secteurActivite70'])
df_adaptif['secteurActivite'] = df_adaptif['secteurActivite'].fillna(df_adaptif['secteurActivite10'])
df_adaptif['secteurActivite'] = df_adaptif['secteurActivite'].fillna(df_adaptif['secteurActivite35'])
df_adaptif['secteurActivite'] = df_adaptif['secteurActivite'].fillna(df_adaptif['secteurActivite45'])
df_adaptif['secteurActivite'] = df_adaptif['secteurActivite'].fillna(df_adaptif['secteurActivite45'])
df_adaptif['secteurActivite'] = df_adaptif['secteurActivite'].fillna(70)
df_adaptif['secteurActivite']=df_adaptif['secteurActivite'].astype(int)
df_adaptif['secteurActivite']=df_adaptif['secteurActivite'].astype(str)

df_adaptif

df_adaptif=df_adaptif.drop(['bac+1', 'contrat1', 'contrat2',
                            'Statut1', 'Statut2', 
                            'dureeTravail1', 'secteurActivite45',
                            'secteurActivite35', 'secteurActivite70',
                            'secteurActivite78', 'secteurActivite62',
                            'secteurActivite10'],axis=1)

df_adaptif.rename(columns={'experienceLibelle': 'experience'}, inplace=True)

df_adaptif['experience'] = df_adaptif['experience'].astype(float)
df_adaptif['experience'] = df_adaptif['experience'].fillna(df_adaptif.groupby(['experienceExige', 'Contrat', 'salaire_moyen'])['experience'].transform('mean'))
df_adaptif['experience']=np.where(df_adaptif['salaire_moyen']>70000, 5, df_adaptif['experience'])
df_adaptif['experience'] = df_adaptif['experience'].round(decimals = 0) 

df_adaptif['mutuelle'] = np.where(df_adaptif[['description','salaire.complement1', 'salaire.complement2']].apply(lambda x: x.str.contains('mutuelle',
                                                 case=False, regex=True)).any(1), 1, 0)

df_adaptif['autre bonus'] = np.where(df_adaptif[['salaire.complement1', 'salaire.complement2']].apply(lambda x: x.str.contains('chèque repas|ce|autre|participation',
                                                 case=False, regex=True)).any(1), 1, 0)


df_adaptif['langages_data'] = np.where(df_adaptif[['intitule', 'appellationlibelle', 'description',
                                            'entreprise.description', 'competences']].apply(lambda x: x.str.contains('python|julia|rstudio|matlab|scala',
                                                 case=False, regex=True)).any(1), 1, 0)

                                                                                                                  
df_adaptif['librairies_data'] = np.where(df_adaptif[['intitule', 'appellationlibelle', 'description',
                                            'entreprise.description', 'competences']].apply(lambda x: x.str.contains('spark|hadoop|pandas|numpy|scikit-learn|tensorflow|keras|pytorch|kafka',
                                                 case=False, regex=True)).any(1), 1, 0)                                                                                                                     
                                                                                                                     
                                                                                                                     
df_adaptif['autres_langages'] = np.where(df_adaptif[['intitule', 'appellationlibelle', 'description',
                                            'entreprise.description', 'competences']].apply(lambda x: x.str.contains('java|flutter|php|css|html',
                                                 case=False, regex=True)).any(1), 1, 0)
                                                                                                                
df_adaptif['bases_donnees'] = np.where(df_adaptif[['intitule', 'appellationlibelle', 'description',
                                            'entreprise.description', 'competences']].apply(lambda x: x.str.contains('sql|postgresql|mysql|nosql|mongodb',
                                                 case=False, regex=True)).any(1), 1, 0)

df_adaptif['cloud'] = np.where(df_adaptif[['intitule', 'appellationlibelle', 'description',
                                            'entreprise.description', 'competences']].apply(lambda x: x.str.contains('aws|azure|ovh|google|cloud',
                                                 case=False, regex=True)).any(1), 1, 0)
                                                                                                                    
df_adaptif['app_analyse'] = np.where(df_adaptif[['intitule', 'appellationlibelle', 'description',
                                            'entreprise.description', 'competences']].apply(lambda x: x.str.contains('powerbi|power bi|tableau|sisense|qlik|sisense|analytics',
                                                 case=False, regex=True)).any(1), 1, 0)

df_adaptif['motscles'] = np.where(df_adaptif[['intitule', 'appellationlibelle', 'description',
                                            'entreprise.description', 'competences']].apply(lambda x: x.str.contains('mathématiques|statistique|supérvisé|learning',
                                                 case=False, regex=True)).any(1), 1, 0)                                                                                                                     


new_df = df_adaptif.filter(['poste','Departement', 'secteurActivite',
                            'experience','Contrat', 'Statut', 'dureeTravail',
                            'bac+', 'Anglais','mutuelle', 'autre bonus',  
                            'langages_data','librairies_data', 'autres_langages',
                            'bases_donnees','cloud', 'app_analyse',
                            'motscles','salaire_moyen'], axis=1)


