#view-source:https://www.airbnb.com.ar/rooms/17735312?_set_bev_on_new_domain=1587691662_203RhqK8ewbhPygB&source_impression_id=p3_1587692048_pmNglr%2BLiHVp2xHe&guests=1&adults=1&display_currency=USD


from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.externals import joblib
from joblib import dump, load
import pickle
import pandas as pd


class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names ):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self._feature_names ]
    
class TextTransformer(BaseEstimator, TransformerMixin):

    def __clean_text(self, x):
        for punct in "/-'":
            x = x.replace(punct, ' ')
        for punct in '&':
            x = x.replace(punct, f' {punct} ')
        for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~•' + '“”’':
            x = x.replace(punct, '')
        return x.lower()
    
    def __parseTextCols(self, x):
        finalTexts = []
        for i in x:
            if(pd.notna(i) and i not in finalTexts):
                finalTexts.append(i)
        text = self.__clean_text(' '.join(finalTexts))
        return text
    
    def fit(self, X, y = None):
        return self
    
    def transform (self, X, y = None):
        return X.apply(self.__parseTextCols, axis=1)
    
    
class custom_Tfidf(TfidfVectorizer, TransformerMixin):
    options= {
        'fitSample': 1
    }
    def __init__(self, params, options = None):
        self.vectorizer = TfidfVectorizer(**params)
        if(options != None):
            for key in options.keys():
                self.options[key] = options[key]
        
    def fit(self, X, y = None):
        self.vectorizer.fit(X.sample(frac=self.options['fitSample']))
        return self
    
    def transform(self, X, y = None):
        return self.vectorizer.transform(X)

class Dummy_explotable_transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        from ast import literal_eval
        self.literal_eval = literal_eval

    def clean_amenities(self, colValues):
        import re
        colValues = colValues.apply(lambda s: re.sub(r'[}{@#$"]', '', s).lower().split(','))
        return pd.get_dummies(
            data = colValues.apply(pd.Series).stack(),
            prefix = 'amenities'
        ).sum(level=0)

    def to_dummy(self, s, col_name):
        return pd.get_dummies(
            data = s.apply(lambda x: self.literal_eval(x) if isinstance(self.literal_eval(x), list) else ['none']).apply(pd.Series).stack(),
            prefix = col_name
        ).sum(level=0)
    
    def fit(self, X, y = None):
        return self
    
    def transform (self, X, y = None):
        X = pd.concat([X, self.to_dummy(X['host_verifications'], 'host_verifications')], axis=1,).fillna(0).drop('host_verifications', axis=1)
        X = pd.concat([X, self.clean_amenities(X['amenities'])], axis=1).fillna(0).drop('amenities', axis=1)
        return X
    
class Dummy_explotable_transformer_2(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        total = len(X)
        self.columns = X.columns[
            (X.apply(sum, axis=0)/total <= .9).values & (X.apply(sum, axis=0)/total >= .2).values
        ]
        return self

    def transform(self, X_transform, y=None):
        match_colums = [col for col in self.columns if (col in X_transform.columns)]
        missing_columns = [col for col in self.columns if (col not in X_transform.columns)]
        X = X_transform[match_colums]
        for col in missing_columns:
            X.loc[:, col] = 0
        X = X[self.columns].astype(float)
        return X


class FeaturesToLower(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.applymap(lambda x: str(x).lower().strip())

class NumericalImputer(BaseEstimator, TransformerMixin):
    def __init__( self, default_strategy = "median"):
        self._default_strategy = default_strategy
        self._default_values = {}
        
    def fit( self, X, y = None ):
        X.host_response_rate = X.host_response_rate.str.replace('%', '').astype(float)
        X.host_acceptance_rate = X.host_acceptance_rate.str.replace('%', '').astype(float)
        
        #Si hay valores infinitos los convertimos en NaN
        X = X.replace( [ np.inf, -np.inf ], np.nan )
        
        for col in X.columns:
            if col=='number_of_reviews_ltm':
                default_value=0
            elif col=='number_of_reviews':
                default_value=0
            elif col=='host_listings_count':
                default_value=1
            elif self._default_strategy=='median':
                default_value=np.median(X[col].dropna())
            elif self._default_strategy=='mode':
                default_value=np.mode(X[col].dropna())
            elif self._default_strategy=='mean':
                default_value=np.mean(X[col].dropna())
            else:
                default_value=np.median(X[col].dropna())
            self._default_values[col]=default_value

        return self 
    
    def transform(self, X, y = None):
        X.host_response_rate = X.host_response_rate.astype(str).str.replace('%', '').astype(float)
        X.host_acceptance_rate = X.host_acceptance_rate.astype(str).str.replace('%', '').astype(float)
        
        for col in X.columns:
            X[col] = X[col].astype(float)
            #Si hay valores infinitos los convertimos en NaN
            X[col] = X[col].replace( [ np.inf, -np.inf ], np.nan)
            X[col].fillna(self._default_values[col],inplace=True)
        return X

class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__( self, log_transform = True):
        self._log_transform = log_transform
        
    def fit( self, X, y = None ):
        return self 
    
    def transform(self, X, y = None):
        
        if self._log_transform:
            for col in X.columns:
                colname = col+"_log"
                X.loc[:,colname] = np.log(X[col]+1)
                
        #Retornamos un array de Numpy ?
        return X
    
class NumericalAddFeatures(BaseEstimator, TransformerMixin):
    def __init__( self, bath_per_bed = True, bath_per_bedroom = True ):
        self._bath_per_bed = bath_per_bed
        self._bath_per_bedroom = bath_per_bedroom
        
    def fit( self, X, y = None ):
        return self 
    
    def transform(self, X, y = None):
        if self._bath_per_bedroom:
            X.loc[X['bedrooms']==0,'bedrooms']=1
            X['bath_per_bedroom'] = X['bathrooms'] / X['bedrooms']
        if self._bath_per_bed:
            X.loc[X['beds']==0,'beds']=1
            X['bath_per_bed'] = X['bathrooms'] / X['beds']
        return X

class DateTransformer( BaseEstimator, TransformerMixin ):
    #Class Constructor
    mode = {}
    def __init__( self ):
        return None
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        for col in X.columns:
            self.mode[col] = X[col].mode()[0]
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        X = X[list(self.mode.keys())]
        for col in self.mode.keys():
            X.loc[:,col] = pd.to_datetime(X.loc[:,col].fillna(self.mode[col])).apply(lambda x: x.timestamp())
            
        from itertools import combinations
        
        for cols in list(combinations(self.mode.keys(), 2)):
            X.loc[:,cols[0]+'_resta_'+cols[1]] = (X.loc[:,cols[0]] - X.loc[:,cols[1]]).apply(abs)
            X.loc[:,cols[0]+'_resta_'+cols[1]+'_log'] = np.log(X.loc[:,cols[0]+'_resta_'+cols[1]] + 1)
            
        return X

class GeoTransformer( BaseEstimator, TransformerMixin ):
    
    def __init__( self, geoCols, cityCol ):
        self._geoCols = geoCols
        self._cityCol = cityCol
        
    def getPercentage(self, _input, _mean, _range):
        if(_range == 0):
            _range = 0.001
        return (_input - _mean) / (_range/2) * 100
    
    def fit( self, X, y = None ):
        cities = X[self._cityCol].unique()
        self._cities = {}
        for cityName in cities:
            cityDf = X[X[self._cityCol] == cityName]
            self._cities[cityName] = {}
            for coord in self._geoCols:
                self._cities[cityName][coord] = {
                    'max': cityDf[coord].max(),
                    'min': cityDf[coord].min(),
                    'mean': (cityDf[coord].max() + cityDf[coord].min()) / 2,
                    'real_mean': cityDf[coord].mean(),
                    'range': cityDf[coord].max() - cityDf[coord].min()
                }
        return self
    
    def add_features(self, df):
        cityName = df.name
        import math
        import numpy as np
        
        if cityName in self._cities.keys():
            cityData = self._cities[cityName]
            for coord in self._geoCols:
                df[coord] = self.getPercentage(df[coord], cityData[coord]['mean'], cityData[coord]['range'])
                df[coord+'_fromMean'] = self.getPercentage(df[coord], cityData[coord]['real_mean'], cityData[coord]['range'])
                df[coord+'_sin'] = np.sin(df[coord])
                df[coord+'_log'] = np.log(np.abs(df[coord]) + 1)
            df['distance'] = np.hypot(df[self._geoCols[0]], df[self._geoCols[1]])
        else:
            for coord in self._geoCols:
                
                df[coord] = 0
                df[coord+'_fromMean'] = 0
                df[coord+'_sin'] = 0
                df[coord+'_log'] = 0
            df['distance'] = 0
        return df
    
    def transform( self, X, y = None ):
        X = X.groupby([self._cityCol]).apply(self.add_features)
        X = X.drop(self._cityCol, axis=1)
        return X


import pickle
def load_model(path):
    return pickle.load(open(path, 'rb'))