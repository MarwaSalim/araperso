from sklearn.base import TransformerMixin,BaseEstimator
from CleaningStage import textCleaning
import  FileManging
import PreprocessingStage

class ExtractTextTransform(BaseEstimator,TransformerMixin):
    def __init__(self,ColumnI,ColumnO, Path="C:\\Users\Marwa\\PycharmProjects\LoadDS\\DataSet_10_Nov_Clean_Normalize_Redundant"):
        self.I = ColumnI
        self.O = ColumnO
        self.Path = Path
    def transform(self,X, **kwargs):
        X[self.O]=X.apply(lambda row: (FileManging.FileManger(self.Path,str(long(row[self.I]))+'.txt').readFile()), axis=1)
        return X[self.O]
    def fit(self,X, y=None, **kwargs):
        return self
    def get_params(self,**kwargs):
        return {'Path':self.Path,'ColumnI':self.I,'ColumnO':self.O}

class ColumnExtractorTransformtion(BaseEstimator,TransformerMixin):
    def __init__(self, cols=[]):
        self.cols = cols
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        Xcols = x[self.cols]
        return Xcols
    def get_params(self, **kwargs):
        return {'cols': self.cols}