import os
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.pipeline import (Pipeline, FeatureUnion)
from customerTransformation import ExtractTextTransform,ColumnExtractorTransformtion
from sklearn.preprocessing import (Imputer, StandardScaler)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pandas as pd
def Begin (df,LabelOfTarget):
    trainSamples=df
    TrainTargets = trainSamples[LabelOfTarget]
    TextFeaturesOnly=[
        ("extractTextUsed", ExtractTextTransform(ColumnO='AUX', ColumnI='ID')),
        ("tfidf", TfidfVectorizer(stop_words=None)),
    ]
    NumericalFeaturesOnly=[
                 ("extract", ColumnExtractorTransformtion()),
                 ("Imputer", Imputer()),
                 ("Scalar", StandardScaler())
             ]
    # Features_Union = FeatureUnion(
    #     transformer_list=[
    #         ("text_Features",Pipeline(TextFeaturesOnly)),
    #         ("Numerical_Features",Pipeline(NumericalFeaturesOnly))
    #     ])
    tuned_parametersNumericalOnlyDT=[
        {
            'clf__n_neighbors':[3,6,9,10,15,20],
            'extract__cols':
                #"NumOfTweets","NumOfFavourites","RetweetCount","ReplyCount"
                [
                    ["NumOfFollowers"], ["NumOfFollowing"], ["NumOfTweetsPerDay"],
                    ["NumOfFollowers", "NumOfFollowing"],
                    ["NumOfFollowers", "NumOfTweetsPerDay"],
                    [ "NumOfFollowing", "NumOfTweetsPerDay"],
                    ["NumOfFollowers", "NumOfFollowing", "NumOfTweetsPerDay"]
                ],
        },
        ]
    tuned_parametersTextOnlyDT=[
        {
            'clf__n_neighbors':[3,6,9,10,15,20],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            'extractTextUsed__Path': [
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_Stem",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords_Stem",
            ],
        },
        ]
    tuned_parametersUnionDT= [
        {
            'clf__n_neighbors': [3,6,9,10,15,20],
            'Features__text_Features__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            'Features__text_Features__extractTextUsed__Path': [
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_Stem",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords_Stem",
            ],
            'Features__Numerical_Features__extract__cols':
                [["NumOfFollowers"], ["NumOfFollowing"], ["NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing"],["NumOfFollowers", "NumOfTweetsPerDay"],[ "NumOfFollowing", "NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing", "NumOfTweetsPerDay"]],
        },
        ]
    pipeTN = imbPipeline([("Features", FeatureUnion(
        transformer_list=[
            ("text_Features", Pipeline(TextFeaturesOnly)),
            ("Numerical_Features", Pipeline(NumericalFeaturesOnly))])),
                        ('oversample', SMOTE(random_state=20)),
                        ('clf', KNeighborsClassifier())
                        ])
    pipeT = imbPipeline([
        ("extractTextUsed", ExtractTextTransform(ColumnO='AUX', ColumnI='ID')),
        ("tfidf", TfidfVectorizer(stop_words=None)),
        ('oversample', SMOTE(random_state=20)),
        ('clf', KNeighborsClassifier())
    ])
    pipeN = imbPipeline([
        ("extract", ColumnExtractorTransformtion()),
        ("Imputer", Imputer()),
        ("Scalar", StandardScaler()),
        ('oversample', SMOTE(random_state=20)),
        ('clf', KNeighborsClassifier())
    ])
    gridSearchT = GridSearchCV(
        pipeT,
        param_grid=tuned_parametersTextOnlyDT,
        cv=5,
        n_jobs=-1,
        scoring="f1")
    gridSearchN = GridSearchCV(
        pipeN,
        param_grid=tuned_parametersNumericalOnlyDT,
        cv=5,
        n_jobs=-1,
        scoring="f1")
    gridSearchTN = GridSearchCV(
        pipeTN,
        param_grid=tuned_parametersUnionDT,
        cv=5,
        n_jobs=-1,
        scoring="f1")
    #gridSearchN.fit(trainSamples, TrainTargets)
    gridSearchT.fit(trainSamples,TrainTargets)
    #gridSearchTN.fit(trainSamples,TrainTargets)
    os.chdir("C:\\Users\Marwa\\PycharmProjects\FinalISA\\")
    #pd.DataFrame(gridSearchN.cv_results_).sort_values(by='rank_test_score').to_csv('Results_GridSearch_Numerical_knn_Over_'+LabelOfTarget+'.csv')
    pd.DataFrame(gridSearchT.cv_results_).sort_values(by='rank_test_score').to_csv('Results_GridSearch_Text_knn_Over_'+LabelOfTarget+'.csv')
    #pd.DataFrame(gridSearchTN.cv_results_).sort_values(by='rank_test_score').to_csv('Results_GridSearch_BothFeatures_knn_Over_'+LabelOfTarget+'.csv')
    #joblib.dump(gridSearchN.best_estimator_,'KNNN_Over_'+LabelOfTarget+'.pkl')
    joblib.dump(gridSearchT.best_estimator_,'KNNT_Over_'+LabelOfTarget+'.pkl')
    #joblib.dump(gridSearchTN.best_estimator_,'KNNTN_Over_'+LabelOfTarget+'.pkl')
    #clf=joblib.load('KNNN_Over_'+LabelOfTarget+'.pkl')
    #clf=joblib.load('KNNT_Over_'+LabelOfTarget+'.pkl')
    #clf=joblib.load('KNNTN_Over_'+LabelOfTarget+'.pkl')
