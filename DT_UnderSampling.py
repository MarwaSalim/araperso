import os
from imblearn.under_sampling import ClusterCentroids
from customerTransformation import ExtractTextTransform,ColumnExtractorTransformtion
from sklearn.preprocessing import (Imputer, StandardScaler)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import (Pipeline, FeatureUnion)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from imblearn.pipeline import Pipeline as imbPipeline
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
            'extract__cols':
                [["NumOfFollowers"], ["NumOfFollowing"], ["NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing"],["NumOfFollowers", "NumOfTweetsPerDay"],[ "NumOfFollowing", "NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing", "NumOfTweetsPerDay"]],
        },
        ]
    tuned_parametersTextOnlyDT=[
        {
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
            'Features__text_Features__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            'Features__text_Features__extractTextUsed__Path': [
                "C:\\Users\Marwa\\PycharmProjects\\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant",
                "C:\\Users\Marwa\\PycharmProjects\\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_Stem",
                "C:\\Users\Marwa\\PycharmProjects\\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords",
                "C:\\Users\Marwa\\PycharmProjects\\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords_Stem",
            ],
            'Features__Numerical_Features__extract__cols':
                [["NumOfFollowers"], ["NumOfFollowing"], ["NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing"],["NumOfFollowers", "NumOfTweetsPerDay"],[ "NumOfFollowing", "NumOfTweetsPerDay"],
                 ["NumOfFollowing", "NumOfFollowing", "NumOfTweetsPerDay"]],
        },
        ]
    pipeTN = imbPipeline([("Features", FeatureUnion(
        transformer_list=[
            ("text_Features", Pipeline(TextFeaturesOnly)),
            ("Numerical_Features", Pipeline(NumericalFeaturesOnly))])),
                          ('oversample', ClusterCentroids(random_state=20)),
                          ('clf', DecisionTreeClassifier())
                          ])
    pipeT = imbPipeline([
        ("extractTextUsed", ExtractTextTransform(ColumnO='AUX', ColumnI='ID')),
        ("tfidf", TfidfVectorizer(stop_words=None)),
        ('oversample', ClusterCentroids(random_state=20)),
        ('clf', DecisionTreeClassifier())
    ])
    pipeN = imbPipeline([
        ("extract", ColumnExtractorTransformtion()),
        ("Imputer", Imputer()),
        ("Scalar", StandardScaler()),
        ('oversample', ClusterCentroids(random_state=20)),
        ('clf', DecisionTreeClassifier())
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
    gridSearchT.fit(trainSamples,TrainTargets)
    #gridSearchN.fit(trainSamples,TrainTargets)
    #gridSearchTN.fit(trainSamples,TrainTargets)
    os.chdir("C:\\Users\Marwa\\PycharmProjects\FinalISA\\")
    #pd.DataFrame(gridSearchN.cv_results_).sort_values(by='rank_test_score').to_csv(
    #     'Results_GridSearch_Numerical_DT_Under_' + LabelOfTarget + '.csv')
    #joblib.dump(gridSearchN.best_estimator_, './DTN_Under_'+LabelOfTarget+'.pkl')
    pd.DataFrame(gridSearchT.cv_results_).sort_values(by='rank_test_score').to_csv(
         'Results_GridSearch_Text_DT_Under_' + LabelOfTarget + '.csv')
    joblib.dump(gridSearchT.best_estimator_, './DTT_Under_'+LabelOfTarget+'.pkl')
    #pd.DataFrame(gridSearchTN.cv_results_).sort_values(by='rank_test_score').to_csv('Results_GridSearch_BothFeatures_DT_Under_'+LabelOfTarget+'.csv')
    #joblib.dump(gridSearchTN.best_estimator_,'./DTTN_Under_'+LabelOfTarget+'.pkl')
    # with open('DTT.pkl', 'wb') as f:
    #     pickl/e.dump(gridSearchT.best_estimator_, f)
    # feature_names = gridSearchT.best_estimator_.named_steps['text_Features'].named_steps['tfidf'].get_feature_names()
    # for f in feature_names:
    #     print f
    # clf=joblib.load('./DTN_Under_'+LabelOfTarget+'.pkl')
    # clf=joblib.load('./DTT_Under_'+LabelOfTarget+'.pkl')
    # clf=joblib.load('./DTTN_Under_'+LabelOfTarget+'.pkl')
