import os
from customerTransformation import ExtractTextTransform,ColumnExtractorTransformtion
from sklearn.preprocessing import (Imputer, StandardScaler)
from sklearn.pipeline import (Pipeline, FeatureUnion)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.externals import joblib
import pandas as pd
def Begin (df,LabelOfTarget):
    trainSamples=df
    TrainTargets = trainSamples[LabelOfTarget]
    TextFeaturesOnly=[
        ("extractTextUsed",ExtractTextTransform(ColumnO='AUX',ColumnI='ID')),
        ("tfidf", TfidfVectorizer(stop_words=None)),
    ]
    NumericalFeaturesOnly=[
                 ("extract", ColumnExtractorTransformtion()),
                 ("Imputer", Imputer()),
                 ("Scalar", StandardScaler())
             ]
    Features_Union = FeatureUnion(
        transformer_list=[
            ("text_Features",Pipeline(TextFeaturesOnly)),
            ("Numerical_Features",Pipeline(NumericalFeaturesOnly))
        ])
    tuned_parametersNumericalOnlySVM=[
        {
            'clf__kernel': ['rbf'],
            'clf__gamma': [1e-2, 1e-3, 1e-4, 1e-5],
            'clf__C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000],
            'Numerical_Features__extract__cols':
                [["NumOfFollowers"], ["NumOfFollowing"], ["NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing"],["NumOfFollowers", "NumOfTweetsPerDay"],[ "NumOfFollowing", "NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing", "NumOfTweetsPerDay"]],
        },
        {
            'clf__kernel': ['sigmoid'],
            'clf__gamma': [1e-2, 1e-3, 1e-4, 1e-5],
            'clf__C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000],
            'Numerical_Features__extract__cols':
                [["NumOfFollowers"], ["NumOfFollowing"], ["NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing"],["NumOfFollowers", "NumOfTweetsPerDay"],[ "NumOfFollowing", "NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing", "NumOfTweetsPerDay"]],
        },
        {
            'clf__kernel': ['linear'],
            'clf__C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000],
            'Numerical_Features__extract__cols':
                [["NumOfFollowers"], ["NumOfFollowing"], ["NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing"],["NumOfFollowers", "NumOfTweetsPerDay"],[ "NumOfFollowing", "NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing", "NumOfTweetsPerDay"]],
        }
        ]
    tuned_parametersTextOnlySVM=[
        {
            'text_Features__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            'clf__kernel': ['rbf'],
            'clf__gamma': [1e-2, 1e-3, 1e-4, 1e-5],
            'clf__C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000],
            'text_Features__extractTextUsed__Path':[
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_Stem",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords_Stem",
            ],
        },
        {
            'text_Features__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            'clf__kernel': ['sigmoid'],
            'clf__gamma': [1e-2, 1e-3, 1e-4, 1e-5],
            'clf__C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000],
            'text_Features__extractTextUsed__Path':[
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_Stem",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords_Stem",
            ],
        },
        {
            'text_Features__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            'clf__kernel': ['linear'],
            'clf__C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000],
            'text_Features__extractTextUsed__Path':[
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_Stem",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords_Stem",
            ],
        }
        ]
    tuned_parametersUnionSVM = [
        {
            'Features__text_Features__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            'clf__kernel': ['rbf'],
            'clf__gamma': [1e-2, 1e-3, 1e-4, 1e-5],
            'clf__C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000],
            'Features__text_Features__extractTextUsed__Path':[
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
        {
            'Features__text_Features__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            'clf__kernel': ['sigmoid'],
            'clf__gamma': [1e-2, 1e-3, 1e-4, 1e-5],
            'clf__C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000],
            'Features__text_Features__extractTextUsed__Path':[
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
        {
            'Features__text_Features__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            'clf__kernel': ['linear'],
            'clf__C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000],
            'Features__text_Features__extractTextUsed__Path':[
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_Stem",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords_Stem",
                ],
            'Features__Numerical_Features__extract__cols':
                [["NumOfFollowers"], ["NumOfFollowing"], ["NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing"],["NumOfFollowers", "NumOfTweetsPerDay"],[ "NumOfFollowing", "NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing", "NumOfTweetsPerDay"]],
        }
        ]
    gridSearchT = GridSearchCV(
        estimator=Pipeline(steps=[
            ("text_Features",Pipeline(TextFeaturesOnly)),
            ("clf", SVC()),
        ]),
        param_grid=tuned_parametersTextOnlySVM,
        cv=5,
        n_jobs=-1,
        scoring="f1")
    gridSearchN = GridSearchCV(
        estimator=Pipeline(steps=[
            ('Numerical_Features', Pipeline(NumericalFeaturesOnly)),
            ("clf", SVC())
        ]),
        param_grid=tuned_parametersNumericalOnlySVM,
        cv=5,
        n_jobs=-1,
        scoring="f1")
    gridSearchTN = GridSearchCV(
        estimator=Pipeline(steps=[
            ('Features', Features_Union),
            ("clf", SVC())
        ]),
        param_grid=tuned_parametersUnionSVM,
        cv=5,
        n_jobs=-1,
        scoring="f1")
    #gridSearchN.fit(trainSamples,TrainTargets)
    gridSearchT.fit(trainSamples,TrainTargets)
    #gridSearchTN.fit(trainSamples,TrainTargets)
    os.chdir("C:\\Users\Marwa\\PycharmProjects\FinalISA\\")
    #pd.DataFrame(gridSearchN.cv_results_).sort_values(by='rank_test_score').to_csv('Results_GridSearch_Numerical_SVM_'+LabelOfTarget+'.csv')
    pd.DataFrame(gridSearchT.cv_results_).sort_values(by='rank_test_score').to_csv('Results_GridSearch_Text_SVM_'+LabelOfTarget+'.csv')
    #pd.DataFrame(gridSearchTN.cv_results_).sort_values(by='rank_test_score').to_csv('Results_GridSearch_BothFeatures_SVM_'+LabelOfTarget+'.csv')
    #joblib.dump(gridSearchN.best_estimator_,'SVMN_'+LabelOfTarget+'.pkl')
    joblib.dump(gridSearchT.best_estimator_,'SVMT_'+LabelOfTarget+'.pkl')
    #joblib.dump(gridSearchTN.best_estimator_,'SVMTN_'+LabelOfTarget+'.pkl')
    # clf=joblib.load('SVMN_'+LabelOfTarget+'.pkl')
    # clf=joblib.load('SVMT_'+LabelOfTarget+'.pkl')
    # clf=joblib.load('SVMTN_'+LabelOfTarget+'.pkl')
