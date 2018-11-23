import os
from customerTransformation import ExtractTextTransform,ColumnExtractorTransformtion
from sklearn.preprocessing import (Imputer, StandardScaler)
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import (Pipeline, FeatureUnion)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from SaveResult import classifaction_report_csv
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
    Features_Union = FeatureUnion(
        transformer_list=[
            ("text_Features",Pipeline(TextFeaturesOnly)),
            ("Numerical_Features",Pipeline(NumericalFeaturesOnly))
        ])
    tuned_parametersNumericalOnlyMNB=[
        {
            'clf__alpha': [0.2,0.4,0.6,0.8,1],
            'Numerical_Features__extract__cols':
                [["NumOfFollowers"], ["NumOfFollowing"], ["NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing"],["NumOfFollowers", "NumOfTweetsPerDay"],[ "NumOfFollowing", "NumOfTweetsPerDay"],
                 ["NumOfFollowing", "NumOfFollowing", "NumOfTweetsPerDay"]],
        },
        ]
    tuned_parametersTextOnlyMNB=[
        {
            'text_Features__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            'clf__alpha': [0.2,0.4,0.6,0.8,1],
            'text_Features__extractTextUsed__Path': [
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_Stem",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords",
                "C:\\Users\Marwa\\PycharmProjects\FinalISA\\DataSet_10_Nov_Clean_Normalize_Redundant_StopWords_Stem",
            ],
        },
        ]
    tuned_parametersUnionMNB= [
        {
            'Features__text_Features__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            'clf__alpha': [0.2,0.4,0.6,0.8,1],
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
    gridSearchT = GridSearchCV(
        estimator=Pipeline(steps=[
            ("text_Features",Pipeline(TextFeaturesOnly)),
            ("clf", MultinomialNB()),
        ]),
        param_grid=tuned_parametersTextOnlyMNB,
        cv=10,
        n_jobs=-1,
        scoring="f1")
    gridSearchN = GridSearchCV(
        estimator=Pipeline(steps=[
            ('Numerical_Features', Pipeline(NumericalFeaturesOnly)),
            ("clf", MultinomialNB())
        ]),
        param_grid=tuned_parametersNumericalOnlyMNB,
        cv=10,
        n_jobs=-1,
        scoring="f1")
    gridSearchTN = GridSearchCV(
        estimator=Pipeline(steps=[
            ('Features', Features_Union),
            ("clf", MultinomialNB())
        ]),
        param_grid=tuned_parametersUnionMNB,
        cv=10,
        n_jobs=-1,
        scoring="f1")
    #gridSearchN.fit(trainSamples,TrainTargets)
    gridSearchT.fit(trainSamples,TrainTargets)
    #gridSearchTN.fit(trainSamples,TrainTargets)
    os.chdir("C:\\Users\Marwa\\PycharmProjects\FinalISA\\")
    #pd.DataFrame(gridSearchN.cv_results_).sort_values(by='rank_test_score').to_csv('Results_GridSearch_Numerical_MNB_Under_'+LabelOfTarget+'.csv')
    pd.DataFrame(gridSearchT.cv_results_).sort_values(by='rank_test_score').to_csv('Results_GridSearch_Text_MNB_'+LabelOfTarget+'.csv')
    #pd.DataFrame(gridSearchTN.cv_results_).sort_values(by='rank_test_score').to_csv('Results_GridSearch_BothFeatures_MNB_Under_'+LabelOfTarget+'.csv')
    #joblib.dump(gridSearchN.best_estimator_,'MNBN_Under_'+LabelOfTarget+'.pkl')
    joblib.dump(gridSearchT.best_estimator_,'MNBT_'+LabelOfTarget+'.pkl')
    #joblib.dump(gridSearchTN.best_estimator_,'MNBTN_Under_'+LabelOfTarget+'.pkl')
    #clf=joblib.load('MNBN_Under_'+LabelOfTarget+'.pkl')
    #clf=joblib.load('MNBT_Under_'+LabelOfTarget+'.pkl')
    #clf=joblib.load('MNBTN_Under_'+LabelOfTarget+'.pkl')
    #predictions = gridSearchT.predict(testSamples)
    #report= classification_report(TestTargets, predictions)
    #conf=confusion_matrix(TestTargets,predictions,labels=[0,1])
    # try:
    #     classifaction_report_csv(report,conf,"MNBT_BestEstimatorTesting_"+LabelOfTarget)
    # except:
    #     print report
    #     print conf
