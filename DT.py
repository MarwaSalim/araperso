import os
from customerTransformation import ExtractTextTransform,ColumnExtractorTransformtion
from sklearn.preprocessing import (Imputer, StandardScaler)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import (Pipeline, FeatureUnion)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
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
    Features_Union = FeatureUnion(
        transformer_list=[
            ("text_Features",Pipeline(TextFeaturesOnly)),
            ("Numerical_Features",Pipeline(NumericalFeaturesOnly))
        ])
    Features=[TextFeaturesOnly,NumericalFeaturesOnly,Features_Union]
    tuned_parametersNumericalOnlyDT=[
        {
            'Numerical_Features__extract__cols':
                [["NumOfFollowers"], ["NumOfFollowing"], ["NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing"],["NumOfFollowers", "NumOfTweetsPerDay"],[ "NumOfFollowing", "NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing", "NumOfTweetsPerDay"]],
        },
        ]
    tuned_parametersTextOnlyDT=[
        {
            'text_Features__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            'text_Features__extractTextUsed__Path': [
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
    gridSearchT = GridSearchCV(
        estimator=Pipeline(steps=[
            ("text_Features",Pipeline(TextFeaturesOnly)),
            ("clf", DecisionTreeClassifier()),
        ]),
        param_grid=tuned_parametersTextOnlyDT,
        cv=5,
        n_jobs=-1,
        scoring="f1")
    gridSearchN = GridSearchCV(
        estimator=Pipeline(steps=[
            ('Numerical_Features', Pipeline(NumericalFeaturesOnly)),
            ("clf", DecisionTreeClassifier())
        ]),
        param_grid=tuned_parametersNumericalOnlyDT,
        cv=5,
        n_jobs=-1,
        scoring="f1")
    gridSearchTN = GridSearchCV(
        estimator=Pipeline(steps=[
            ('Features', Features_Union),
            ("clf", DecisionTreeClassifier())
        ]),
        param_grid=tuned_parametersUnionDT,
        cv=5,
        n_jobs=-1,
        scoring="f1")
    gridSearchT.fit(trainSamples,TrainTargets)
    #gridSearchN.fit(trainSamples,TrainTargets)
    #gridSearchTN.fit(trainSamples,TrainTargets)
    os.chdir("C:\\Users\Marwa\\PycharmProjects\FinalISA\\")
    #pd.DataFrame(gridSearchN.cv_results_).sort_values(by='rank_test_score').to_csv(
    #     'Results_GridSearch_Numerical_DT_' + LabelOfTarget + '.csv')
    #joblib.dump(gridSearchN.best_estimator_, './DTN_'+LabelOfTarget+'.pkl')
    pd.DataFrame(gridSearchT.cv_results_).sort_values(by='rank_test_score').to_csv(
         'Results_GridSearch_Text_DT_' + LabelOfTarget + '.csv')
    joblib.dump(gridSearchT.best_estimator_, './DTT_'+LabelOfTarget+'.pkl')
    #pd.DataFrame(gridSearchTN.cv_results_).sort_values(by='rank_test_score').to_csv('Results_GridSearch_BothFeatures_DT_'+LabelOfTarget+'.csv')
    #joblib.dump(gridSearchTN.best_estimator_,'./DTTN_'+LabelOfTarget+'.pkl')
    # with open('DTT.pkl', 'wb') as f:
    #     pickl/e.dump(gridSearchT.best_estimator_, f)
    # feature_names = gridSearchT.best_estimator_.named_steps['text_Features'].named_steps['tfidf'].get_feature_names()
    # for f in feature_names:
    #     print f
    #clf=joblib.load('./DTN_'+LabelOfTarget+'.pkl')
    #clf=joblib.load('./DTT_'+LabelOfTarget+'.pkl')
    #clf=joblib.load('./DTTN_'+LabelOfTarget+'.pkl')
