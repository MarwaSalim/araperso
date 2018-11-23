import os
from imblearn.under_sampling import ClusterCentroids
from imblearn.pipeline import Pipeline as imbPipeline
from customerTransformation import ExtractTextTransform,ColumnExtractorTransformtion
from sklearn.preprocessing import (Imputer, StandardScaler)
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import (Pipeline, FeatureUnion)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from SaveResult import classifaction_report_csv
import pandas as pd
def Begin (Path,LabelOfTarget):
    df = pd.read_csv(Path)
    trainSamples, testSamples = train_test_split(df,test_size=0.34)
    TrainTargets = trainSamples[LabelOfTarget]
    TestTargets=testSamples[LabelOfTarget]
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
            'clf__hidden_layer_sizes':[(30,20,10),(60,50,40),(70,80,90)],
            'clf__activation':['identity', 'logistic', 'tanh', 'relu'],
            'extract__cols':
                [["NumOfFollowers"], ["NumOfFollowing"], ["NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing"],["NumOfFollowers", "NumOfTweetsPerDay"],[ "NumOfFollowing", "NumOfTweetsPerDay"],
                 ["NumOfFollowers", "NumOfFollowing", "NumOfTweetsPerDay"]],
        },
        ]
    tuned_parametersTextOnlyDT=[
        {
            'clf__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'clf__hidden_layer_sizes': [(30, 20, 10), (60, 50, 40), (70, 80, 90)],
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
            'clf__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'clf__hidden_layer_sizes': [(30, 20, 10), (60, 50, 40), (70, 80, 90)],
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
                          ('oversample', ClusterCentroids(random_state=20)),
                          ('clf', MLPClassifier())
                          ])
    pipeT = imbPipeline([
        ("extractTextUsed", ExtractTextTransform(ColumnO='AUX', ColumnI='ID')),
        ("tfidf", TfidfVectorizer(stop_words=None)),
        ('oversample', ClusterCentroids(random_state=20)),
        ('clf', MLPClassifier())
    ])
    pipeN = imbPipeline([
        ("extract", ColumnExtractorTransformtion()),
        ("Imputer", Imputer()),
        ("Scalar", StandardScaler()),
        ('oversample', ClusterCentroids(random_state=20)),
        ('clf', MLPClassifier())
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
    gridSearchN.fit(trainSamples, TrainTargets)
    gridSearchT.fit(trainSamples,TrainTargets)
    gridSearchTN.fit(trainSamples,TrainTargets)
    os.chdir("C:\\Users\Marwa\\PycharmProjects\FinalISA\\")
    #pd.DataFrame(gridSearchN.cv_results_).sort_values(by='rank_test_score').to_csv('Results_GridSearch_Numerical_MLP_'+LabelOfTarget+'.csv')
    pd.DataFrame(gridSearchT.cv_results_).sort_values(by='rank_test_score').to_csv('Results_GridSearch_Text_MLP_'+LabelOfTarget+'.csv')
    #pd.DataFrame(gridSearchTN.cv_results_).sort_values(by='rank_test_score').to_csv('Results_GridSearch_BothFeatures_MLP_'+LabelOfTarget+'.csv')
    #joblib.dump(gridSearchN.best_estimator_,'MLPN_'+LabelOfTarget+'.pkl')
    joblib.dump(gridSearchT.best_estimator_,'MLPT_'+LabelOfTarget+'.pkl')
    #joblib.dump(gridSearchTN.best_estimator_,'MLPTN_'+LabelOfTarget+'.pkl')
    # clf=joblib.load('MLPN_'+LabelOfTarget+'.pkl')
    clf=joblib.load('MLPT_'+LabelOfTarget+'.pkl')
    # clf=joblib.load('MLPTN_'+LabelOfTarget+'.pkl')
    predictions = gridSearchT.predict(testSamples)
    print classification_report(TestTargets, predictions)
Begin("DataSet_11_Nov_Tweets_csv1.csv","O")
Begin("DataSet_11_Nov_Tweets_csv1.csv","C")
Begin("DataSet_11_Nov_Tweets_csv1.csv","E")
Begin("DataSet_11_Nov_Tweets_csv1.csv","A")
Begin("DataSet_11_Nov_Tweets_csv1.csv","N")