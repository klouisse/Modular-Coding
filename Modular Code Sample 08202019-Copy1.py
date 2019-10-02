
# coding: utf-8

# # Sample Functions

# Import Packages

# In[64]:




import pandas as pd
import numpy as np
import matplotlib as plt
from varclushi import VarClusHi 
import scikitplot as skplt
import time

from scipy.stats import uniform as flt
from scipy.stats import randint as itr


#Algorithm
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold , cross_val_score, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import  roc_auc_score, brier_score_loss, precision_score, recall_score, roc_curve
from sklearn.calibration import calibration_curve
from matplotlib import pyplot
import pickle
import lightgbm as lgb

from lightgbm import LGBMClassifier
                                                                                                        
#Explainer

import shap


# Import Dataset
# 

# In[103]:


def get_data(data):
    """Used to import dataset
    
    Returns: Input Datframe
    """
    return pd.read_sas(data)


# Data Preprocessing

# In[104]:


def data_preparation(data_in, data_index, remove_list, target):
    
    """Prepare dataset for feature generation
    
    Steps:
    1. Set index to unique ID in column
    2. Drop unnecessary columns from source
    4. Creates Feature Dataframe and Target Variable
    
     Args:
 
    data_in: input data
    index: input unique ID in data
    list: list of columns to drop
    y: target variable

    Returns:

    tuple containing 0: dataframe , 1:target
    """
  
    data_ft = data_in.set_index(data_in[data_index], inplace = True)
    data_ft = data_in.drop(columns=remove_list)
    target =  data_in[target]

    return data_ft, target


# Data Model: Train and Test Set
# 

# In[105]:


def data_partition(df_features, target, seed=123, prop=0.3):
    """Define Train and Test Set
    
    Args:
    
    1. df_features: dataframe of features
    2. Target: Target Series
    3. Seed: Random Seed default values
    4. Prop: Test proportion
    
    Returns:
    
    Tuple of Train and Test Sets
    """
    
    x_train, x_test, y_train, y_test = train_test_split(df_features, target, stratify = target,
                                                       random_state=seed, test_size=prop)
    return x_train, x_test, y_train, y_test


# Feature Selection 

# In[106]:


def feature_generation(df_features, target, corr_lmt = 0.9 ,eigval=1, n_clus=None):
    
    """Apply Feature Selection 
    
    1. Imputes NaN in Dataset
    2. Remove features with Constant variance
    3. Shortlists based on Correlation
    4. Performs Variable Clustering using VarclusHi function
    
    Args:
    
    1. train_set: dataframe of features in training data
    2. train_resp: target variable
    3. corr_lmt : max correlation to consider on shotlist. Default is 0.9
    4. eigval : max eigenvalue when doing clustering. Default is 1
    5. n_clus : Clusters to be considered. Default value is none - builds all possible clusters
    
    
    Returns:

    Tuple of Shortlisted dataset, shortlisted variables
    """
    
    #Impute
    df_features[df_features.isnull()] = 0
    
    #Remove Constant
    constant_filter = VarianceThreshold(threshold = 0)
    constant_filter.fit(df_features)
    nonconstant = df_features.columns[constant_filter.get_support()].tolist()
    df_features = df_features[nonconstant]
    
    #2 Remove Correlated variables
    for_corr =[df_features, target]
    for_corr = pd.concat((df_features, target), axis=1, join='inner')
    corr = abs(for_corr.corr().iloc[:,-1])
    corr = corr[(corr < corr_lmt)]
    corr_slist = corr.index.tolist()
    df_features = df_features[corr_slist]
    
    #3 Variable Clustering using VarClusHi
    
    cluster = VarClusHi(df_features, maxeigval2 = eigval, maxclus = n_clus)
    cluster.varclus()
    # Get top variables per cluster
    list1 = cluster.rsquare.loc[cluster.rsquare.groupby(["Cluster"])['RS_Ratio'].idxmin()]
    list1 = list1['Variable'].tolist()
    df_features = df_features[list1]
    
    
    return df_features, list1


# Model Selection

# In[107]:


def model_selection(x_train, y_train, x_test, y_test):
    """Perform Classification Model Selection
    
    Args:
    
    1. x_train = training data feature
    2. y_train = training data label
    3. x_test = test data feature
    4. y_test = test data label
    
    Returns:
    
    Prints model scoring for model selection
    """
    

    pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(random_state=123, solver = 'liblinear' ))])

    pipe_rf = Pipeline([('classifier', RandomForestClassifier(random_state=123 , n_estimators = 1000))])


    pipe_lgb = Pipeline([('lgb' , lgb.LGBMClassifier())])


    pipe_rlgb = Pipeline([('lgb' , lgb.LGBMClassifier())])

    cv_params = {'lgb__learning_rate': flt(0.001, 0.2),
             'lgb__max_depth': [3, 10],
             'lgb__min_data_in_leaf': [20, 100],
             'lgb__colsample_by_tree':flt(0.5,0.9),
             'lgb__min_child_weight' :[1,3],
             'lgb__bagging_fraction' :flt(0, 1)
             }


    rs_pipe_lgb = RandomizedSearchCV(estimator=pipe_rlgb,
                                 param_distributions=cv_params,
                                 scoring='roc_auc',
                                 n_iter =10,
                                 cv=10,
                                 random_state=123)

    pipelines = [pipe_lr, pipe_rf, pipe_lgb, rs_pipe_lgb]


    start_time = time.time()
    params =[]
    for p in pipelines:
        params =  p.fit(x_train, y_train)
        
        print('Fit time: %s' % (time.time() - start_time))
        print("model score: %.3f" % p.score(x_test[shortlist], y_test))
                                        
    return 


# Model Evaluation

# In[108]:


def model_fit(x_train, y_train, x_test, y_test):
    """Fits chosen model in dataset
    
    Args:
    
    1. x_train = training data feature
    2. y_train = training data label
    3. x_test = test data feature
    4. y_test = test data label
    
    Returns:
    
    1. Fitted Model
    2. Predicted Group
    3. Predicted Probability
    4. Probability Cutoff for Model Selection
    5. Model Performance Metrics
    """
    
    #Apply Transformation to x_test
    
    x_test[x_test.isnull()] = 0
    
    #Raw Scores
    model = clf.fit(x_train, y_train
                   ,early_stopping_rounds =100, verbose=False, eval_metric = 'logloss'
                   ,eval_set = (x_test, y_test))
    
    p_raw = model.predict_proba(x_train)
    
    cutoff = np.around(np.percentile(p_raw[:,1], np.arange(0, 100, 10)), decimals = 7)
    
    #Predicted Group
    p_grp = model.predict(x_test)
    metrics        = {"ROC" : roc_auc_score(y_test, p_grp),
                      "Precision" : precision_score(y_test, p_grp),
                      "Recall": recall_score(y_test, p_grp)}
   
    #Save Model
    
    pickle.dump(model, open("lapse_model_mod.p" , "wb"))

    return model, p_grp, p_raw, cutoff, metrics


# Model Validation

# In[115]:


def model_prediction(df, target, fitted_model):
    
    """Evaluates model on Validation Set
    
    Args:
    
    1. df = validation set
    2. target = response in validation set
    3. model = fitted model

    
    Returns:
    
    1. test data with predicted prob [p_1], decile score [decile] and original target
    2. test data without original target

    """
    
    model = pickle.load(open(fitted_model , "rb"))
    df['p_1'] = model.predict_proba(df)[:,1]
    
    conditions = [
    (df['p_1'] <   0.0339419),
    (df['p_1'] >=  0.0339419) & (df['p_1'] < 0.0411623),
    (df['p_1'] >=  0.0411623) & (df['p_1'] < 0.0544696),
    (df['p_1'] >=  0.0544696) & (df['p_1'] < 0.0794567),
    (df['p_1'] >=  0.0794567) & (df['p_1'] < 0.1010139),
    (df['p_1'] >=  0.1010139) & (df['p_1'] < 0.1374370),
    (df['p_1'] >=  0.1374370) & (df['p_1'] < 0.1973212),
    (df['p_1'] >=  0.1973212) & (df['p_1'] < 0.3045569),
    (df['p_1'] >=  0.3045569) & (df['p_1'] < 0.6198479),
    (df['p_1'] >=  0.6198479)]
    choices = [10,9,8,7,6,5,4,3,2,1]

    df['decile'] = np.select(conditions, choices, default='null').astype(int)
   
    val_list = [df , target]
    validation_sets = val_list[0].join(val_list[1])
    
    return validation_sets, df


# Reports Generation

# In[119]:


def generate_report(data, y):
    """Generates Decile Analysis
    
    Args:
    
    1. data = data for decile analysis
    2. y = target variable
    
    Returns:
    
    1. decile analysis

    """
    
    data['nonresp'] = 1-data[y]
    
    deciles = pd.pivot_table(data=data,index=['decile'],
                             values=[y,'nonresp'],
                             aggfunc={y:[np.sum],
                                     'nonresp':[np.sum]})
    
    deciles.to_csv(r'C:\Users\martken\Documents\sample.csv')
    
    return deciles


# # Apply Functions to Projects

# In[109]:


input_data = get_data("C:/Users/file.csv")


# In[110]:


#Prepare Input data

#Columns to drop

remove_list = []

feature_data , target =data_preparation(input_data, 'id', remove_list , target ='y')


# In[111]:


x_train, x_test, y_train, y_test = data_partition(feature_data , target)


# In[112]:


x_train, shortlist = feature_generation(x_train, y_train, 0.9)
x_test = x_test[shortlist]


# In[113]:


pipelines = model_selection(x_train, y_train, x_test, y_test)


# In[114]:


#Use LGBM as Final Model

clf = LGBMClassifier(random_state=42)
model, pred_grp, pred_prob, p_cutoff, metrics= model_fit(x_train ,y_train, x_test, y_test)
print(metrics)
print(p_cutoff)


# In[116]:


#Load Model

labeled_test, pred_test = model_prediction(x_test, y_test, fitted_model = "lapse_model_mod.p" )


# In[120]:


decile_rep = generate_report(labeled_test, y='f_ever_lapse15')
print(decile_rep)

