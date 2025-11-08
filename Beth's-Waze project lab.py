#!/usr/bin/env python
# coding: utf-8

# # **Waze Project**
# 

# Your team is close to completing their user churn project. Previously, you completed a project proposal, and used Python to explore and analyze Waze’s user data, create data visualizations, and conduct a hypothesis test. Most recently, you built a binomial logistic regression model based on multiple variables.
# 
# Leadership appreciates all your hard work. Now, they want your team to build a machine learning model to predict user churn. To get the best results, your team decides to build and test two tree-based models: random forest and XGBoost.
# 
# Your work will help leadership make informed business decisions to prevent user churn, improve user retention, and grow Waze’s business.
# 

# # **Build a machine learning model**
# 

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 
# In this stage, consider the following questions:
# 
# 1.   What are you being asked to do?
# > _Predict if a customer will churn or be retained._
# 
# 2.   What are the ethical implications of the model? What are the consequences of your model making errors?
#   *   What is the likely effect of the model when it predicts a false negative (i.e., when the model says a Waze user won't churn, but they actually will)?
#   > _Waze will fail to take proactive measures to retain users who are likely to stop using the app. For example, Waze might proactively push an app notification to users, or send a survey to better understand user dissatisfaction._
#   *   What is the likely effect of the model when it predicts a false positive (i.e., when the model says a Waze user will churn, but they actually won't)?
#   > _Waze may take proactive measures to retain users who are NOT likely to churn. This may lead to an annoying or negative experience for loyal users of the app._
# 3.   Do the benefits of such a model outweigh the potential problems?
#   > _The proactive measueres taken by Waze might have unintended effects on users, and these effects might encourage user churn. Follow-up analysis on the effectiveness of the measures is recommended. If the measures are reasonable and effective, then the benefits will most likely outweigh the problems._
# 4.   Would you proceed with the request to build this model? Why or why not?
# 
#   >_Yes. There aren't any significant risks for building such a model._
# 

# ### **Task 1. Imports and data loading**
# 
# Import packages and libraries needed to build and evaluate random forest and XGBoost classification models.

# In[32]:


# Import packages for data manipulation
import numpy as np
import pandas as pd

# Import packages for data visualization
import matplotlib.pyplot as plt

# This lets us see all of the columns, preventing Juptyer from redacting them.
pd.set_option('display.max_columns', None)

# Import packages for data modeling
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# This is the function that helps plot feature importance
from xgboost import plot_importance

# This module lets us save our models once we fit them.
import pickle


# Now read in the dataset as `df0` and inspect the first five rows.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[33]:


# Import dataset
df0 = pd.read_csv('waze_dataset.csv')


# In[34]:


# Inspect the first five rows
df0.head()


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2. Feature engineering**

# In[35]:


# Copy the df0 dataframe
df = df0.copy()


# Call `info()` on the new dataframe so the existing columns can be easily referenced.

# In[36]:


df.info()


# #### **`km_per_driving_day`**
# 
# 1. Create a feature representing the mean number of kilometers driven on each driving day in the last month for each user. Add this feature as a column to `df`.
# 
# 2. Get descriptive statistics for this new feature
# 
# 

# In[38]:


# 1. Create `km_per_driving_day` feature
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']

# 2. Get descriptive stats
df['km_per_driving_day'].describe()


# #### **`percent_sessions_in_last_month`**
# 
# 1. Create a new column `percent_sessions_in_last_month` that represents the percentage of each user's total sessions that were logged in their last month of use.
# 
# 2. Get descriptive statistics for this new feature

# In[39]:


# 1. Convert infinite values to zero
df.loc[df['km_per_driving_day']==np.inf, 'km_per_driving_day'] = 0

# 2. Confirm that it worked
df['km_per_driving_day'].describe()


# In[40]:


# 1. Create `percent_sessions_in_last_month` feature
df['percent_sessions_in_last_month'] = df['sessions'] / df['total_sessions']

# 2. Get descriptive stats
df['percent_sessions_in_last_month'].describe()


# #### **`total_sessions_per_day`**
# 
# Now, create a new column that represents the mean number of sessions per day _since onboarding_.

# In[41]:


# Create `professional_driver` feature
df['professional_driver'] = np.where((df['drives'] >= 60) & (df['driving_days'] >= 15), 1, 0)


# In[42]:


# Create `total_sessions_per_day` feature
df['total_sessions_per_day'] = df['total_sessions'] / df['n_days_after_onboarding']


# As with other features, get descriptive statistics for this new feature.

# In[43]:


# Get descriptive stats
df['total_sessions_per_day'].describe()


# #### **`km_per_hour`**
# 
# Create a column representing the mean kilometers per hour driven in the last month.

# In[44]:


# Create `km_per_hour` feature
df['km_per_hour'] = df['driven_km_drives'] / (df['duration_minutes_drives'] / 60)
df['km_per_hour'].describe()


# #### **`km_per_drive`**
# 
# Create a column representing the mean number of kilometers per drive made in the last month for each user. Then, print descriptive statistics for the feature.

# In[45]:


# Create `km_per_drive` feature
df['km_per_drive'] = df['driven_km_drives'] / df['drives']
df['km_per_drive'].describe()


# This feature has infinite values too. Convert the infinite values to zero, then confirm that it worked.

# In[46]:


# 1. Convert infinite values to zero
df.loc[df['km_per_drive']==np.inf, 'km_per_drive'] = 0

# 2. Confirm that it worked
df['km_per_drive'].describe()


# In[47]:


# Create `percent_of_sessions_to_favorite` feature
df['percent_of_drives_to_favorite'] = (
    df['total_navigations_fav1'] + df['total_navigations_fav2']) / df['total_sessions']

# Get descriptive stats
df['percent_of_drives_to_favorite'].describe()


# ### **Task 3. Drop missing values**
# 
# Because you know from previous EDA that there is no evidence of a non-random cause of the 700 missing values in the `label` column, and because these observations comprise less than 5% of the data, use the `dropna()` method to drop the rows that are missing this data.

# In[48]:


# Drop rows with missing values
df = df.dropna(subset=['label'])


# ### **Task 4. Outliers**
# 
# You know from previous EDA that many of these columns have outliers. However, tree-based models are resilient to outliers, so there is no need to make any imputations.

# ### **Task 5. Variable encoding**

# #### **Dummying features**
# 
# In order to use `device` as an X variable, you will need to convert it to binary, since this variable is categorical.
# 
# In cases where the data contains many categorical variables, you can use pandas built-in [`pd.get_dummies()`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html), or you can use scikit-learn's [`OneHotEncoder()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) function.
# 
# **Note:** Each possible category of each feature will result in a feature for your model, which could lead to an inadequate ratio of features to observations and/or difficulty understanding your model's predictions.
# 
# Because this dataset only has one remaining categorical feature (`device`), it's not necessary to use one of these special functions. You can just implement the transformation directly.
# 
# Create a new, binary column called `device2` that encodes user devices as follows:
# 
# * `Android` -> `0`
# * `iPhone` -> `1`

# In[49]:


# Create new `device2` variable
df['device2'] = np.where(df['device']=='Android', 0, 1)
df[['device', 'device2']].tail()


# In[50]:


# Create binary `label2` column
df['label2'] = np.where(df['label']=='churned', 1, 0)
df[['label', 'label2']].tail()


# ### **Task 6. Feature selection**
# 
# Tree-based models can handle multicollinearity, so the only feature that can be cut is `ID`, since it doesn't contain any information relevant to churn.
# 
# Note, however, that `device` won't be used simply because it's a copy of `device2`.
# 
# Drop `ID` from the `df` dataframe.

# In[51]:


# Drop `ID` column
df = df.drop(['ID'], axis=1)


# ### **Task 7. Evaluation metric**
# 
# Before modeling, you must decide on an evaluation metric. This will depend on the class balance of the target variable and the use case of the model.
# 
# First, examine the class balance of your target variable.

# In[52]:


# Get class balance of 'label' col
df['label'].value_counts(normalize=True)


# ### **Task 8. Modeling workflow and model selection process**
# 
# The final modeling dataset contains 14,299 samples. This is towards the lower end of what might be considered sufficient to conduct a robust model selection process, but still doable.
# 
# 1. Split the data into train/validation/test sets (60/20/20)
# 
# Note that, when deciding the split ratio and whether or not to use a validation set to select a champion model, consider both how many samples will be in each data partition, and how many examples of the minority class each would therefore contain. In this case, a 60/20/20 split would result in \~2,860 samples in the validation set and the same number in the test set, of which \~18%&mdash;or 515 samples&mdash;would represent users who churn.
# 2. Fit models and tune hyperparameters on the training set
# 3. Perform final model selection on the validation set
# 4. Assess the champion model's performance on the test set
# 
# ![](https://raw.githubusercontent.com/adacert/tiktok/main/optimal_model_flow_numbered.svg)

# In[53]:


# 1. Isolate X variables
X = df.drop(columns=['label', 'label2', 'device'])

# 2. Isolate y variable
y = df['label2']

# 3. Split into train and test sets
X_tr, X_test, y_tr, y_test = train_test_split(X, y, stratify=y,
                                              test_size=0.2, random_state=42)

# 4. Split into train and validate sets
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, stratify=y_tr,
                                                  test_size=0.25, random_state=42)


# Verify the number of samples in the partitioned data.

# In[54]:


for x in [X_train, X_val, X_test]:
    print(len(x))


# This aligns with expectations.

# ### **Task 10. Modeling**

# **Note:** If your model fitting takes too long, try reducing the number of options to search over in the grid search.

# In[55]:


# 1. Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=42)

# 2. Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [None],
             'max_features': [1.0],
             'max_samples': [1.0],
             'min_samples_leaf': [2],
             'min_samples_split': [2],
             'n_estimators': [300],
             }

# 3. Define a list of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']

# 4. Instantiate the GridSearchCV object
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='recall')


# Now fit the model to the training data.

# In[56]:


get_ipython().run_cell_magic('time', '', 'rf_cv.fit(X_train, y_train)\n')


# Examine the best average score across all the validation folds.

# In[57]:


# Examine best score
rf_cv.best_score_


# Examine the best combination of hyperparameters.

# In[58]:


# Examine best hyperparameter combo
rf_cv.best_params_


# In[59]:


def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, or accuracy

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean 'metric' score across all validation folds.
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy',
                   }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          },
                         )

    return table


# Pass the `GridSearch` object to the `make_results()` function.

# In[60]:


results = make_results('RF cv', rf_cv, 'recall')
results


# #### **XGBoost**
# 
#  Try to improve your scores using an XGBoost model.
# 
# 1. Instantiate the XGBoost classifier `xgb` and set `objective='binary:logistic'`. Also set the random state.
# 
# 2. Create a dictionary `cv_params` of the following hyperparameters and their corresponding values to tune:
#  - `max_depth`
#  - `min_child_weight`
#  - `learning_rate`
#  - `n_estimators`
# 
# 3. Define a list `scoring` of scoring metrics for grid search to capture (precision, recall, F1 score, and accuracy).
# 
# 4. Instantiate the `GridSearchCV` object `xgb_cv`. Pass to it as arguments:
#  - estimator=`xgb`
#  - param_grid=`cv_params`
#  - scoring=`scoring`
#  - cv: define the number of cross-validation folds you want (`cv=_`)
#  - refit: indicate which evaluation metric you want to use to select the model (`refit='recall'`)

# In[61]:


# 1. Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=42)

# 2. Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [6, 12],
             'min_child_weight': [3, 5],
             'learning_rate': [0.01, 0.1],
             'n_estimators': [300]
             }

# 3. Define a list of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']

# 4. Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=4, refit='recall')


# Now fit the model to the `X_train` and `y_train` data.
# 
# Note this cell might take several minutes to run.

# In[62]:


get_ipython().run_cell_magic('time', '', 'xgb_cv.fit(X_train, y_train)\n')


# Get the best score from this model.

# In[63]:


# Examine best score
xgb_cv.best_score_


# And the best parameters.

# In[64]:


# Examine best parameters
xgb_cv.best_params_


# Use the `make_results()` function to output all of the scores of your model. Note that the function accepts three arguments.

# In[65]:


# Call 'make_results()' on the GridSearch object
xgb_cv_results = make_results('XGB cv', xgb_cv, 'recall')
results = pd.concat([results, xgb_cv_results], axis=0)
results


# This model fit the data even better than the random forest model. The recall score is nearly double the recall score from the logistic regression model from the previous course, and it's almost 50% better than the random forest model's recall score, while maintaining a similar accuracy and precision score.

# ### **Task 11. Model selection**
# 
# Now, use the best random forest model and the best XGBoost model to predict on the validation data. Whichever performs better will be selected as the champion model.

# #### **Random forest**

# In[66]:


# Use random forest model to predict on validation data
rf_val_preds = rf_cv.best_estimator_.predict(X_val)


# Use the `get_test_scores()` function to generate a table of scores from the predictions on the validation data.

# In[67]:


def get_test_scores(model_name:str, preds, y_test_data):
    '''
    Generate a table of test scores.

    In:
        model_name (string): Your choice: how the model will be named in the output table
        preds: numpy array of test predictions
        y_test_data: numpy array of y_test data

    Out:
        table: a pandas df of precision, recall, f1, and accuracy scores for your model
    '''
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy]
                          })

    return table


# In[68]:


# Get validation scores for RF model
rf_val_scores = get_test_scores('RF val', rf_val_preds, y_val)

# Append to the results table
results = pd.concat([results, rf_val_scores], axis=0)
results


# Notice that the scores went down from the training scores across all metrics, but only by very little. This means that the model did not overfit the training data.

# #### **XGBoost**
# 
# Now, do the same thing to get the performance scores of the XGBoost model on the validation data.

# In[69]:


# Use XGBoost model to predict on validation data
xgb_val_preds = xgb_cv.best_estimator_.predict(X_val)

# Get validation scores for XGBoost model
xgb_val_scores = get_test_scores('XGB val', xgb_val_preds, y_val)

# Append to the results table
results = pd.concat([results, xgb_val_scores], axis=0)
results


# Just like with the random forest model, the XGBoost model's validation scores were lower, but only very slightly. It is still the clear champion.

# ### **Task 12. Use champion model to predict on test data**
# 
# Now, use the champion model to predict on the test dataset. This is to give a final indication of how you should expect the model to perform on new future data, should you decide to use the model.

# In[70]:


# Use XGBoost model to predict on test data
xgb_test_preds = xgb_cv.best_estimator_.predict(X_test)

# Get test scores for XGBoost model
xgb_test_scores = get_test_scores('XGB test', xgb_test_preds, y_test)

# Append to the results table
results = pd.concat([results, xgb_test_scores], axis=0)
results


# ### **Task 13. Confusion matrix**
# 
# Plot a confusion matrix of the champion model's predictions on the test data.

# In[71]:


# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, xgb_test_preds, labels=xgb_cv.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=['retained', 'churned'])
disp.plot();


# ### **Task 14. Feature importance**
# 
# Use the `plot_importance` function to inspect the most important features of your final model.

# In[72]:


plot_importance(xgb_cv.best_estimator_);##


# ### **Task 15. Conclusion**
# 
# Now that you've built and tested your machine learning models, the next step is to share your findings with the Waze leadership team. Consider the following questions as you prepare to write your executive summary. Think about key points you may want to share with the team, and what information is most relevant to the user churn project.
# 
# **Questions:**
# 
# 1. Would you recommend using this model for churn prediction? Why or why not?
# 
# > _It depends. What would the model be used for? If it's used to drive consequential business decisions, then no. The model is not a strong enough predictor, as made clear by its poor recall score. However, if the model is only being used to guide further exploratory efforts, then it can have value._
# 
# 2. What tradeoff was made by splitting the data into training, validation, and test sets as opposed to just training and test sets?
# 
# > _Splitting the data three ways means that there is less data available to train the model than splitting just two ways. However, performing model selection on a separate validation set enables testing of the champion model by itself on the test set, which gives a better estimate of future performance than splitting the data two ways and selecting a champion model by performance on the test data._
# 
# 3. What is the benefit of using a logistic regression model over an ensemble of tree-based models (like random forest or XGBoost) for classification tasks?
# 
# > _Logistic regression models are easier to interpret. Because they assign coefficients to predictor variables, they reveal not only which features factored most heavily into their final predictions, but also the directionality of the weight. In other words, they tell you if each feature is positively or negatively correlated with the target in the model's final prediction._
# 
# 4. What is the benefit of using an ensemble of tree-based models like random forest or XGBoost over a logistic regression model for classification tasks?
# 
# > _Tree-based model ensembles are often better predictors. If the most important thing is the predictive power of the model, then tree-based modeling will usually win out against logistic regression (but not always!). They also require much less data cleaning and require fewer assumptions about the underlying distributions of their predictor variables, so they're easier to work with._
# 
# 5. What could you do to improve this model?
# 
# > _New features could be engineered to try to generate better predictive signal, as they often do if you have domain knowledge. In the case of this model, the engineered features made up over half of the top 10 most-predictive features used by the model. It could also be helpful to reconstruct the model with different combinations of predictor variables to reduce noise from unpredictive features._
# 
# 6. What additional features would you like to have to help improve the model?
# 
# > _It would be helpful to have drive-level information for each user (such as drive times, geographic locations, etc.). It would probably also be helpful to have more granular data to know how users interact with the app. For example, how often do they report or confirm road hazard alerts? Finally, it could be helpful to know the monthly count of unique starting and ending locations each driver inputs._
# 

# In[ ]:




