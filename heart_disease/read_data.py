import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filename='heart.csv'):
    dt = pd.read_csv(filename)
    dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
    dt.loc[dt['sex'] == 0, 'sex'] = 'female'
    dt.loc[dt['sex'] == 1, 'sex'] = 'male'

    dt.loc[dt['chest_pain_type'] == 1, 'chest_pain_type'] = 'typical angina'
    dt.loc[dt['chest_pain_type'] == 2, 'chest_pain_type'] = 'atypical angina'
    dt.loc[dt['chest_pain_type'] == 3, 'chest_pain_type'] = 'non-anginal pain'
    dt.loc[ dt['chest_pain_type'] == 4, 'chest_pain_type'] = 'asymptomatic'

    dt.loc[dt['fasting_blood_sugar'] == 0, 'fasting_blood_sugar'] = 'lower than 120mg/ml'
    dt.loc[dt['fasting_blood_sugar'] == 1, 'fasting_blood_sugar'] = 'greater than 120mg/ml'

    dt.loc[dt['rest_ecg'] == 0, 'rest_ecg'] = 'normal'
    dt.loc[dt['rest_ecg'] == 1, 'rest_ecg'] = 'ST-T wave abnormality'
    dt.loc[dt['rest_ecg'] == 2, 'rest_ecg'] = 'left ventricular hypertrophy'

    dt.loc[dt['exercise_induced_angina'] == 0, 'exercise_induced_angina'] = 'no'
    dt.loc[dt['exercise_induced_angina'] == 1, 'exercise_induced_angina'] = 'yes'

    dt.loc[dt['st_slope'] == 1, 'st_slope'] = 'upsloping'
    dt.loc[dt['st_slope'] == 2, 'st_slope'] = 'flat'
    dt.loc[dt['st_slope'] == 3, 'st_slope'] = 'downsloping'

    dt.loc[dt['thalassemia'] == 1, 'thalassemia'] = 'normal'
    dt.loc[dt['thalassemia'] == 2, 'thalassemia'] = 'fixed defect'
    dt.loc[dt['thalassemia'] == 3, 'thalassemia'] = 'reversable defect'
    
    dt['sex'] = dt['sex'].astype('object')
    dt['chest_pain_type'] = dt['chest_pain_type'].astype('object')
    dt['fasting_blood_sugar'] = dt['fasting_blood_sugar'].astype('object')
    dt['rest_ecg'] = dt['rest_ecg'].astype('object')
    dt['exercise_induced_angina'] = dt['exercise_induced_angina'].astype('object')
    dt['st_slope'] = dt['st_slope'].astype('object')
    dt['thalassemia'] = dt['thalassemia'].astype('object')
    
    dt = pd.get_dummies(dt, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(dt.drop('target', 1), dt['target'], test_size = .2, random_state=10)
    
    standardize_columns = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression']
    for column in standardize_columns:
        mean = np.mean(X_train[column])
        sd   = np.std(X_train[column])
        X_train[column] = (X_train[column] - mean) / sd
        X_test[column]  = (X_test[column]  - mean) / sd
        
    return X_train, X_test, y_train, y_test