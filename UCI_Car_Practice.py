import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


df = pd.read_csv('UCI_car.csv')
csv_heading = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']  # noqa: E501


# *********************** Data Preprocessing ***********************
buy_mapping = {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}
maint_mapping = {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}
doors_mapping = {'2': 2, '3': 3, '4': 4, '5more': 9}
persons_mapping = {'2': 2, '4': 4, 'more': 9}
lug_boot_mapping = {'small': 1, 'med': 2, 'big': 3}
safety_mapping = {'low': 1, 'med': 2, 'high': 3}
class_mapping = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}

df['buying_no'] = df['buying'].map(buy_mapping)
df['maint_no'] = df['maint'].map(maint_mapping)
df['doors_no'] = df['doors'].map(doors_mapping)
df['persons_no'] = df['persons'].map(persons_mapping)
df['lug_boot_no'] = df['lug_boot'].map(lug_boot_mapping)
df['safety_no'] = df['safety'].map(safety_mapping)
df['class_no'] = df['class'].map(class_mapping)

# The preprocessed columns extracted into a dataframe
df_new = df[['buying_no', 'maint_no', 'doors_no', 'persons_no', 'lug_boot_no', 'safety_no', 'class_no']]  # noqa: E501

# The rows that are classified as 'good' and 'vgood' are reclassified into 'acc' or 'unacc'  # noqa: E501
test_data = df_new[(df_new.class_no == 3) | (df_new.class_no == 4)]

X_test_data = np.array(test_data[['buying_no', 'maint_no', 'doors_no', 'persons_no', 'lug_boot_no', 'safety_no']])  # noqa: E501

# The rows that are NOT classified as 'good' or 'vgood' are used for training the classifier  # noqa: E501
df_new = df_new[(df_new.class_no != 3) & (df_new.class_no != 4)]

# Extracting the required columns for X and y
df_array = np.array(df_new[['buying_no', 'maint_no', 'doors_no', 'persons_no', 'lug_boot_no', 'safety_no']])  # noqa: E501
variety = np.array(df_new['class_no'])

# ******************************************************************
# Classifying into a Car being Acceptable or not acceptable
knn = KNeighborsClassifier(n_neighbors=4)

cv_pred = cross_val_predict(knn, df_array, variety, cv=5)

conf_mat = confusion_matrix(variety, cv_pred)

target_names = ['Unacceptable Car', 'Acceptable Car']
print(classification_report(variety, cv_pred, target_names=target_names))

# Training with X and y
knn.fit(df_array, variety)

y_pred = knn.predict(X_test_data)

output = X_test_data[:]

output = output.tolist()

for i in range(len(output)):
    output[i].append(y_pred[i])

# ************** Testing ************

count_unacc = 0
count_acc = 0

for i in range(len(output)):
    if(output[i][6] == 1):
        count_unacc = count_unacc + 1
        print(output[i])
    else:
        count_acc = count_acc + 1

# ************** Testing ************

# TODO
# Hyperparameter tuning using GridSearchCV
# Logging
# Try TSNE for clustering visualization
