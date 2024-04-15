
import pandas as pd
#You will need the scikit-learn to do the spliting:  pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Loads the train annotations
train_csv_path = 'heart_disease_data.csv'
dataset_df = pd.read_csv(train_csv_path)

categorical = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
dataset_df  = pd.get_dummies(dataset_df, columns=categorical)

sc = MinMaxScaler()
dataset_df[dataset_df.columns] = sc.fit_transform(dataset_df[dataset_df.columns])

# Splits the training data into train and test
train_set, test_set, _, _ = train_test_split(dataset_df, dataset_df['target'], stratify=dataset_df['target'], test_size=0.25)

# dump the new sets into csvs
train_set.to_csv('train.csv')
test_set.to_csv('test.csv')

# gloryous