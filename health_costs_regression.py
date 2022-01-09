# Import libraries. You may or may not use all of these.
!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Import data
!wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')
dataset.tail()

features = ['sex','smoker','region','age','bmi','children']
categorical_columns = ['sex','smoker','region']
# numeric_columns = ['age','bmi','children']

ds = dataset

#Instantiate SimpleImputer 
si=SimpleImputer(missing_values = np.nan, strategy="median")
si.fit(ds[['age', 'bmi']])
  
#Filling missing data with median
ds[['age', 'bmi']] = si.transform(ds[['age', 'bmi']])

for category in categorical_columns:
  ds[category] = pd.factorize(ds[category])[0]

ds['more_than_1_child']=ds.children.apply(lambda x:1 if x>1 else 0)

kmeans = KMeans(n_clusters=2)
kmeans.fit(ds[features])
ds['type'] = kmeans.predict(ds[features])

dataset = ds

#splits the dataset into 80:20 training and testing data
dftrain, dftest = train_test_split(dataset, test_size=0.2)
#pops the expenses columns off into their own variable set
train_label = dftrain.pop('expenses')
test_label = dftest.pop('expenses')


''' THIS IS THE TENSORFLOW METHOD (WORSE)
normalizer = layers.experimental.preprocessing.Normalization()
normalizer.adapt(np.array(dftrain))

model = keras.Sequential([
    normalizer,
    layers.Dense(16),
    layers.Dense(4),
    layers.Dropout(.2),
    layers.Dense(1),
])

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mae',
    metrics=['mae', 'mse']
)
model.build()
model.summary()

history = model.fit(
    dftrain,
    train_label,
    epochs=100,
    validation_split=0.5,
    verbose=0, # disable logging
)

print(history)
'''

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer

model = xg.XGBRFRegressor(n_estimators = 51, max_depth=7, reg_lambda=0.27)  #n_estimators = 51, max_depth=7, reg_lambda=0.27

regr_trans = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(output_distribution='normal'))
regr_trans.fit(dftrain, train_label)
yhat = regr_trans.predict(dftest)

# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
#loss, mae, mse = model.evaluate(dftest, test_label, verbose=2)

mae = round(mean_absolute_error(test_label, yhat),3)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
