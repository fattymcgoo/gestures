import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import stats
layers = tf.keras.layers

# Pull in the raw training data and pivot so that there is one line per instance
mydata = pd.read_csv(r"MY FILEPATH FOR TRAINING DATA", names=["Instance", "Timeslice", "Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Classification"])
mydata.head()
mydata_features = mydata.copy()
mydata_features = pd.pivot_table(mydata_features, index=["Classification","Instance"], aggfunc='mean')
mydata_features = mydata_features.reset_index()

# Get the classification field from the training data
mydata_labels = mydata_features.pop('Classification')

# Add a normalization layer and compile/fit a sequential model
mydata_features = np.array(mydata_features)
normalize = layers.Normalization()
normalize.adapt(mydata_features)
mydata_model = tf.keras.models.Sequential([normalize, layers.Dense(64), layers.Dense(4)])
mydata_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())
mydata_model.fit(mydata_features, mydata_labels, epochs=30)

# Pull in the test data and again pivot so that there's one line per instance
mytest = pd.read_csv(r"MY FILEPATH FOR TEST DATA", names=["Instance", "Timeslice", "Ax", "Ay", "Az", "Gx", "Gy", "Gz"])
mytest.head()
mytest_features = mytest.copy()
mytest_features = pd.pivot_table(mytest_features, index="Instance", aggfunc='mean')
mytest_features = mytest_features.reset_index()
mytest_features = np.array(mytest_features)
mytest_features = stats.zscore(mytest_features)

# Get predictions
y_pred = mydata_model.predict(mytest_features)
clas=np.argmax(y_pred, axis=1)
print(clas)

