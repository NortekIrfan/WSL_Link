import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt 
import seaborn as sns
import lmfit
from livelossplot import PlotLossesKeras
import tensorflow as tf
from tensorflow import keras
import keras.layers


"""Based on https://github.com/mzakharo/tubby/blob/main/fc_model.ipynb"""

TARGET = 'fc'
# Reading the data
df = pd.read_csv('HydrolabLogs\orp.csv')

# Copying the data 
df_copy = df.copy()
df['ppmTestData'] = df['ppm CL']*2.25

# Copys the data, and drop the coloumn selected
labels = df_copy.pop('ppm CL')

vs = []
for column in df_copy:
    vals = df_copy[column].to_numpy()
    c = float(column)
    for i, v in enumerate(vals):
        vs.append((v, c, labels[i]))
dataset_orig = pd.DataFrame(vs, columns=('orp', 'ph', 'fc'))

# Model that uses the points defined, and tries to build a linear function that tries to fit the graph
models = {}
for ph in df_copy.columns:
    if (ph == 'ppm CL'):
        continue
    xdata = df[str(ph)].to_numpy()
    ydata = df['ppm CL'].to_numpy()
    print(xdata)
    print(ydata)
    lmodel = lmfit.models.ExponentialModel()
    params = lmodel.guess(ydata, xdata)
    fit = lmodel.fit(ydata, params, x=xdata)
    models[str(ph)] = fit
    


vs = []
for ph, fit in models.items():
    for orp in range(200, 1000):
        fc = fit.eval(x=orp)        
        vs.append((orp, float(ph), fc))
dataset = pd.DataFrame(vs, columns= ('orp', 'ph', 'fc'))
dataset.describe()

train_dataset = dataset.sample(frac=0.8)#, random_state=0)
test_dataset = dataset.drop(train_dataset.index) #dataset_orig # dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
train_labels = train_features.pop(TARGET)

test_features = test_dataset.copy()
test_labels = test_features.pop(TARGET)

print(train_dataset.describe().transpose()[['mean', 'std']])
#sns.pairplot(train_dataset, diag_kind='kde')

inorm = tf.keras.layers.Normalization(axis=-1, input_shape=[2, ])
inorm.adapt(np.array(train_features))
onorm = tf.keras.layers.Normalization(axis=-1, invert=True)
onorm.adapt(train_labels)

checkpoint_filepath = 'checkpoint.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

def build_and_compile_model(inorm, onorm):
  model = keras.Sequential([
      inorm,
      layers.Dense(8, activation='elu'),
      layers.Dense(4, activation='elu'),
      layers.Dense(1), 
      onorm,
  ])
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model
model = build_and_compile_model(inorm, onorm)
model.summary()

history = model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=1, epochs=100, callbacks=[PlotLossesKeras(), model_checkpoint_callback])
print('mse:', model.evaluate(test_features, test_labels, verbose=0))

fmodel = checkpoint_filepath
#fmodel = 'model_fc.h5'

model = keras.models.load_model(fmodel, compile=False)
model.layers[-1].invert = True #Bug in Keras https://github.com/keras-team/keras/issues/17556


y = model.predict(test_features, verbose=0)
y = pd.DataFrame(y)[0]
sns.scatterplot(x=test_labels, y=y.to_numpy(), alpha=0.7)

model.save(f'model_{TARGET}.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(f'model_{TARGET}.tflite', "wb") as f:
  f.write(tflite_model)
print('done')