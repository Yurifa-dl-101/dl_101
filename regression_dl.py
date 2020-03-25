import warnings
warnings.filterwarnings("default", category=DeprecationWarning)
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense
from keras.optimizers import SGD

import keras
import pydot as pyd
keras.utils.vis_utils.pydot = pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

# Generate data from -20, -19.75, -19.5, .... , 20
train_x = np.arange(-20, 20, 0.25)

# Calculate Target : sqrt(2x^2 + 1)
train_y = np.sqrt((2*train_x**2)+1)

# Create Network
inputs = Input(shape=(1,))
h_layer = Dense(8, activation='relu')(inputs) # 1*8 + 8 = 16
h_layer = Dense(4, activation='relu')(h_layer) # 8*4 + 4 = 36
outputs = Dense(1, activation='linear')(h_layer) # 4*1 + 1 = 5
model = Model(inputs=inputs, outputs=outputs)

# Optimizer / Update Rule
sgd = SGD(lr=0.001)
# Compile the model with Mean Squared Error Loss
model.compile(optimizer=sgd, loss='mse')
model.summary()

def visualize_model(model):
    return SVG(model_to_dot(model).create(prog='dot', format='svg'))

visualize_model(model)

# Train the network and save the weights after training
# model.fit(train_x, train_y, batch_size=20, epochs=10000, verbose=1)
# model.save_weights('weights_regression.h5')
model.load_weights('weights_regression.h5')

# Predict training data
predict = model.predict(np.array([26]))
print('f(26) = ', predict)

predict_y = model.predict(train_x)

predict_y.shape

train_x.shape

train_y.shape

# # Draw target vs prediction
plt.plot(train_x, train_y, 'r')
plt.plot(train_x, predict_y, 'b')
plt.show()




