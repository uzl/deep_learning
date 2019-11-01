import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras.backend as K
from pathlib import Path
from collections import namedtuple

# Generate dummy data
import numpy as np
x_train = np.random.random((25, 7))
y_train = keras.utils.to_categorical(np.random.randint(3, size=(25, 1)), num_classes=3)

# Create simple model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=7))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# Custom callback function to capture y and y pred after each batch
YLog = namedtuple('YLog', ['Y_pred', 'Y_true'])

class record_last_layer(keras.callbacks.Callback):
    def __init__(self, model, X_train, Y_train ):
        self.model = model
        self.x = X_train
        self.y = Y_train

    def on_train_begin(self, logs={}):
        self.y_y_pred_log = []
        return

    def on_batch_end(self, batch, logs={}):
        inp = model.input  
        outputs = model.layers[-1].output  
        functors = K.function([inp, K.learning_phase()], [outputs])
        layer_outs = functors([self.x, 1.])

        self.y_y_pred_log.append(YLog(layer_outs, self.y))
        return

ylogger = record_last_layer(model = model, X_train = x_train, Y_train=y_train)

# compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss= 'categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy']
             )
             
# Start training with our custom callback
model.fit(x_train, y_train,
          epochs=5,
          batch_size=128, 
          callbacks=[ylogger]
  )


# just print our target value
print(ylogger.y_y_pred_log)
