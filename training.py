import numpy as np
np.set_printoptions(threshold=np.nan)
from keras.layers import LSTM, Dense, Input, CuDNNLSTM, Masking, Bidirectional, Concatenate,Dropout, TimeDistributed
from keras.layers import Activation, dot, concatenate
from keras.models import Model,Sequential
import matplotlib.pyplot as plt
import h5py
from keras import backend as K
from keras import optimizers
from attention_decoder import AttentionDecoder
from keras.callbacks import ModelCheckpoint
import data_load


num_encoder_tokens = 3
num_decoder_tokens = 80



encoder_inputs = Input(shape=(2000, num_encoder_tokens))
encoder, forward_h, forward_c, backward_h, backward_c =   Bidirectional(CuDNNLSTM(500, return_sequences=True,return_state=True)) (encoder_inputs)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

decoder_inputs = Input(shape=(70, num_decoder_tokens))
decoder = CuDNNLSTM(1000, return_sequences=True)(decoder_inputs, initial_state=[state_h, state_c])

att= AttentionDecoder(70, 80)(decoder)

model = Model([encoder_inputs, decoder_inputs], att)

# Run training
adam=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
print ("Inputs: {}".format(model.input_shape))
print ("Outputs: {}".format(model.output_shape))
print ("Actual input: {}".format(data_load.X_train.shape))
print ("Actual output: {}".format(data_load.y_train.shape))

filepath='bis2s.h5'
checkpoint= ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
callbacks_list= [checkpoint]


history=model.fit([data_load.X_train, data_load.y_train2], data_load.y_train,
          batch_size=8,
          epochs=10,
          callbacks=callbacks_list,
          verbose=1,
          validation_split=0.2)


# Save model
model.save('model.h5')
model.save_weights('my_model_weights.h5')

print ('training done ')


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()