import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
from file_handler import *


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)

# making labels to model understandable format
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)



# making training_sentences to model understandable format
vocab_size = 1000
embedding_dim = 16
max_len = 50
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)



# model training
model = keras.Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_len)) # input_length is the max length of the sequence

model.add(layers.GlobalAveragePooling1D()) # GlobalAveragePooling1D is a layer that takes the average of all the values in a sequence
model.add(layers.Dense(16, activation='relu')) # Dense is a layer that takes the average of all the values in a sequence
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

model.summary()


# training the model
epochs = 500

history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)


# saving the model
model.save("spam_or_not_model")


# pickeling the tokenizer
import pickle

# saving the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# saving the label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
