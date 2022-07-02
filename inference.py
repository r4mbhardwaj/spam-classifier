import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import pickle


model = keras.models.load_model('spam_or_not_model')


# load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

max_len = 50


while True:
    # input the test data
    inp = input('Enter the comment: ')

    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating='post', maxlen=max_len))

    tag = lbl_encoder.inverse_transform([np.argmax(result)]) # get the tag from the result
    print(tag)