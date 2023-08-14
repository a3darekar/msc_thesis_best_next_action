import numpy as np
import warnings

from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, RepeatVector, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')

def build(max_seq_length, num_activity_types):
    # Build LSTM model
    activity_seqs = Input(shape=(max_seq_length-1, num_activity_types + 1), name="activity_types")
    # status = Input(shape=(2,), name="status")

    lstm = LSTM(100, return_sequences=True, dropout=0.2, name="lstm")(activity_seqs)
    bn = BatchNormalization()(lstm)
    # merged = Concatenate(axis=-1)([lstm, status])
    lstm2 = LSTM(100, return_sequences=False, dropout=0.2, name="lstm2")(bn)
    output = Dense(num_activity_types, activation='softmax', name="dense")(lstm2)
    model = Model(inputs=activity_seqs, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train(model, X, y, epochs=25, batch_size=32, validation_split=0.2, callbacks = None):
    if callbacks:
        return model.fit(X, y, epochs, batch_size, validation_split, callbacks)
    else:
        return model.fit(X, y, epochs, batch_size, validation_split)

def predict(model, X):
    return np.argmax(model.predict(X), axis=-1)

def predict_proba(model, X):
    return model.predict(X)



def get_score(kpi_probability_table, sequence, unseen_score):
    return kpi_probability_table.get(tuple(sequence), unseen_score)

def predict_sequence(seq, model):
    days_data = np.array([np.random.randint(0, 7) for _i in seq])
    x2 = tf.keras.preprocessing.sequence.pad_sequences([days_data], maxlen=max_seq_length - 1)

    pad_seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_seq_length - 1)
    pad_seq = to_categorical(pad_seq, num_classes=num_activity_types)
    pad_seq = np.concatenate([pad_seq, x2.reshape(x2.shape[0], x2.shape[1], 1)], axis=2)

    return model.predict(pad_seq, verbose=0)
