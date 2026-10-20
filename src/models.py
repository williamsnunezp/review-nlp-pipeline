import tensorflow as tf

# def build_textcnn(vocab_size, max_len=100):
#     """
#     TextCNN para clasificación de sentimientos.
#     Arquitectura equivalente a la utilizada en el notebook de referencia.
#     """
#     inputs = tf.keras.layers.Input(shape=(max_len,))
#     embedded = tf.keras.layers.Embedding(vocab_size, 64)(inputs)
#     x = tf.keras.layers.Conv1D(64, 3, activation='relu')(embedded)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.MaxPool1D(3)(x)
#     x = tf.keras.layers.Conv1D(64, 5, activation='relu')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.MaxPool1D(5)(x)
#     x = tf.keras.layers.Conv1D(64, 5, activation='relu')(x)
#     x = tf.keras.layers.GlobalMaxPool1D()(x)
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(100, activation='relu')(x)
#     outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
#     return tf.keras.Model(inputs=inputs, outputs=outputs)

def build_lstm(vocab_size, max_len=100):
    """
    Red neuronal recurrente con tres capas LSTM apiladas.
    """
    inputs = tf.keras.layers.Input(shape=(max_len,))
    embedded = tf.keras.layers.Embedding(vocab_size,100,mask_zero=True)(inputs)
    x = tf.keras.layers.LSTM(64,return_sequences=True)(embedded)
    x = tf.keras.layers.LSTM(64,return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
