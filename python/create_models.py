import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class IdentityModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.identity(inputs)


class USEModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        return self.dense(self.embedding(tf.squeeze(inputs)))


class MultimodalModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        embedding = self.embedding(tf.squeeze(inputs['string_input']))
        features = tf.concat((inputs['float_input'], embedding), axis=1)
        return self.dense(features)


identity_model = IdentityModel()
identity_model.compile(loss='binary_crossentropy', optimizer='sgd')
identity_model.fit([0.0, 1.0], [0.0, 1.0])
identity_model.save('export/identity/0')

use_model = USEModel()
use_model.compile(loss='binary_crossentropy', optimizer='sgd')
use_model.fit(['a sentence', 'b sentence'], [0.0, 1.0])
use_model.save('export/use/0')

multimodal_inputs = {
    'string_input': tf.constant(np.array(['a sentence', 'b sentence'])),
    'float_input': tf.constant(np.array([0.0, 1.0]))
}
multimodal_model = MultimodalModel()
multimodal_model.compile(loss='binary_crossentropy', optimizer='sgd')
multimodal_model.fit(multimodal_inputs, np.array([0.0, 1.0]))
multimodal_model.save('export/multimodal/0')