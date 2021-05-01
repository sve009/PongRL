import tensorflow as tf
import numpy as np
import random

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(5, activation='relu', input_shape=(5,), batch_size=None),
        tf.keras.layers.Dense(3, activation='linear')
        ])
    return model

def train_step(model, s, a, sp, reward, alpha):
    with tf.GradientTape() as tape:
        x = model(s)
        x = x[0, a]
    y = model(sp)[0]
    grad = tape.gradient(x, model.trainable_variables)

    w = model.get_weights()

    for i in range(len(grad)):
        w[i] += alpha * (reward + max(y) - x) * grad[i]

    model.set_weights(w)

def return_action(model, s):
    if random.randint(0, 5) < 1:
        return random.randint(0,2)
    else:
        x = model(s)[0]
        m = 0
        for i in range(len(x)):
            if x[i] > x[m]:
                m = i
        return m
