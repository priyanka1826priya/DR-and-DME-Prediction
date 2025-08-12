import numpy as np
from keras import layers, models
from keras.src.optimizers import Adam
from Classificaltion_Evaluation import ClassificationEvaluation


def adaptive_residual_block(x, units):
    residual = layers.Dense(units, activation=None, kernel_initializer="glorot_uniform")(x)
    adaptive_output = layers.Dense(units, activation=None, kernel_initializer="glorot_uniform")(residual)
    return layers.ReLU()(layers.Add()([x, adaptive_output]))


def Model_A_Bi_RNN(trainX, trainY, testX, testY, BS=None, sol=None):
    if sol is None:
        sol = [5, 0.01, 1]
    if BS is None:
        BS = 32

    Activation = ['linear', 'relu', 'tanh', 'softmax', 'sigmoid', 'leaky relu']

    time_steps = 5
    IMG_SIZE = 32
    input_shape = (time_steps, IMG_SIZE, IMG_SIZE, 3)
    num_classes = testY.shape[-1]

    # Preparing Training and Testing Data
    Train_X = np.zeros((trainX.shape[0], time_steps, IMG_SIZE, IMG_SIZE, 3))
    for i in range(trainX.shape[0]):
        temp = np.resize(trainX[i], (time_steps, IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (time_steps, IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((testX.shape[0], time_steps, IMG_SIZE, IMG_SIZE, 3))
    for i in range(testX.shape[0]):
        temp = np.resize(testX[i], (time_steps, IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (time_steps, IMG_SIZE, IMG_SIZE, 3))

    inputs = layers.Input(shape=input_shape)
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'))(inputs)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)

    # Bidirectional LSTM with Residual Connections
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, activation='relu'))(x)
    x = adaptive_residual_block(x, 256)  # Adaptive Residual Block
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, activation='relu'))(x)

    # Fully Connected Layers
    x = layers.Dense(int(sol[0]), activation=Activation[int(sol[2])])(x)  # 64  'relu'
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=sol[1]), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(Train_X, trainY, epochs=5, batch_size=BS, validation_data=(Test_X, testY), verbose=2)
    pred = model.predict(Test_X)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = pred.astype('int')
    Eval = ClassificationEvaluation(testY, pred)
    return Eval, pred
