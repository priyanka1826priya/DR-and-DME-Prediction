import numpy as np
from keras import layers, models
from keras.src.optimizers import Adam
from Classificaltion_Evaluation import ClassificationEvaluation


def Model_RNN(trainX, trainY, testX, testY, BS=None):
    if BS is None:
        BS = 32

    input_shape = (5, 32, 32, 3)  # input_shape is (time_steps, height, width, channels)
    num_classes = testY.shape[-1]
    IMG_SIZE = 32
    Train_X = np.zeros((trainX.shape[0], 5, IMG_SIZE, IMG_SIZE, 3))
    for i in range(trainX.shape[0]):
        temp = np.resize(trainX[i], (5, IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (5, IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((testX.shape[0], 5, IMG_SIZE, IMG_SIZE, 3))
    for i in range(testX.shape[0]):
        temp = np.resize(testX[i], (5, IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (5, IMG_SIZE, IMG_SIZE, 3))
    model = models.Sequential()
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.LSTM(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    model.fit(Train_X, trainY, epochs=50, batch_size=BS, verbose=1, steps_per_epoch=10, validation_data=(Test_X, testY))
    testPredict = model.predict(Test_X)
    pred = np.round(testPredict)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = pred.astype('int')
    Eval = ClassificationEvaluation(testY, pred)
    return Eval, pred
