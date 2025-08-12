from keras.src.optimizers import Adam
from Classificaltion_Evaluation import ClassificationEvaluation
from stellargraph.layer import GraphConvolution
from keras.src.layers import Input, Dropout, Dense
from keras.src.models import Model
import keras
import numpy as np


def model_GCN(n_nodes, n_features, n_classes):
    kernel_initializer = "glorot_uniform"
    bias_initializer = "zeros"

    x_features = Input(shape=(n_nodes, n_features))
    x_adjacency = Input(shape=(n_nodes, n_nodes))

    x = Dropout(0.5)(x_features)
    x = GraphConvolution(32, activation='relu', use_bias=True, kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)([x, x_adjacency])
    x = Dropout(0.5)(x)
    x = GraphConvolution(32, activation='relu', use_bias=True, kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)([x, x_adjacency])

    # Apply Dense layer directly without GatherIndices
    x = Dense(64, activation='relu')(x)
    x = Dense(n_classes, activation='sigmoid')(x)
    x = keras.layers.GlobalAveragePooling1D()(x)  # Ensure the output shape is (None, classes)

    model = Model(inputs=[x_features, x_adjacency], outputs=x)
    model.summary()

    return model


def create_adjacency_and_indices(data, n_nodes):
    n_samples = data.shape[0]
    adjacency_matrices = np.ones((n_samples, n_nodes, n_nodes)) - np.eye(n_nodes)
    indices = np.tile(np.arange(n_nodes), (n_samples, 1))
    return adjacency_matrices, indices


def Model_GCN(X, Y, test_X, test_Y, BS=32, epochs=5):
    n_nodes = 10
    n_features = 100
    n_classes = Y.shape[-1]

    Train_X = np.zeros((X.shape[0], n_nodes, n_features))
    for i in range(X.shape[0]):
        temp = np.resize(X[i], (n_nodes, n_features))
        Train_X[i] = np.reshape(temp, (n_nodes, n_features))

    Test_X = np.zeros((test_X.shape[0], n_nodes, n_features))
    for i in range(test_X.shape[0]):
        temp = np.resize(test_X[i], (n_nodes, n_features))
        Test_X[i] = np.reshape(temp, (n_nodes, n_features))

    train_adjacency, _ = create_adjacency_and_indices(X, n_nodes)
    test_adjacency, _ = create_adjacency_and_indices(test_X, n_nodes)

    model = model_GCN(n_nodes, n_features, n_classes)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit([Train_X, train_adjacency], Y, epochs=50, batch_size=BS, steps_per_epoch=5, validation_data=([Test_X, test_adjacency], test_Y))

    pred = model.predict([Test_X, test_adjacency])
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = pred.astype('int')
    Eval = ClassificationEvaluation(pred, test_Y)
    return Eval, pred

