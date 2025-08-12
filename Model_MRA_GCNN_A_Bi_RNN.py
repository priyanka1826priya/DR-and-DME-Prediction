from sklearn.model_selection import train_test_split
from Model_A_Bi_RNN import Model_A_Bi_RNN
from keras.src.optimizers import Adam
from keras.src.layers import Input, Dropout, Dense, Layer, GlobalAveragePooling1D
from keras.src.models import Model
from stellargraph.layer import GraphConvolution
import numpy as np
import tensorflow as tf


#  Multi-Head Attention Layer
class MultiHeadRegionAttention(Layer):
    def __init__(self, num_heads, region_size, **kwargs):
        super(MultiHeadRegionAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.region_size = region_size

    def build(self, input_shape):
        self.Wq = self.add_weight(shape=(input_shape[-1], self.num_heads * self.region_size),
                                  initializer="glorot_uniform",
                                  trainable=True, name="Wq")
        self.Wk = self.add_weight(shape=(input_shape[-1], self.num_heads * self.region_size),
                                  initializer="glorot_uniform",
                                  trainable=True, name="Wk")
        self.Wv = self.add_weight(shape=(input_shape[-1], self.num_heads * self.region_size),
                                  initializer="glorot_uniform",
                                  trainable=True, name="Wv")
        super(MultiHeadRegionAttention, self).build(input_shape)

    def call(self, inputs):
        queries = tf.matmul(inputs, self.Wq)
        keys = tf.matmul(inputs, self.Wk)
        values = tf.matmul(inputs, self.Wv)

        # Split heads and regions
        queries = tf.reshape(queries, (-1, tf.shape(queries)[1], self.num_heads, self.region_size))
        keys = tf.reshape(keys, (-1, tf.shape(keys)[1], self.num_heads, self.region_size))
        values = tf.reshape(values, (-1, tf.shape(values)[1], self.num_heads, self.region_size))

        # Scaled dot-product attention
        attention_scores = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(float(self.region_size))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        # Weighted sum of values
        weighted_values = tf.matmul(attention_weights, values)
        weighted_values = tf.reshape(weighted_values, (-1, tf.shape(values)[1], self.num_heads * self.region_size))

        return weighted_values


def model_MHRA_GCN(n_nodes, n_features, n_classes, num_heads=4, region_size=8):
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

    # Apply Multi-Head Region Attention
    x = MultiHeadRegionAttention(num_heads=num_heads, region_size=region_size)(x)

    # Classification layers
    x = Dense(64, activation='relu')(x)
    x = Dense(n_classes, activation='softmax')(x)
    x = GlobalAveragePooling1D()(x)

    model = Model(inputs=[x_features, x_adjacency], outputs=x)
    model.summary()
    return model

def create_adjacency_and_indices(data, n_nodes):
    n_samples = data.shape[0]
    adjacency_matrices = np.ones((n_samples, n_nodes, n_nodes)) - np.eye(n_nodes)
    indices = np.tile(np.arange(n_nodes), (n_samples, 1))
    return adjacency_matrices, indices


def Model_MHRA_GCN(X, Y, test_X, test_Y, BS=None, sol=None):
    if BS is None:
        BS = 4
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

    model = model_MHRA_GCN(n_nodes, n_features, n_classes, num_heads=4, region_size=8)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit([Train_X, train_adjacency], Y, epochs=5, batch_size=BS, validation_data=([Test_X, test_adjacency], test_Y))

    pred = model.predict([Test_X, test_adjacency])
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = pred.astype('int')

    # Eval = ClassificationEvaluation(test_Y, pred)
    layerNo = -3
    intermediate_model = Model(inputs=model.input, outputs=model.layers[layerNo].output)
    Feats = intermediate_model.predict(
        [np.concatenate((Train_X, Test_X)), np.concatenate((train_adjacency, test_adjacency))])
    Feats = np.asarray(Feats)
    Feature = np.resize(Feats, (Feats.shape[0], Train_X.shape[1], Train_X.shape[2]))
    return Feature


def Model_MRA_GCNN_A_Bi_RNN(Train_x, Train_y, Test_X, Test_y, BS=None, sol=None):
    if BS is None:
        BS = 4
    if sol is None:
        sol = [5, 0.01, 1]
    Feature = Model_MHRA_GCN(Train_x, Train_y, Test_X, Test_y, BS=BS)
    Targets = np.concatenate((Train_y, Test_y))
    train_data, test_data, train_target, test_target = train_test_split(Feature, Targets, random_state=104, test_size=0.25, shuffle=True)
    Eval, pred = Model_A_Bi_RNN(train_data, train_target, test_data, test_target, BS=BS, sol=sol)
    return Eval, pred
