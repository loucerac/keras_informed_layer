import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import to_categorical

from keras_informed_layer.layers import Informed


def test_edge_consistency_after_fit():
    input_dim = 3
    output_dim = 5
    # adjacency matrix specifiying the connections between
    # the input layer and the hidden layer (not a dense MLP)
    adj = np.array([[0, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1]])
    sparse_layer = Informed(adj=adj)
    inputs = Input(shape=(input_dim,))
    x = sparse_layer(inputs)
    x = Dense(output_dim, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(
        loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
    )

    # Fit model and check that if an input
    # and a node is not coonected, the weight is still zero
    # after training

    num_samples = 10
    X = np.random.rand(num_samples, input_dim)
    y = np.random.randint(low=0, high=output_dim, size=num_samples)
    y = to_categorical(y, num_classes=output_dim)
    model.fit(X, y, epochs=10, batch_size=10)

    out = [
        np.all(w.numpy()[adj == 0] == 0)
        for w in model.layers[1].weights
        if "kernel" in w.name
    ]
    assert all(out) == True
