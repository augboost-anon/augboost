
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from sklearn.utils import gen_batches, check_random_state

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

import numpy as np

def random_feature_subsets(array, batch_size, random_state):
    """ Generate K subsets of the features in X """
    random_state = check_random_state(random_state)
    features = list(range(array.shape[1]))
    random_state.shuffle(features)
    for batch in gen_batches(len(features), batch_size):
        yield features[batch]

def nn_augment_model(X_train, y_train, max_epochs):

#giving 'patience' a very high value to neutralize the early stopping
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=0)#,
#        ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    dim = X_train.shape[1]
    embed_dim = X_train.shape[1]

    input_1 = Input(shape = (dim, ))

    dropout_rate = 0   #place holder, currently dropout isn't used

    dense_1 = Dense(embed_dim, activation='relu')(input_1)
    dropout_1 = Dropout(dropout_rate)(dense_1)
    dense_2 = Dense(embed_dim, activation='relu')(dropout_1)
    dropout_2 = Dropout(dropout_rate)(dense_2)
    dense_3 = Dense(embed_dim, activation='relu')(dropout_2)
    dropout_3 = Dropout(dropout_rate)(dense_3)
    output_1 = Dense(1)(dropout_3)
    nn_loss = 'mse'

    net = Model(inputs = [input_1], outputs=[output_1])
    net.compile(optimizer = 'Adam', loss = nn_loss)
    net.fit(X_train, y_train, batch_size = min(300, round(float(len(X_train))/15)), epochs=max_epochs, callbacks=callbacks, validation_split=0.15)

    partial_model = Model(inputs = [input_1], outputs=[dense_3])

    return (partial_model, net.history.history)

def augment(X, augmentation_matrix):
    return np.dot(X, augmentation_matrix)

def get_transformed_params(X, y, n_features_per_subset, max_epochs, random_state=777, augmentation_method='nn'):
    if random_state == 777:
        random_state = np.random.randint(0,700)
    transforming_params = []
    for i, subset in enumerate(random_feature_subsets(X, n_features_per_subset, random_state=random_state)):
        if(augmentation_method=='nn')|(augmentation_method=='NN'):
            model, nn_history = nn_augment_model(X[:, subset], y, max_epochs = max_epochs)
            transforming_params.append((model, subset, nn_history))
        else:
            if(augmentation_method == 'pca')|(augmentation_method == 'PCA'):
                pca = PCA()
                model = pca.fit(X[:, subset])
                transforming_params.append((model, subset))
            else:
                if (augmentation_method == 'rp') | (augmentation_method == 'RP'):
                    rp = GaussianRandomProjection(len(subset))
                    model = rp.fit(X[:, subset])
                    transforming_params.append((model, subset))
                else:
                    raise ValueError("`augmentation_method` must be `pca`, 'rp' or `nn`, but was %s" % augmentation_method)
    return transforming_params

def get_transformed_matrix(X, transforming_params, augmentation_method):
    transformed_matrix = np.zeros(X.shape, dtype=np.float32)
    for tup in transforming_params:
        if(augmentation_method=='nn')|(augmentation_method=='NN'):
            model, subset, nn_history = tup
            transformed_matrix[np.ix_([True] * len(X), subset)] = model.predict(X[:, subset])
        else:
            if(augmentation_method == 'pca')|(augmentation_method == 'PCA')|(augmentation_method == 'rp')|(augmentation_method == 'RP'):
                model, subset = tup
                transformed_matrix[np.ix_([True] * len(X), subset)] = model.transform(X[:, subset])
            else:
                raise ValueError("`augmentation_method` must be `pca`, 'rp' or `nn`, but was %s" % augmentation_method)
    return transformed_matrix
