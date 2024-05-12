import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def nn_model():
    mnist_nn_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(784,input_shape = (784,)),
        tf.keras.layers.Dense(397, activation = 'relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])

    mnist_nn_model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy']   
    )

    return mnist_nn_model

mnist = pd.read_csv('MNIST.csv', header = None, skiprows = 1)
X = mnist.iloc[:,1:]
y = mnist.iloc[:,0]

from sklearn.model_selection import KFold

folds_amt = KFold(n_splits = 5, shuffle = True)

figure, axes = plt.subplots(1,5, figsize=(30,5))

accuracy_list = []

for i, (training_indexes,testing_indexes) in enumerate(folds_amt.split(X)):

    X_train = X.iloc[ training_indexes[0] : training_indexes[training_indexes.size-1], : ]
    y_train = y.iloc[ training_indexes[0] : training_indexes[training_indexes.size-1]]

    X_test = X.iloc[ testing_indexes[0] : testing_indexes[testing_indexes.size-1], : ]
    y_test = y.iloc[ testing_indexes[0] : testing_indexes[testing_indexes.size-1]]

    
    mnist_nn_model = nn_model()
    nn_model_history = mnist_nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    loss, acc = mnist_nn_model.evaluate(X_test, y_test, verbose = 0)
    accuracy_list.append(acc)

    axes[i].plot(nn_model_history.history['loss'], label='Training Loss')
    axes[i].plot(nn_model_history.history['val_loss'], label = 'Validation Loss')
    axes[i].plot(f"Fold #{i+1}")
    axes[i].set_xlabel('epochs')
    axes[i].set_ylabel('loss')

figure.legend(loc='upper left', ncol = 2)
plt.tight_layout()
plt.show()

print("Accuracy over five folds is: ", accuracy_list)
print("Average Accuracy: ", np.mean(accuracy_list))