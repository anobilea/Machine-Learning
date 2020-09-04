import pandas as pd
from tensorflow import keras
from sklearn.model_selection import StratifiedShuffleSplit

classes = ['M', 'G', 'D', 'P', 'S']

def load_data(filename):

    data = pd.read_csv(filename, sep=';', delimiter=';', index_col=False)
    data['X95'].replace({'M':0 , 'G':1, 'D':2, 'P':3, 'S':4}, inplace=True)

    data.infer_objects()

    X_data = data.iloc[:,1:-1]
    Y_data = data.iloc[:,-1:]

    return X_data, Y_data

def train_and_save():

    X_data, Y_data = load_data("Diagnosticus.CSV")

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(X_data, Y_data):
        X_train_set = X_data.loc[train_index]
        Y_train_set = Y_data.loc[train_index]

        X_test_set = X_data.loc[test_index]
        Y_test_set = Y_data.loc[test_index]


    model = keras.models.Sequential([
        keras.layers.Dense(30),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dense(5, activation='softmax')])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    history = model.fit(X_train_set, Y_train_set, epochs=15, validation_data=(X_test_set, Y_test_set))
    print(history.params)

    model.save("Diagnosticus.h5")

    model.evaluate(X_test_set, Y_test_set)

    return model

model = train_and_save()
