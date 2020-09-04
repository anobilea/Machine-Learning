import numpy as np
import pandas as pd
from tensorflow import keras


classes = ['M', 'G', 'D', 'P', 'S']

data = pd.read_csv("Diagnosticus_test.CSV", sep=';', delimiter=';', index_col=False)
X = data.iloc[:,1:-1]

model = keras.models.load_model("Diagnosticus.h5")
out = model.predict(X)
for o in out:
    print(f"The prediction is {classes[np.argmax(o)]}")
