import os
import pickle
import traceback
import pandas as pd
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

app = Flask(__name__)

root_dir = ""
maxlen = 100

# inputs
training_data = root_dir + 'data/excel.csv'
include_columns = ['first_name', 'gender']
dependent_variable = 'gender'
model_directory = root_dir + 'model'

# These will be populated at training time
model = None
tokenizer = None


@app.route('/predict', methods=['POST'])  # Create http://host:port/predict POST end point
def predict():
    if model and tokenizer:
        try:
            request_json = request.json  # capture the json from POST
            names = []
            for dictionary_object in request_json:
                names.append(dictionary_object["first_name"])
            tokenized_names = tokenizer.texts_to_sequences(names)
            padded_names = pad_sequences(tokenized_names, padding='post', maxlen=maxlen)
            prediction = list(model.predict(padded_names))
            response = {"prediction": ["M" if x[0] >= .5 else "F" for x in prediction]}
            return jsonify(response)
        except Exception as ex:
            return jsonify({'error': str(ex), 'trace': traceback.format_exc()})
    else:
        print('Train first')
        return 'No model here'


@app.route('/train', methods=['GET'])  # Create http://host:port/train GET end point
def train():
    global model, tokenizer

    df = pd.read_csv(training_data, header=None)
    df.drop(2, axis=1, inplace=True)
    df.columns = ['first_name', 'gender']
    df['gender'] = (df['gender'] == "M").astype(int)
    names = df['first_name'].values
    y = df['gender'].values

    names_train, names_test, y_train, y_test = train_test_split(
        names, y, test_size=0.25)

    tokenizer = Tokenizer(num_words=500, char_level=True)
    tokenizer.fit_on_texts(names_train)
    x_train = tokenizer.texts_to_sequences(names_train)
    x_test = tokenizer.texts_to_sequences(names_test)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
    x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

    embedding_dim = 100

    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size,
                               output_dim=embedding_dim,
                               input_length=maxlen))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train,
              epochs=20,
              verbose=True,
              validation_data=(x_test, y_test),
              batch_size=10)
    loss, accuracy = model.evaluate(x_train, y_train, verbose=True)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=True)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    # saving
    with open(model_directory + '/tokenizer.pickle', 'wb') as fp:
        pickle.dump(tokenizer, fp, protocol=pickle.HIGHEST_PROTOCOL)

    model.save(model_directory)

    return 'Model Trained Successfully'


if __name__ == '__main__':
    try:
        model = tf.keras.models.load_model(model_directory)
        # loading
        with open(model_directory + '/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print('Model loaded')
    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        model = None
        tokenizer = None

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
