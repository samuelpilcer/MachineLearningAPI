#!flask/bin/python
from flask import Flask, jsonify, send_from_directory
from flask import request
from flask import make_response
from flask import abort
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import pandas as pd

#model = Sequential()
#model.add(Dense(12, input_dim=8, activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X, Y, epochs=150, batch_size=10)
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("model.h5")


# load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")


# curl -i -H "Content-Type: application/json" -X POST -d '{"description":"Test", "inputs":784, "layers":[23,32,44]}' http://localhost:5000/create
# curl -i -H "Content-Type: application/json" -X POST -d '{"description":"Test", "training_file":"training/training_mnist_0783-784.csv", "training_columns":784, "output_column":785}' http://localhost:5000/train/3
# curl -i -H "Content-Type: application/json" -X POST -d '{"saving_file":"files/test_mnist"}' http://localhost:5000/save/3



app = Flask(__name__, static_url_path='/static')

model1 = Sequential()
model1.add(Dense(12, input_dim=8, activation='relu'))
model1.add(Dense(8, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model2 = Sequential()
model2.add(Dense(12, input_dim=8, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

models = [
    {
        'id': 1,
        'model': model1,
        'description': u'Test model number 1. 3 dense layers.', 
        'trained': False
    },
    {
        'id': 2,
        'model': model2,
        'description': u'Test model number 2. 2 dense layers.', 
        'trained': False
    }
]

@app.route('/', methods=['GET'])
def get_models():
    models_to_display=[]
    for i in models:
        models_to_display.append(
        {'id': i["id"],
        'model': i["model"].to_json(),
        'description': i["description"], 
        'trained': i["trained"]
        })
    return jsonify({'models': models_to_display})


@app.route('/<int:model_id>', methods=['GET'])
def get_model(model_id):
    model = [model for model in models if model['id'] == model_id]
    if len(model) == 0:
        abort(404)
    return jsonify({'model': {'id': model[0]["id"],
                    'model': model[0]["model"].to_json(),
                    'description': model[0]["description"], 
                    'trained': model[0]["trained"]
                    }})


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/create', methods=['POST'])
def create_model():
    if not request.json or not 'layers' in request.json or not 'inputs' in request.json:
        abort(400)
    model_keras = Sequential()
    layers=request.json['layers']
    inputs=request.json['inputs']
    model_keras.add(Dense(layers[0], input_dim=inputs, activation='relu'))
    for i in layers:
        model_keras.add(Dense(i, activation='relu'))
    model_keras.add(Dense(1, activation='sigmoid'))
    model_keras.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model = {
        'id': models[-1]['id'] + 1,
        'model': model_keras,
        'description': request.json['description'],
        'trained': False
    }
    models.append(model)
    return jsonify({'id':models[-1]['id'],'layers': request.json['layers'],'description': request.json['description']})

@app.route('/train/<int:model_id>', methods=['POST'])
def train_model(model_id):
    if not request.json or not 'training_file' in request.json or not 'training_columns' in request.json or not 'output_column' in request.json:
        abort(400)
    training_data=pd.read_csv(request.json['training_file'])
    model=[model for model in models if model['id'] == model_id]
    if len(model) == 0:
        abort(404)
    model=model[0]
    columns=[]
    for i in range(request.json['training_columns']):
        columns.append(str(i))
    model['model'].fit(np.array(training_data.loc[:,columns]), np.array(training_data.loc[:,str(request.json['output_column']-1)]), epochs=1, batch_size=10)
    model['trained']=True
    return jsonify({'id':models[-1]['id']})

@app.route('/upload-training/<int:model_id>', methods=['POST'])
def upload_training(model_id):
    print(request.files)
    # checking if the file is present or not.
    if 'file' not in request.files:
        return "No file found"
    file = request.files['file']
    file.save("static/training/training_"+str(model_id)+".csv")
    return "file successfully saved"

@app.route('/get-structure/<int:model_id>')
def download_model_structure(model_id):
    training_data=pd.read_csv(request.json['training_file'])
    model=[model for model in models if model['id'] == model_id]
    if len(model) == 0:
        abort(404)
    model=model[0]
    json_model=model.to_json()
    return json_model

@app.route('/get-weights/<int:model_id>')
def download_model_weights(path):
    return send_from_directory('static', path)

@app.route('/save/<int:model_id>', methods=['POST'])
def save_model(model_id):
    if not request.json or not 'saving_file' in request.json:
        abort(400)
    saving_file=request.json['saving_file']
    model=[model for model in models if model['id'] == model_id]
    if len(model) == 0:
        abort(404)
    model=model[0]["model"]
    model_json = model.to_json()
    with open(saving_file+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(saving_file+".h5")
    return jsonify({'id':model_id})

@app.route('/<path:path>')
def send_html(path):
    return send_from_directory('static', path)

@app.route('/upload', methods=['POST'])
def upload_file():
    print(request.files)
    # checking if the file is present or not.
    if 'file' not in request.files:
        return "No file found"
    file = request.files['file']
    file.save("static/test.jpg")
    return "file successfully saved"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)