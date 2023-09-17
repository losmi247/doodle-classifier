# parts from https://medium.com/analytics-vidhya/how-to-deploy-digit-recognition-model-into-drawing-app-6e59f82a199c
# to run the app, use
#
#   flask --app src.flask run --debug
#
import os
import numpy as np
import cv2
import base64
from flask import (
    Flask, render_template, request
)
import matplotlib.pyplot as plt

from src.neural_network.classifier.network import NeuralNetwork
from src.naive_bayes.classifier.naive_bayes import NaiveBayesClassifier


#
# Loading Trained Models
#

# load the trained Neural Network model (gets the deployable model)
nn_model = NeuralNetwork.load_network('network.txt')
# load the trained Naive Bayes model (gets the deployable model)
nb_model = NaiveBayesClassifier.load_nb('bayes.txt')


#
# Flask App
#

# application factory function
def create_app(test_config=None):
    # create and configure the app
    app = Flask("src.flask", instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Handle GET request
    @app.route('/', methods=['GET'])
    def drawing():
        return render_template('/home.html', lastcanvasbuttonclicked='draw', lastpickerbuttonclicked='neural_network')

    # Handle POST request
    @app.route('/', methods=['POST'])
    def home():
        # Receive base64 data from the user form
        canvasdata = request.form['canvasimg']
        encoded_data = request.form['canvasimg'].split(',')[1]

        # Decode base64 image to python array
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        gray_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Resize to (28, 28) - the downsampled image
        gray_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
        
        # turn it into a numpy array, and normalise pixel values
        preprocessed_image = np.array(gray_image / 255.0)

        try:
            # remember which canvas button was last clicked (excluding clear)
            lastcanvasbuttonclicked = request.form['last_canvas_button_clicked']
            
            # remember which picker button was last clicked (i.e. which classifier is being used)
            lastpickerbuttonclicked = request.form['last_picker_button_clicked']
            
            # classify the drawn image using the selected classifier
            output = None
            prediction = None
            if lastpickerbuttonclicked == 'naive_bayes':
                # take the output from the NB model - NB uses 28x28 images that are not normalised (still 0-255) 
                output = nb_model.get_probabilities(gray_image)
                # get the maximum as the prediction
                prediction = nb_model.categories[np.argmax(output)]
                
                # calculate probabilities for each category, sorted in decreasing order.
                # first subtract max of these negative values to bring the logs closer to 0 to avoid underflow.
                output = output - np.max(output)
                denominator = np.sum(np.exp(output))
                output = list(zip(np.around(np.exp(output)/denominator * 100.0, decimals=2),nb_model.categories))
            if lastpickerbuttonclicked == 'neural_network':
                # take the output from the NN model
                output = nn_model.feed_forward(preprocessed_image.flatten())
                # get the maximum as the prediction
                prediction = nn_model.categories[np.argmax(output)]
                
                # calculate probabilities for each category, sorted in decreasing order
                output = list(zip(np.around((output / np.sum(output)) * 100.0, decimals=2),nn_model.categories))
            
            # sort the list of (probability,category) in decreasing order by probability
            output.sort(key = lambda x: -x[0])
            #create a string of probabilities for each digit to display
            percentages = [str(digit)+': {:.2f}'.format(p)+"%" for p,digit in output]
            
            # upscale the 28x28 image to display it
            output_size = 336
            scale = output_size // 28
            output_image = np.zeros((output_size, output_size), dtype=gray_image.dtype)
            # duplicate pixels to upscale
            for i in range(gray_image.shape[0]):
                for j in range(gray_image.shape[1]):
                    output_image[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = gray_image[i, j]     
            # convert the downsampled image to a base64 data URL
            _, buffer = cv2.imencode('.png', output_image)
            downsampled_data_url = 'data:image/png;base64,' + base64.b64encode(buffer).decode()

            # rerender the template
            print(f"Prediction Result : {str(prediction)}")
            return render_template('/home.html', response=str(prediction), percentages=percentages, canvasdata=canvasdata, \
                processed_image=downsampled_data_url, lastcanvasbuttonclicked=lastcanvasbuttonclicked, lastpickerbuttonclicked=lastpickerbuttonclicked)
        except Exception as e:
            return render_template('/home.html', response=str(e), percentages=None, canvasdata=canvasdata, \
                processed_image=None, lastcanvasbuttonclicked='draw', lastpickerbuttonclicked='neural_network')

        #return render_template('/home.html') - find a way to force reclassification after canvas is updated

    return app