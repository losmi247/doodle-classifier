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

from src.neural_network.classifier.network import NeuralNetwork

# load the trained models
nn_model = NeuralNetwork.load_network('network.txt')

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
        return render_template('/home.html')

    # Handle POST request
    @app.route('/', methods=['POST'])
    def home():
        # Recieve base64 data from the user form
        canvasdata = request.form['canvasimg']
        encoded_data = request.form['canvasimg'].split(',')[1]

        # Decode base64 image to python array
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert 3 channel image (RGB) to 1 channel image (GRAY)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to (28, 28)
        gray_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_LINEAR)

        # turn it into a numpy array, and normalise pixel values
        image = np.array(gray_image / 255.0)
        print(image)
        
        from PIL import Image
        myimg = Image.fromarray(np.uint8(image * 255.0), 'L')
        myimg.show()

        try:
            # take the output from the NN model
            output = nn_model.feed_forward(image.flatten())
            # get the maximum as the prediction
            prediction = np.argmax(output)

            # calculate probabilities for each digit, sorted in decreasing order
            output = list(zip(np.around((output / np.sum(output)) * 100.0, decimals=2),np.arange(10)))
            output.sort(key = lambda x: -x[0])
            percentages = "\n".join([str(digit)+': {:.2f}'.format(p)+"%" for p,digit in output])

            print(f"Prediction Result : {str(prediction)}")
            return render_template('/home.html', response=str(prediction), percentages=percentages, canvasdata=canvasdata)
        except Exception as e:
            return render_template('/home.html', response=str(e), percentages=percentages, canvasdata=canvasdata)

        #return render_template('/home.html') - find a way to force reclassification after canvas is updated

    return app