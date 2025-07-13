from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.models import load_model
import PIL
from PIL import Image
from keras.utils import load_img

app = Flask(__name__)

# Load the pre-trained model
model = load_model('pretrained_model.h5') 

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        file = request.files['image']
        
        # Save the image to a temporary location
        file_path = 'temp.jpg'  
        file.save(file_path)
        img = load_img(file_path, target_size=(224, 224))

        # Preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make predictions
        preds = model.predict(x)
        decoded_preds = decode_predictions(preds, top=3)[0]

        # Return the predictions
        return render_template('index.html', predictions=decoded_preds) 
    
    # If it's a GET request, just render the HTML template
    return render_template('index.html') 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
