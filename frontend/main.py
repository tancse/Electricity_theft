from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load the trained model
model_path = '../Model/model.h5'
if not os.path.exists(model_path):
    print(f"Model file '{model_path}' not found.")
    exit()

tlgru_model = load_model(model_path)

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Load the new data
    new_data = pd.read_csv(file)
    
    # Extract features
    x_new = new_data.iloc[:, 2:].values
    
    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    x_new_imputed = imputer.fit_transform(x_new)
    
    # Reshape data for model compatibility
    x_new_reshaped = np.reshape(x_new_imputed, (x_new_imputed.shape[0], x_new_imputed.shape[1], 1))
    
    # Pad or truncate sequences to match the required length
    max_sequence_length = 1034  # Adjust this value according to your model's input shape
    x_new_padded = pad_sequences(x_new_reshaped, maxlen=max_sequence_length, padding='post', truncating='post')
    
    # Make predictions
    predictions = tlgru_model.predict(x_new_padded)
    predictions_binary = (predictions > 0.5).astype(int)
    
    # Count the occurrences of each class
    class_counts = {0: 0, 1: 0}
    for label in predictions_binary:
        class_counts[label[0]] += 1
    
    # Plot the bar chart
    plt.bar(class_counts.keys(), class_counts.values(), color=['blue', 'red'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(list(class_counts.keys()), ['Normal', 'Theft'])
    plt.title('Predicted Labels Distribution')
    
    # Save plot to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the plot in base64
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # Close the plot
    plt.close()
    
    return render_template('result.html', plot_base64=plot_base64)

if __name__ == '__main__':
    app.run(debug=True)
