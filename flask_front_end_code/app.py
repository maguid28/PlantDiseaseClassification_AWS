from flask import Flask, render_template, request, redirect, url_for, jsonify
import boto3
import uuid
import json
import numpy as np
from PIL import Image
import os

def load_and_preprocess_image(img_path):
    #Open image using Pillow
    img = Image.open(img_path)
    #Resize image
    img = img.resize((299, 299))
    #Convert the PIL Image object to np array
    img_array = np.array(img)
    #Expand dimensions of the array to add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    #Normalise the pixel values to the range [0, 1]
    img_array = img_array / 255.0
    
    return img_array

def invoke_endpoint(endpoint_name, payload):
    runtime = session.client('sagemaker-runtime')
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload
    )
    result = response['Body'].read()
    return json.loads(result)

app = Flask(__name__)

#AWS credentials and region
session = boto3.Session(
    aws_access_key_id='xxxxxxxxxxxx',
    aws_secret_access_key='xxxxxxxxxxxxxx',
    region_name='us-east-1'
)

s3 = session.client('s3')
sagemaker_runtime = session.client('sagemaker-runtime')
dynamodb = session.resource('dynamodb')

#S3 bucket name
BUCKET_NAME = 'awsprojectbucket123321'
#Sagemaker endpoint name
ENDPOINT_NAME = 'tensorflow-inference-2024-04-30-10-36-17-777'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image_file = request.files['image']
    image_id = str(uuid.uuid4())
    image_key = f'uploads/{image_id}.jpg'

    #Create temp directory if it doesnt exist
    os.makedirs('temp', exist_ok=True)

    #Save uploaded image temporarily
    image_path = f'temp/{image_id}.jpg'
    image_file.save(image_path)

    #Preprocess image
    image_data = load_and_preprocess_image(image_path)

    #Convert image data to correct format using JSON
    payload = json.dumps({"instances": image_data.tolist()})

    #Upload preprocessed image data to S3
    s3.put_object(Body=payload, Bucket=BUCKET_NAME, Key=image_key)

    endpoint_name = 'tensorflow-inference-2024-04-30-10-36-17-777'
    response = invoke_endpoint(endpoint_name, payload)
    print('Inference results:', response)

    #Process inference result
    predictions = response['predictions'][0]  
    #Index of maximum probability
    predicted_index = np.argmax(predictions)

    #List of class names that correspond to each index
    class_names = ['Apple: Apple scab',
                   'Apple: Black rot',
                   'Apple: Cedar apple rust',
                   'Apple: healthy',
                   'Blueberry: healthy',
                   'Cherry: healthy',
                   'Cherry: Powdery mildew',
                   'Corn: Cercospora leaf spot Gray leaf spot',
                   'Corn: Common rust',
                   'Corn: healthy',
                   'Corn: Northern Leaf Blight',
                   'Grape: Black rot',
                   'Grape: Esca (Black Measles)',
                   'Grape: healthy',
                   'Grape: Leaf_blight (Isariopsis_Leaf_Spot)',
                   'Orange: Haunglongbing (Citrus_greening)',
                   'Peach: Bacterial spot',
                   'Peach: healthy',
                   'Bell Pepper: Bacterial_spot',
                   'Bell Pepper: healthy',
                   'Potato: Early blight',
                   'Potato: healthy',
                   'Potato: Late blight',
                   'Raspberry: healthy',
                   'Soybean: healthy',
                   'Squash: Powdery mildew',
                   'Strawberry: healthy',
                   'Strawberry: Leaf_scorch',
                   'Tomato: Bacterial spot',
                   'Tomato: Early blight',
                   'Tomato: healthy',
                   'Tomato: Late blight',
                   'Tomato: Leaf Mold',
                   'Tomato: Septoria leaf spot',
                   'Tomato: Spider mites Two-spotted spider mite',
                   'Tomato: Target Spot',
                   'Tomato: Tomato mosaic virus',
                   'Tomato: Tomato Yellow Leaf Curl Virus']

    predicted_class = class_names[predicted_index]
    confidence_score = predictions[predicted_index]

    print(f"Predicted Class: {predicted_class}, Probability: {confidence_score:.4f}")

    #Store classification result in DynamoDB
    table = dynamodb.Table('plantnet-results')
    table.put_item(
        Item={
            'image-id': image_id,
            'predicted_class': predicted_class,
            'confidence_score': str(confidence_score)
        }
    )

    #Redirect to result page with the image ID
    return redirect(url_for('result', image_id=image_id))

@app.route('/result/<image_id>')
def result(image_id):
    return render_template('result.html', image_id=image_id)

@app.route('/check_classification/<image_id>')
def check_classification(image_id):
    #Retrieve the classification result from DynamoDB
    table = dynamodb.Table('plantnet-results')
    response = table.get_item(Key={'image-id': image_id})

    if 'Item' in response:
        result = response['Item']
        predicted_class = result['predicted_class']
        confidence_score = result['confidence_score']
        return jsonify({'predicted_class': predicted_class, 'confidence_score': confidence_score})
    else:
        return jsonify({'predicted_class': None, 'confidence_score': None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)