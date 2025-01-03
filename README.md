# Designing and Implementing a User-Friendly Plant Disease Classification System in the AWS Ecosystem: Insights and Practical Approaches

![UI upload image](/img/upload_ui.png)

![UI Classification results](/img/classification_results.png)

## Introduction
In this project, an end-to-end system for classifying various plant diseases using Amazon Web Services (AWS) has been developed to allow users to upload images of plants through a web interface, which are then processed and analyzed using a machine learning model deployed on AWS SageMaker. The classification results, along with confidence scores, are saved in Amazon DynamoDB.

## AWS Architecture
The plant disease classification application utilizes various AWS components to create a comprehensive and scalable architecture. Key components include:

- **Web Interface (EC2 Instance):** Hosts the Flask-based web application.
- **Image Storage (S3):** Stores uploaded plant images securely in an S3 bucket.
- **Image Preprocessing (EC2 - Flask Application):** Adjusts image sizes before classification.
- **Model Inference (SageMaker Endpoint):** Analyzes images using a deployed machine learning model.
- **Result Storage (DynamoDB):** Stores classification results with unique identifiers for retrieval.

![Architecture Diagram](/img/aws_architecture.png)

## Dataset Used
The dataset chosen was the Plant Village disease dataset.
![Plant Village dataset examples](/img/plant_village_examples.png)


## Model

![Xception Model Diagram](/img/xception_architecture.png)



## Performance Results
The model achieved a validation accuracy of 94% after three epochs of training.

![Model Performance Metric](/img/performance_metrics.png)

![Model Performance Graph](img/accurancy_loss_plots.png)

![Confusion Matrix](/img/confusion_matrix.png)

## Scalability Considerations
The system's scalability is well supported by AWS services:

- **Data Storage Scalability (S3):** Handles growing amounts of image data.
- **Compute Scalability (EC2 Auto Scaling):** Manages varying compute needs.
- **Model Inference Scalability (SageMaker):** Supports high volumes of inference requests.
- **Database Scalability (DynamoDB):** Scales horizontally to manage increased traffic and data.

## Conclusion
The project demonstrates the potential of AWS in deploying and scaling machine learning applications efficiently, providing a robust solution for plant disease classification that can be easily extended to other domains.






## References
- AWS Documentation on various services like S3, SageMaker, and DynamoDB.
- Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions.

