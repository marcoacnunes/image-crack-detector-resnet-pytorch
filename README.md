# Deep Learning Classification using ResNet18 on Watson Studio

A binary image classification project using pre-trained ResNet18. The focus of this project is to classify images into two categories using the ResNet18 model on Watson Studio.

## Introduction

This project leverages a pre-trained ResNet18 model to classify images into positive and negative categories using tensors.

## Setting up on Watson Studio

1. **Login to Watson Studio**: Navigate to the [Watson Studio dashboard](https://dataplatform.cloud.ibm.com/) and login.
2. **Create a New Project**: Click on `New project` and select `Standard`.
3. **Import a Jupyter Notebook:**
   - Go to the "Assets" tab.
   - Under the "Notebook" section, click on "Add notebook".
   - Choose the "From URL" tab.
   - Enter the URL: https://github.com/marcoacnunes/image-crack-detector-resnet-pytorch/blob/main/main.ipynb.
   - Configure the notebook settings, select an appropriate runtime (preferably with GPU if available).
   - Click "Create" to import and create the notebook in Watson Studio.

4. **Run the Notebook:**
   - Once the notebook is imported, you can open it in Watson Studio.
   - Run the notebook cells sequentially. The script will automatically download the necessary data and start the training process.

## Workflow

1. **Data Acquisition**: The tensors for positive and negative images are automatically downloaded using wget.
2. **Dataset Preparation**: Custom dataset classes are defined for training and validation datasets. Data is loaded from tensors and prepared for the training and evaluation.
3. **Model Initialization**: A pre-trained ResNet18 model is fetched and its final layer is modified for binary classification.
4. **Training**: The model is trained for a defined number of epochs. The loss for each iteration is stored.
5. **Evaluation**: After training, the model's accuracy on the validation dataset is calculated.
6. **Visualization**: 
    - Loss plots: The loss for each iteration is plotted to visualize the training process.
    - Misclassified Samples: Some of the misclassified samples are displayed with their predicted and actual labels.

## Notes

- Training deep learning models can be computationally intensive and time-consuming. Consider using GPU environments on Watson Studio for faster training.
- The number of epochs, batch size, and other hyperparameters can be adjusted as needed in the script.

#### **Screenshots**
Below are the screenshots illustrating various stages of the project:

- **Accuracy & Loss Plot**: This screenshot displays the plotted accuracy and loss function after training.  
![Accuracy & Loss Plot](https://github.com/marcoacnunes/image-crack-detector-resnet-pytorch/blob/main/img1.jpg)

- **Misclassified Samples**: This screenshot shows the first four misclassified samples from the validation dataset.  
![Misclassified Samples](https://github.com/marcoacnunes/image-crack-detector-resnet-pytorch/blob/main/img2.png)



