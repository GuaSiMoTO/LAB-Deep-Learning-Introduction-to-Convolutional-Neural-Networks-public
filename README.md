# LAB-Deep-Learning-Introduction-to-Convolutional-Neural-Networks

![](https://user-images.githubusercontent.com/970858/63474771-d6734700-c469-11e9-83bb-9429da563909.png)

## Introduction

In this lab, you will work through three exercises to understand and apply Convolutional Neural Networks (CNNs). These exercises will guide you through training models for different image classification tasks. You will start by detecting whether an image contains a cat, then move on to classifying different types of clothing. As a bonus, you will train a model to classify images into 10 different animal categories.

By the end of this lab, you will have practical experience in building, training, and evaluating CNN models for image classification.

## Setup Instructions

To run the notebook and reproduce the results, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone git@github.com:EduKami-ai/LAB-Deep-Learning-Introduction-to-Convolutional-Neural-Networks.git
    cd LAB-Deep-Learning-Introduction-to-Convolutional-Neural-Networks
    ```

2. **Create a virtual environment and install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Start Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

## Workflow

### Iteration 1: Cat vs. Non-Cat Classification

#### Step 1: Obtain the Dataset
You will begin by obtaining a dataset that contains images of cats and non-cats. This dataset will be used to train a model that can classify whether a given image contains a cat.

#### Step 2: Extract the Data
Once you have the dataset, extract the images and labels. You may need to preprocess the images to ensure they are in a suitable format for training a neural network.

#### Step 3: Preprocess the Images
Preprocess the images by resizing them, normalizing pixel values, and possibly augmenting the data to improve model robustness.

#### Step 4: Build the Model
Design a CNN model using Keras/TensorFlow. Start with a simple architecture and gradually increase complexity as needed.

#### Step 5: Visualize the Model Architecture
Before training, visualize the model architecture to verify that it has been constructed correctly.

#### Step 6: Train the Model
Train the model using the preprocessed dataset. Monitor the training and validation accuracy and loss to ensure the model is learning effectively.

#### Step 7: Evaluate the Model
After training, evaluate the model's performance on a test set. Calculate accuracy, and other relevant metrics, and compare the results.

### Iteration 2: Fashion MNIST Classification

#### Step 1: Obtain the Dataset
Download the Fashion MNIST dataset, which contains images of various clothing items.

#### Step 2: Extract the Data
Extract the dataset and explore its contents to understand the structure and types of images you will be working with.

#### Step 3: Preprocess the Images
Prepare the images by normalizing them and reshaping them to fit the input shape required by your CNN model.

#### Step 4: Previsualize the Dataset
Visualize a few random images from the dataset to get an idea of the data distribution and class labels.

#### Step 5: Build the Model
Construct a CNN model tailored to the Fashion MNIST dataset, ensuring it has the appropriate input and output layers.

#### Step 6: Visualize the Model Architecture
Visualize the architecture of your model to ensure all layers are connected correctly.

#### Step 7: Train the Model
Train the model on the Fashion MNIST dataset, adjusting hyperparameters such as learning rate, batch size, and epochs.

#### Step 8: Evaluate the Model
Evaluate the model's performance on the test set. Generate a confusion matrix to understand how well the model distinguishes between different clothing items.

### Iteration 3: Animal Classification (Bonus Exercise)

In this bonus exercise, you will apply what you've learned to a new dataset containing images of 10 different animals.

#### Step 1: Obtain the Dataset
Download the Animals-10 dataset from Kaggle.

#### Step 2: Explore and Preprocess the Data
Explore the dataset to understand its structure and preprocess the images by resizing, normalizing, and splitting them into training, validation, and test sets.

#### Step 3: Build and Train the Model
Design and train a CNN model to classify the images into the correct animal categories.

#### Step 4: Evaluate the Model
After training, evaluate the model's performance and experiment with different architectures and techniques to improve accuracy.

## Additional Resources

If you need additional help or references while working on these exercises, consider checking out the following resources:
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf)
- [Fashion MNIST Dataset Documentation](https://github.com/zalandoresearch/fashion-mnist)
- [Animals-10 Dataset on Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

## Submission

Once you have completed all exercises, push your Jupyter notebook to your forked GitHub repository. Ensure that your final submission includes narratives explaining your visualizations and insights.

```bash
$ git add .
$ git commit -m "done"
$ git push origin main
