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

## Iteration 1: Cat vs. Non-Cat Classification

### Step 1: Obtain the Dataset
In this step, the goal is to acquire a dataset that includes images of both cats and non-cats. Popular datasets such as **Cat vs. Non-Cat** from Kaggle or **ImageNet** could be suitable. The dataset should include a variety of images to ensure the model can generalize well to different environments and perspectives of cats. If using a custom dataset, ensure it includes both labeled images of cats and other non-cat objects.

---

### Step 2: Extract the Data
Once the dataset has been obtained, the next step is to extract the images and their corresponding labels (e.g., "cat" and "non-cat"). This step may involve loading images from directories, converting them into arrays of pixel values, and pairing each image with its label.

**Key considerations:** 
- Ensure images are loaded in a format compatible with your model, typically as NumPy arrays.
- Consider automating this process using libraries like `os` for directory traversal or `tensorflow.keras.preprocessing.image` for loading and converting images.

---

### Step 3: Preprocess the Images
Preprocessing is crucial for the model to effectively learn patterns. Common preprocessing steps include:
1. **Resizing**: Neural networks often require all images to have the same dimensions. Resize the images to a uniform size (e.g., 64x64 or 128x128).
2. **Normalization**: To ensure pixel values are within a consistent range, you can normalize them by scaling pixel values (from 0-255) down to the range [0,1].

---

### Step 4: Build the Model
In this step, you will design a simple Convolutional Neural Network (CNN) using Keras/TensorFlow to classify the images. The model will consist of several key layers:

1. **Convolutional Layer**: The first layer will apply 32 filters, each of size 3x3, to the input images. These filters help detect important features like edges and textures. The activation function used in this layer is **ReLU (Rectified Linear Unit)**, which introduces non-linearity and helps the model learn more complex patterns.

2. **MaxPooling Layer**: After the convolutional layer, you will add a MaxPooling layer to down-sample the feature maps. This layer reduces the spatial dimensions of the image representation (2x2 pooling window), which helps decrease the computational load and prevents overfitting by focusing on the most important features.

3. **Flattening Layer**: This layer converts the 2D feature maps from the previous layer into a 1D vector. The flattened output will serve as the input to the fully connected layers.

4. **Fully Connected (Dense) Layer**: This layer consists of 128 neurons with a **ReLU** activation function. It processes the flattened input to learn higher-level patterns.

5. **Output Layer**: The final layer is a single neuron with a **sigmoid** activation function. Since this is a binary classification problem (cat vs. non-cat), the sigmoid function will output a probability between 0 and 1, where values closer to 0 indicate "non-cat" and values closer to 1 indicate "cat."

---

### Step 5: Visualize the Model Architecture
Before training, it's important to visualize and verify the architecture of your model. You can use functions in the framework to print out or diagram the structure of the network.

---

### Step 6: Train the Model
Now, train the model using the preprocessed dataset. Here are key considerations:
- **Loss Function**: For binary classification (cat vs. non-cat), use the appropriate loss function.
- **Optimizer**: You can use optimizers like `Adam` for efficient training.
- **Metrics**: Monitor metrics like accuracy to track how well the model is performing during training.

Monitor the training and validation accuracy/loss to ensure that the model is learning. If the model overfits (i.e., performs well on training but poorly on validation), consider reducing the model complexity or applying regularization techniques.

---

### Step 7: Evaluate the Model
Once training is complete, evaluate the model’s performance on a separate test set that was not used during training or validation. Important metrics include:
- **Accuracy**: The proportion of correctly classified images.
- **Confusion Matrix**: This gives a more detailed view of true positives, false positives, true negatives, and false negatives.
- **Precision and Recall**: These metrics are particularly useful when the dataset is imbalanced.

## Iteration 2: Fashion MNIST Classification

### Step 1: Obtain the Dataset
In this step, you will download the **Fashion MNIST** dataset, which is a collection of 28x28 grayscale images representing various types of clothing items. The dataset consists of 10 different classes, including items like t-shirts, trousers, shoes, etc. Fashion MNIST is widely used as a benchmark for image classification tasks, offering a more challenging alternative to the traditional MNIST digits dataset.

---

### Step 2: Extract the Data
Once the dataset is downloaded, you will extract and explore its contents. This includes:
- Loading the images and labels.
- Understanding the distribution of the different clothing categories.
- Inspecting the data types and dimensions to ensure compatibility with your model.

**Key considerations:**
- Ensure that the dataset is divided into training, validation, and test sets.
- Check the labels (0-9) to know which clothing item each number represents.

---

### Step 3: Preprocess the Images
Preprocessing is critical to ensure that the images are in a format suitable for training a neural network. Key preprocessing steps include:
1. **Normalizing the Images**: Since Fashion MNIST images have pixel values ranging from 0 to 255, normalizing them to the range [0,1] helps the neural network learn more efficiently.
2. **Reshaping**: The images need to be reshaped to match the input shape expected by the CNN. Typically, this means reshaping the 28x28 grayscale images into the format (28, 28, 1), where the last dimension indicates a single channel (grayscale).

---

### Step 4: Previsualize the Dataset
Before diving into model training, it’s helpful to visualize a few random images from the dataset. By doing so, you can:
- Verify that the images are loaded correctly.
- Get a feel for the types of clothing items the model will need to classify.
- Observe the variety within each class, which is important for the model’s generalization ability.

---

### Step 5: Build the Model
In this step, you will construct a Convolutional Neural Network (CNN) tailored to the Fashion MNIST dataset. The model will consist of the following components:

1. **Input Layer**: The input shape will be (28, 28, 1), which corresponds to the dimensions of the Fashion MNIST images (28x28 pixels with 1 channel for grayscale).

2. **First Convolutional Block**:
   - **Conv2D Layer**: This layer will apply 32 filters of size 3x3 to the input images. These filters will detect local patterns like edges and textures.
   - **MaxPooling Layer**: A pooling layer with a 2x2 window will be used to reduce the spatial dimensions of the feature maps, helping the model focus on the most important features while reducing the computational load.

3. **Second Convolutional Block**:
   - **Conv2D Layer**: This layer will apply 64 filters of size 3x3 to detect more complex patterns.
   - **MaxPooling Layer**: Another pooling layer with a 2x2 window to further downsample the feature maps.

4. **Third Convolutional Block**:
   - **Conv2D Layer**: Another layer with 64 filters of size 3x3 to capture even more complex features from the images. This layer will not be followed by pooling to retain more spatial information for the fully connected layers.

5. **Flatten Layer**: After the convolutional blocks, the output feature maps are flattened into a 1D vector, which will serve as the input to the dense layers.

6. **Dense Layers**:
   - **Dense Layer (64 neurons)**: This fully connected layer with 64 neurons uses the **ReLU activation function** to introduce non-linearity and learn high-level patterns from the extracted features.
   - **Dense Output Layer (10 neurons)**: The final layer consists of 10 neurons (one for each clothing category) with the **softmax activation function**. The softmax function outputs a probability distribution over the 10 classes, where the class with the highest probability will be the model's prediction.


---

### Step 6: Visualize the Model Architecture
After building the model, it’s important to visualize the architecture to ensure all layers are connected correctly. This can be done by printing a summary of the model, which will display the number of parameters, the output shape of each layer, and the total number of layers.

---

### Step 7: Train the Model
With the model ready, you will train it using the preprocessed Fashion MNIST dataset. During training, key hyperparameters to consider include:
- **Learning Rate**: Controls how quickly the model adjusts the weights during optimization.
- **Batch Size**: Determines the number of samples processed before the model’s internal parameters are updated.
- **Epochs**: Defines how many times the learning algorithm will work through the entire training dataset.

Monitor the training process by observing the accuracy and loss on both the training and validation sets. If you observe overfitting (i.e., high accuracy on the training set but low accuracy on the validation set), consider using regularization techniques or adjusting the complexity of the model.

---

### Step 8: Evaluate the Model
Once training is complete, it’s time to evaluate the model's performance on the test set. A few key steps include:
- **Accuracy**: Measure the overall accuracy of the model on unseen test data.
- **Confusion Matrix**: Generate a confusion matrix to get a more detailed understanding of how well the model distinguishes between the different classes. This will show the number of true positives, false positives, true negatives, and false negatives for each class.
- **Other Metrics**: Consider additional metrics like precision, recall, and F1-score, especially if the dataset is imbalanced.

# Bonus: Animal Classification

**Note: For this iteration, create a new notebook.**

In this bonus exercise, you will apply the skills learned in previous iterations to classify images from the **Animals-10** dataset, which contains images of 10 different animal species. Start by downloading the dataset from [Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10) and exploring it to understand its structure, checking for class imbalances, and visualizing a few samples. You will need to preprocess the data by resizing the images to a consistent size, normalizing the pixel values, and splitting the dataset into training, validation, and test sets. Using a convolutional neural network (CNN), design and train a model capable of classifying the images into their correct animal categories. After training, evaluate the model’s performance, and experiment with different architectures and techniques to improve the accuracy. This is an open-ended exercise, so feel free to apply your creativity and push your understanding further by experimenting with hyperparameters, regularization techniques, or data augmentation to achieve better results.


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
```
