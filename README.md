# Dogs vs. Cats Classifier Using CNN

This project is a Convolutional Neural Network (CNN) model to classify images of dogs and cats. The dataset used consists of labeled images of cats and dogs, and the model is trained to predict whether an image contains a cat or a dog.

## Dataset
The dataset used is the [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) from Kaggle. It contains 25,000 images of cats and dogs. The dataset is split into training and validation sets for the model training and evaluation process.

- **Training Data:** 20,000 images
- **Validation Data:** 5,000 images

## Model Architecture
The model is built using TensorFlow and Keras libraries. It is a sequential model with three convolutional layers followed by pooling, flattening, and dense layers.

- **Input Layer:** 256x256x3 images (RGB)
- **Convolutional Layers:**
  - Conv2D with 32 filters, kernel size 3x3, ReLU activation, followed by MaxPooling2D
  - Conv2D with 64 filters, kernel size 3x3, ReLU activation, followed by MaxPooling2D
  - Conv2D with 128 filters, kernel size 3x3, ReLU activation, followed by MaxPooling2D
- **Fully Connected Layers:**
  - Dense layer with 128 units, ReLU activation
  - Dropout to prevent overfitting
  - Dense layer with 64 units, ReLU activation
  - Dropout
  - Output layer: 1 unit with sigmoid activation (binary classification)
  
Batch normalization is applied to stabilize and speed up training. Dropout is included to reduce overfitting.

## Training
The model is compiled using the Adam optimizer, and binary cross-entropy is used as the loss function because it is a binary classification problem.

- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy

## Results
- **Accuracy:** The model reaches a training accuracy of over 95%, but validation accuracy tends to show signs of overfitting, which was improved using batch normalization and dropout layers.
- **Loss Curves:** The loss curves show the model's training and validation loss over epochs, indicating areas of improvement and overfitting.

## Reducing Overfitting
To reduce overfitting, the following methods were used:
- **Batch Normalization:** Normalizes the inputs to each layer to reduce covariate shift.
- **Dropout:** Introduces regularization to randomly drop units and reduce the risk of overfitting.
- **Data Augmentation:** Can be further applied to artificially increase the size of the dataset and improve model generalization.

## Testing
The model was tested on new images of cats and dogs to check its performance on unseen data. The images were resized to 256x256 pixels and reshaped for prediction.

- **Test on Dog Image:**
  ```python
  model.predict(test_input)  # Result: Predicted as Dog (1)
  ```
- **Test on Cat Image:**
  ```python
  model.predict(test_input)  # Result: Predicted as Cat (0)
  ```

## Future Improvements
- Further regularization techniques such as L1/L2 regularizers.
- Data augmentation for better generalization.
- Try deeper architectures like ResNet or transfer learning using pre-trained models like VGG16 or MobileNet.

## Conclusion
This CNN model effectively classifies dogs and cats with high accuracy. Overfitting has been reduced with regularization techniques, and the model is generalizable to new images.    

