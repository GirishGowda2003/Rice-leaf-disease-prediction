# Rice-leaf-disease-prediction
An AI-powered rice leaf disease prediction system using deep learning and MobileNetV2 to classify rice leaf images into disease categories.
# 🌾 Rice Leaf Disease Prediction — End-to-End Deep Learning Project

**A beginner-friendly, step-by-step image classification project**

In this notebook we will build a deep learning model that predicts rice leaf diseases based on leaf images. The model will learn to classify images into different disease categories such as **Bacterial Leaf Blight, Brown Spot, Leaf Smut**, or other classes available in the dataset.

We will go through the **complete deep learning workflow**:

1. Load & explore the raw image dataset
2. Basic checks and clean the dataset
3. Analyze the number of images in each disease class
4. Preprocess and resize the images
5. Perform image augmentation to improve model generalization
6. Handle class imbalance and remove/identify problematic images
7. Split the dataset into **training, validation, and testing** sets
8. Build a **CNN-based image classification model**
9. Train the CNN model and monitor training/validation accuracy and loss
10. Apply **Transfer Learning** using pretrained models such as **MobileNetV2, EfficientNetB0, or ResNet50**
11. Tune hyperparameters such as **learning rate, batch size, number of epochs, dropout, and optimizer**
12. Evaluate the model using **accuracy, precision, recall, F1-score, confusion matrix, and classification report**
13. After running all the models, compare their performance
14. Choose the model with the **best validation/test performance** and select it as the final model
15. Save the final trained model as a `.h5` or `.keras` file
16. Reload the saved model and make a prediction on a new rice leaf image to prove it works
17. Display the predicted disease name and prediction confidence
18. Build a UI using **Streamlit** for a better user experience and allow users to upload a rice leaf image and get the predicted disease **(optional)**

> 💡 Every step includes comments explaining **why** we do it, not just how — this notebook is written for someone new to deep learning and image classification.
