# Bird-Species-Classification

# **Introduction**

The project focuses on classifying **525 bird species** using a comprehensive dataset containing nearly 84,000 training images, 2,600 test images, and 2,600 validation images. Each bird species is represented by five images in the test and validation sets. The images are of high quality, featuring a single bird occupying at least 50% of the frame. The project aims to leverage deep learning techniques, specifically transfer learning, to achieve high classification accuracy. The **InceptionV3 pre-trained model** from Keras was utilized for this purpose.

# **Process**
The project followed a structured approach to build and train the model:

1.   Data Exploration and Visualization:
  *   Objective: Understand the dataset structure and inspect image quality.
  *   Steps:
      * Explored the directory using 'os.walk()' to identify bird species classes.
      * Created a sorted list of classes using 'pathlib.Path()' and 'numpy.array()'.
      * Visualized random images with 'random.sample()' and 'plt.imshow()' to assess data quality.

2.   Data Preprocessing:
  *  Objective: Prepare the dataset for training with normalization and augmentation.
  *  Steps:
      * Rescaled pixel values to [0, 1] for consistency.
      * Applied data augmentation with 'ImageDataGenerator()' for better generalization.
      * Loaded images in batches using 'flow_from_directory()' to optimize training.

3.   Model Creation and Compilation:
  *  Objective: Build and compile a custom deep learning model.
  *  Steps:
      * Used InceptionV3 as the base model with pre-trained weights.
      * Added custom layers like 'GlobalAveragePooling2D' and 'Dense'.
      * Compiled the model with 'categorical_crossentropy' loss and 'Adam' optimizer.
4.   Model Training:
  * Objective: Train the model and optimize its performance.
  * Steps:
      * Trained the model using 'fit()' while monitoring accuracy and loss.
      * Implemented early stopping to prevent overfitting.  

5.   Visualization of Training Results:
  * Objective: Analyze model performance over epochs.
  * Steps:
      * Plotted accuracy and loss curves using matplotlib.pyplot to compare
      training vs. validation performance.

6.   Fine-Tuning:

  * Objective: Improve accuracy by refining the model.
  * Steps:
      * Unfroze specific layers for fine-tuning with a lower learning rate.
      * Recompiled and trained the model on the adjusted layers.
      * Saved the fine-tuned model.



# **Models Used**
The primary model used for bird species classification was the **InceptionV3** pre-trained model. This deep learning architecture was selected for its proven efficiency in handling large-scale image classification tasks. The model's top layers were initially frozen to utilize the pre-trained features, and later, the deeper layers were fine-tuned to enhance the model's performance.

# **Final Results**
The initial training of the **InceptionV3 model**, with the top layers frozen, yielded an accuracy of **89.37%**. Upon fine-tuning the deeper layers of the model, the accuracy improved to **91.05%**. This significant enhancement in accuracy demonstrated the effectiveness of transfer learning and model fine-tuning in achieving high-performance classification for a diverse dataset of bird species.

# **Conclusion**
This project successfully demonstrated the application of deep learning and transfer learning techniques to classify **525 bird species** with high accuracy. By employing the InceptionV3 model and carefully fine-tuning it, the project achieved an accuracy of **91.05%**. The systematic approach, from dataset exploration to model fine-tuning, highlighted the importance of understanding the data, visualizing results, and refining models to achieve superior performance in image classification tasks.
