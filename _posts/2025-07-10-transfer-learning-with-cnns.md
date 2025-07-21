---
layout: post
title: "Transfer Learning with CNNs"
date: 2025-07-10 10:00:00 -0500
author: "Ben Johnson"
math: true
toc: true
reading_time: "5 min @ 250 wpm"
excerpt: "Training CNNs from scratch requires massive data and compute. Transfer learning is a fine-tuning technique which uses pre-trained models - enabling customization with limited data and compute."
references:
  - title: "Gradient-Based Learning Applied to Document Recognition"
    authors: "LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P."
    year: 1998
    journal: "Proceedings of the IEEE"
    url: "http://yann.lecun.com/exdb/publis/pdf/lecun98.pdf"
  - title: "How trasnferrable are features in deep neural networks?"
    authors: "Yosinski, J., Clune, J., Bengio, Y., & Lipson, H."
    year: 2014
    journal: "Advances in Neural Information Processing Systems"
    url: "https://arxiv.org/abs/1411.1792"
  - title: "PyTorch in One Hour"
    authors: "Sebastian Raschka"
    year: 2023
    url: "https://sebastianraschka.com/blog/2023/pytorch-in-one-hour.html"
---

Convolutional Neural Networks (CNNs) have existed since 1998 and serve as the backbone of many computer vision tasks, from image classification to object detection [1].

However, training CNNs requires large amounts of labeled data and compute, which can be a barrier for many applications.

This article explores transfer learning as a solution which adapts pre-trained CNNs for new tasks. Transfer learning allows us to train models with **smaller datasets** and **reduced compute**.


## What is a CNN?

CNNs are a type of deep learning model specifically designed for processing grid-like data, such as images. They consist of multiple layers that automatically learn spatial hierarchies of features.

Each layer of the CNN takes input data, applies a *convolution*, and produces output data. The convolution is a function which extracts local features, such as edges, corners, and textures. This hierarchical feature learning allows CNNs to capture complex patterns in images.

<figure style="margin: 30px 0; text-align: center;">
  <img src="{{ '/assets/images/lenet5-architecture.png' | relative_url }}" 
       alt="LeNet-5 CNN Architecture" 
       style="max-width: 100%; padding: 20px; margin: 20px auto; display: block; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
  <figcaption style="margin-top: 15px; font-style: italic; color: #666;">Figure 1: Architecture of LeNet-5, a Convolutional Neural Network for digit recognition. Each plane represents a feature map with constrained weights.</figcaption>
</figure>

Different layers of the CNN learn to detect different levels of features:
- **Early layers** capture low-level features like edges and textures
- **Middle layers** learn more complex shapes and patterns
- **Final layers** combine these features to recognize high-level concepts like objects or faces

## Transfer Learning

Open-source CNN models, such as MobileNet and ResNet, have been trained on large datasets to learn rich feature representations that can be reused for various tasks. MobileNet, for example, is trained on the ImageNet 1000 dataset, which contains over 1 million images across 1000 classes.

Transfer learning adapts a pre-trained model to a new task by reusing its learned features. This is particularly useful when you have limited data for your specific task, as it allows you to leverage the knowledge captured by the pre-trained model.

During transfer learning, we will update the weights of the pre-trained model to better fit our specific task. This typically involves:
1. **Freezing early layers** that detect low-level features
2. **Fine-tuning later layers** that are more task-specific


## Implementation

The full implementation is documented in the Github repository [here](https://github.com/bdjohnson529/tree_classifier).

We use a use a two-phase approach to retrain the CNN:
1. **Feature Extraction**: Freeze the convolutional base and only train the classifier.
2. **Fine-tuning**: Unfreeze some or all layers and train with a lower learning rate.

We are going to use Tensorflow and Keras to implement transfer learning with MobileNetV2 in this guide. This model is lightweight and efficient, making it suitable for mobile and edge devices.

### Feature Extraction

First we import the existing MobileNetV2 model and freeze its convolutional layers. This allows us to use the pre-trained weights for feature extraction without modifying them.

We will unfreeze the last N layers of the base model, and add several new layers on top for classification. The new layers will be trained from scratch on our specific dataset, and will have classes specific to our task.

{: .code-container}
```python
class MobileNetTransferLearning:
    def build_model(self, model_name):
        base_model = self.create_base_model(model_name)

        # Unfreeze only the last N layers of the base model for feature extraction
        num_frozen = len(base_model.layers) - self.config.NUM_FROZEN_LAYERS
        for layer in base_model.layers[:num_frozen]:
            layer.trainable = False
        for layer in base_model.layers[num_frozen:]:
            layer.trainable = True

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(self.config.DROPOUT_RATE),
            layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])

        return model
```

After building the model, we need to train it with an optimizer and loss function. We will use the Adam optimizer with a learning rate of 0.001, and the categorical crossentropy loss function for multi-class classification.

{: .code-container}
```python
    def compile_model(self, model, model_name, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.config.INITIAL_LEARNING_RATE
        if model_name == "MobileNetV2":
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = optimizers.SGD(
                learning_rate=learning_rate,
                momentum=0.9
            )
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
```

### Fine-tuning

After feature extraction, we can fine-tune the model by unfreezing some or all layers and training with a lower learning rate. This allows the model to adapt its learned features to better fit our specific task.

{: .code-container}
```python
    def fine_tune_model(self, model, model_name, train_data, val_data):
        model = self.compile_model(
            model, 
            model_name, 
            learning_rate=self.config.FINE_TUNE_LEARNING_RATE
        )

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5),
            ModelCheckpoint(
                f"{self.config.MODEL_SAVE_DIR}/{model_name}_fine_tuned.h5",
                save_best_only=True
            )
        ]

        start_time = time.time()
        history = model.fit(
            train_data,
            epochs=self.config.FINE_TUNE_EPOCHS,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
```

## Results

We trained the model on an [urban tree dataset](https://www.kaggle.com/datasets/erickendric/tree-dataset-of-urban-street-classification-tree) with 4804 images across 23 classes. The data was split into training, validation, and test sets.

We used the following hyperparameters, set in [config.py](https://github.com/bdjohnson529/tree_classifier/blob/master/src/config.py):
```python
# Training parameters
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 1
INITIAL_LEARNING_RATE = 0.003
FINE_TUNE_LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.3
```

Initially, we focused on the feature extraction phase. The model quickly learned the training data, achieving an accuracy of 98% after just 10 epochs. However, it struggled with the validation set, indicating overfitting.

We adjusted the hyperparameters, reducing the learning rate and increasing the dropout rate. However, these did not have much of an effect - the accuracy on the validation data remained low. 

<figure style="margin: 30px 0; text-align: center;">
  <img src="{{ '/assets/images/overtrain.png' | relative_url }}" 
       alt="Overfitting Example" 
       style="max-width: 75%; padding: 20px; margin: 20px auto; display: block; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
  <figcaption style="margin-top: 15px; font-style: italic; color: #666;">Figure 1: Overfitting in one of the early training runs. The model performs well on training data but poorly on unseen data.</figcaption>
</figure>

Eventually we discovered that we were unfreezing too many layers in the base model. This caused the model to overfit to the training data, as it was trying to learn too many task-specific features. We reduced the number of unfrozen layers to 2, and the validation accuracy improved significantly.

<figure style="margin: 30px 0; text-align: center;">
  <img src="{{ '/assets/images/improved-validation.png' | relative_url }}" 
       alt="Improved Validation Accuracy" 
       style="max-width: 75%; padding: 20px; margin: 20px auto; display: block; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
  <figcaption style="margin-top: 15px; font-style: italic; color: #666;">Figure 2: Improved validation accuracy after reducing the number of unfrozen layers.</figcaption>
</figure>

The fine-tuning phase did not yield significant improvements, likely due to the limited size of the dataset. However, the model was able to adapt its learned features to better fit the specific task.

<figure style="margin: 30px 0; text-align: center;">
  <img src="{{ '/assets/images/fine-tuned.png' | relative_url }}" 
       alt="Fine-Tuned Model Accuracy" 
       style="max-width: 75%; padding: 20px; margin: 20px auto; display: block; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
  <figcaption style="margin-top: 15px; font-style: italic; color: #666;">Figure 3: Fine-tuned model accuracy after further training.</figcaption>
</figure>

## Conclusion

We took a model which was trained on a generalized dataset and fine-tuned it on our specific tree dataset. This transfer learning approach allowed us to adapt the model's knowledge to a specific task with limited data.

We were able to increase the validation accuracy to 60% for the tree classification task with 23 classes. This is significantly better than the accuracy we would have achieved with a randomly initialized model. However, it is far from reliable for production use.

Future work would focus on improving the model's performance by:
- **Collecting more data**: Increasing the size and diversity of the dataset can help the model learn better representations.
- **Different models**: Using other pre-trained models like ResNet or EfficientNet may yield better results.
- **Hyperparameter tuning**: Further optimizing the learning rate, dropout rate, and number of unfrozen layers can improve performance.
- **Data augmentation**: Applying techniques like rotation, flipping, and cropping can help the model generalize better to unseen data.