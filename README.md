# Computer Vision with MNIST Dataset

Welcome to my Computer Vision project focusing on the MNIST dataset! This repository contains a Jupyter Colab notebook with implementations of three different models for digit classification.

## Models Overview

### 1. Custom CNN Model, LeNet-5, and Parameter-Reduced LeNet-5
- In the notebook (`mnist_digit_classification.ipynb`), I've implemented three different models for digit classification using the MNIST dataset.
  
  - **Custom CNN Model:**
    - Designed a Convolutional Neural Network (CNN) with two blocks, each consisting of Conv2d, MaxPool2d, and ReLU layers. The model is then followed by a classifier layer with Flatten and a linear layer to achieve the desired output shape.

  - **LeNet-5 Architecture:**
    - Implemented the classic LeNet-5 architecture from scratch. This model has been historically significant in the development of Convolutional Neural Networks for image recognition tasks.

  - **Modified LeNet-5 for Parameter Reduction:**
    - Tweaked the LeNet-5 architecture to significantly reduce the number of parameters while maintaining a high level of accuracy. This optimization aims to achieve a more computationally efficient model.

## How to Use

1. Open the notebook (`mnist_digit_classification.ipynb`) in Google Colab or Jupyter Notebook.
2. Run the cells sequentially to train and evaluate the models.
3. Explore the impact of architecture changes on accuracy and model efficiency.

## Dataset
The project uses the MNIST dataset for digit recognition. You can find more information about the dataset [here](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST).


## Future Directions

I plan to expand on this project by exploring more advanced architectures, experimenting with data augmentation techniques, and possibly incorporating transfer learning for improved performance on related datasets.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code for your own learning and projects.

## Acknowledgments

Special thanks to the MNIST dataset contributors and the PyTorch community for their valuable resources and frameworks.

Happy coding!
