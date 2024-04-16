# COMP425 - Homework 3: Visual Recognition System
### Overview
- **Subject**: COMP425/COMP6341 Computer Vision, Winter 2024 (Université Concordia).
- **Grade**: 20/20.

### Key Components
- **Defining a filter bank**: I used LMFilters.py to generate a bank containing 48 49x49-dimensional filters.
- **Feature extraction**: Applying the filters and to training images and generate the responses (that will lead to extracting features).
- **K-Means Clustering and L2 distance**: By running a clustering algorithm (K-Means), each pixel is assigned to one of the K visual words according to which cluster it is closest to (defining closeness using the L2 distance).
- **Bag-of-Words Representation**: Represent the frequency of a particular visual word in the image as a histogram.
- **Classification**: Matching the testing images with each of the training images in terms of the L2 distance between their BoW representations. 

### Instructions to run the code
Navigate to the corresponding directory within the project folder in your terminal and execute the specific script for each part of the assignment. 
- For generating the model 'model.pkl' and the BoW representations of the training images, run the following command:
  
    ```bash
    python run_train.py
  
- To classify the testing images based on the model, run the following command:
    ```bash
    python run_test.py

### Contribution Instructions
Contributions are welcome!
Please fork the repository and submit a pull request with your changes.
Feel free to customize it further to match your repository's specific details and needs!
