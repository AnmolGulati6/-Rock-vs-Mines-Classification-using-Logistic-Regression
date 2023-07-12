# Rock vs Mines Classification using Logistic Regression

This project focuses on building a logistic regression model to classify rocks and mines based on sonar data. The dataset used for this project is a supervised learning dataset, where each data entry represents various features of sonar signals, and the corresponding label indicates whether it is a rock or a mine.

## Project Overview

The main objective of this project is to train a logistic regression model using the sonar data to accurately predict whether a submarine is running into a rock or a mine. The following steps are performed in this project:

1. Data collection and processing: The sonar data is loaded into a pandas dataframe, and basic exploratory data analysis is conducted to understand the dataset.

2. Data preparation: The data is separated into input features (X) and the corresponding labels (Y).

3. Training and test data split: The dataset is split into training and test sets using the `train_test_split` function from scikit-learn. This allows us to evaluate the performance of the model on unseen data.

4. Model training: A logistic regression model is created using scikit-learn's `LogisticRegression` class and trained on the training data.

5. Model evaluation: The accuracy of the model is measured using the training and test data. This gives us an understanding of how well the model performs.

6. Making predictions: Finally, a predictive system is implemented to make predictions on new, unseen data. A sample data point is taken, converted into a numpy array, and fed into the trained model for prediction.

## Dependencies

The following dependencies are required to run the project:

- numpy 
- pandas
- scikit-learn

## Instructions

1. Ensure that the required dependencies are installed on your system.

2. Download the dataset 'sample_data.csv' and place it in the same directory as the project files.

3. Run the 'rocks_vs_mines_logistic_regression.py' script.

4. The script will train the logistic regression model, evaluate its accuracy on the training and test data, and then make a prediction on a sample data point.

5. The predicted label (rock or mine) will be displayed in the console.

Feel free to explore and modify the code according to your requirements.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

The sonar dataset used in this project is sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)).
