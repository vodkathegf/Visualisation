Learning Process Exploration with Scattered Dots
This project represents my initial foray into training a machine learning model and understanding the learning process. My primary goal was to gain insights into how a model learns from data and makes predictions.

Objective
Data Generation: Initially, I created scattered dots with two distinct colors - purple (labeled as 0) and yellow (labeled as 1). These colors serve as the target labels for the model to predict.

Model Training: I instructed the model to analyze the distribution of these scattered dots and learn to predict their colors based on their spatial arrangement. The model architecture consists of a feedforward neural network.

Prediction: The ultimate aim was for the model to accurately predict the colors of the scattered dots based on their positions in the dataset.

Observations
Accuracy Progression: I observed that the accuracy of the model in the initial epoch was considerably lower than in subsequent epochs. This can be attributed to the random initialization of the model's weights and the model's learning process improving over time.

Learning Rate Influence: The lower level of accuracy in the first epoch can also be attributed to the choice of a lower learning rate. As the model progresses through epochs, the learning rate allows it to refine its predictions, resulting in improved accuracy.

Overfitting Challenge: Despite the improvement in accuracy over epochs, I encountered challenges in preventing overfitting. Overfitting occurs when the model learns to memorize the training data instead of generalizing well to unseen data. This remains an area of exploration for further refinement of the model training process.

This project has provided valuable insights into the fundamentals of training a machine learning model, and I look forward to further experimentation and refinement to enhance its performance and generalization capabilities.
