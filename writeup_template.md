#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/pidatascience/carnd_traffic_signs/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Numpy's array [shape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html) method can be used to extract
the size of the training and testing sets. The number of unique labels in the training set can also be obtained directly witht the
help of Numpy's [unique](https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html) method call.

Detailed use of these functions can be observed in the second code cell with the following results

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Two verification methods were used to identify the data set. First, a visual verification of a random sample of 4 images and their corresponding identification labels are displayed in order to perform a sanity check on the set of predefined labels loaded as a [pandas](http://pandas.pydata.org/) dataset from set of [provided labels](https://github.com/pidatascience/carnd_traffic_signs/blob/master/signnames.csv). 

![end of no passing][https://github.com/pidatascience/carnd_traffic_signs/blob/master/label_verification.png]

Next a statistical visualization of the data was performed as a histogram of class labels in the training set:

![training histogram][https://github.com/pidatascience/carnd_traffic_signs/blob/master/training_histogram.png

There some visible variance in the distribution of training samples that may skew the output softmax probabilities of the
model, especially when dealing with under represented classes if the traning distribution differs greatly from the distribution of classes observed in german roads. 

This effect could be accounted for by treating the output of the softmax as the approximation of the joint probability and normalize using Bayes under a different prior but we lack any evidence to suggest a better distribution other than the one provided with the training set. It was chosen to continue and use the softmax output of the model without further modification.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Input data preprocessing was performed in the 5th code cell and consists of simple normalization of the input pixel values to the [-1, 1] range and one-hot-encoding of the output lables.

This method was selected both as a recommendation on [George's Hinton Neural Networks for Machine Learning](https://www.coursera.org/learn/neural-networks) course and from the project Q/A on Youtube. There was an alternative YUV conversion method considered but was dropped favoring the selected one in terms of implementation/verification time.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

In the 6th code cell a 'kfold' training set is defined that merges the training and validation set provided with the project. This set is used in two ways:
* As a training set to select the best set of hyperparameters using Grid Search with a low epoch count and k-fold cross validation with K=3
* As a final training set for the model with a full epoch count

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model and hyperparameters are defined in the 7th code cell using [Keras](https://keras.io); the model consists of the following layers:

| Layer         		|     Description	        					            | 
|:---------------------:|:---------------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							            | 
| Convolution        	| 1x1 stride, 3x3 kernel size, valid padding, 16 filters  	|
| Max pooling	      	| 2x2 pool size, 1x1 stride      				            |
| Activation    		| relu											            |
| Dropout       		| rate=0.2										            |
| Convolution        	| 1x1 stride, 3x3 kernel size, valid padding, 8 filters  	|
| Max pooling	      	| 2x2 pool size, 1x1 stride      				            |
| Activation    		| relu											            |
| Dropout       		| rate=0.2										            |
| Flatten				|           												|
| Fully connected		| 129 output size        									|
| Fully connected		| 82 output size        									|
| Fully connected		| 43 output size        									|
| Softmax				|                                                           |

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Training takes place in the 9th and 10th cells in two steps. 

First, a small epoch count (20) is first used to a small exploration of the parameter space wrapping the Keras model into numpy using a [KerasClassifier](https://keras.io/scikit-learn-api/#wrappers-for-the-scikit-learn-api) and performing GridSearchCV over a small section of the parameter space using k=3 fold cross validation.
Since this Grid search increases with the cardinality of the hyperparameter space (12 in this case) it was imperative to limit the interesting region to no more than 2 dimensions and values given hardware constraints and a local execution model.
In any case the sample application of the method presented allowed to select an optimal (in the search space) dropout rate of 0.2.

Secondly, the full training set was used to train the model and selected hyperparameter values with a full epoch count (200) in order to achieve the maximum possible accuracy for traning and validation with the defined folds. These were graphed to give a visual indication of the behavior of the model with increased training:

[Training accuracy](https://github.com/pidatascience/carnd_traffic_signs/blob/master/train_accuracy.png)

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 11th code cell of the Ipython notebook.

Final model results were:
* Test set accuracy of 80.54%

The model is loosely based on the LeNet 5 architecture discussed in the lessons with the addition of dropout for the hidden layers. This seems like a reasonable choice to balance implementation (and verification) simplicity for learning purposes with a reasonable expectation of medium level accuracy.

The test accuracy results show accuracy to be rather low at around 80%. Given the plateau of the k-fold validation accuracy graphed during training, the expectation of significantly increased accuracy with increased training epochs seems fairly low compared to similar values typical of DNN approaches. Increasing network capacity through a mixture of addition of layers and increased layer capacity (convolution filters and fully connected layer sizes) would have to be considered 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Image 1][https://github.com/pidatascience/carnd_traffic_signs/blob/master/image1.jpg] 
![Image 2][https://github.com/pidatascience/carnd_traffic_signs/blob/master/image2.jpg]
![Image 3][https://github.com/pidatascience/carnd_traffic_signs/blob/master/image3.jpg] 
![Image 4][https://github.com/pidatascience/carnd_traffic_signs/blob/master/image4.jpg]
![Image 5][https://github.com/pidatascience/carnd_traffic_signs/blob/master/image5.jpg]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on the final model is located in the 12th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Roundabout mandatory	| Rondabout mandatory							|
| Speed Limit (30km/h)	| Speed Limit (30km/h)							|
| No Entry	      		| Priority Road 				 				|
| Pedestrians			| Slippery Road      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of 80%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th code cell of the Ipython notebook.

 The top five soft max probabilities for each of the 5 figures

Figure 1 (stop sign):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .67         			| Stop sign   									| 
| .14     				| 60 km/h 										|
| .08					| 70 km/h										|
| .07	      			| 30 km/h       				 				|
| .03				    | Slippery Road      							|

Figure 2 (roundabout):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .74         			| Roundabout mandatory							| 
| .17     				| Priority road									|
| .03					| Keep right									|
| .02	      			| Go straight or right			 				|
| .01				    | No Vehicles         							|

Figure 3 (30 km/h):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .89         			| 30 km/h           							| 
| .06     				| 20 km/h           							| 
| .03					| 50 km/h           							| 
| .01	      			| 60 km/h           							| 
| .00				    | 80 km/h           							| 

Figure 4 (no entry):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .47         			| Priority road        							| 
| .25     				| Ahead only         							| 
| .09					| No Vehicles       							| 
| .04	      			| Turn left ahead    							| 
| .04				    | Go straight or right 							| 

Figure 5 (pedestrians):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .39         			| Slippery road        							| 
| .36     				| Right-of-way at the next intersection         | 
| .07					| Traffic signals      							| 
| .05	      			| Wild animals crossing							| 
| .03				    | Double curve          						| 