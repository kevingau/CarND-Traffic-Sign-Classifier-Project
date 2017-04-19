**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualization.png "Visualization"
[image2]: ./beforegray.png "beforegray"
[image3]: ./aftergray.png "aftergray"
[image4]: ./test_images/4.jpg "Traffic Sign 1"
[image5]: ./test_images/14.jpg "Traffic Sign 2"
[image6]: ./test_images/17.jpg "Traffic Sign 3"
[image7]: ./test_images/35.jpg "Traffic Sign 4"
[image8]: ./test_images/40.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

The code for this step is contained in the second code cell of the IPython notebook.

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.

Here is an exploratory visualization of the data set. Here's a histogram showing the distribution inside both the training and the testing set.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

To add more data to the data set, I rotate some images then add it to the dataset, ensuring that the labels which have larger number of images do not bias the network.

For labels with #images < 500: I have generated 3 additional images by rotating the images by 3 randomly chosen angles between -10 and 10 degrees.

For labels with #images < 1000: I have generated 1 additional image by rotating the images by 1 randomly chosen angle between -10 and 10 degrees.

The new size of training set is 57028.

After adding images,I convert the images to grayscale and normalize it because my algorithm can detect edges easier.

Here is an example of a traffic sign image before and after grayscaling and normalizing.

![alt text][image2] ![alt text][image3]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifth cell of the ipython notebook.


| Layer					| Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x1 grayscale image						| 
| Convolution layer		| 1x1 filter, SAME								|
| Convolution layer		| 5x5 filter, stride 1 to 28x28x6				|
| RELU					|												|
| Max pooling			| 2x2 kernel, 2x2 stride, outputs 14x14x6 		|
| Convolution layer		| 5x5 filter, stride 1 to 10x10x16				|
| RELU					|												|
| Max pooling			| 2x2 kernel, 2x2 stride, outputs 5x5x16 		|
| Flatten				| 400 nodes										|
| Dropout				| keep probability = 0.75						|
| Fully connected layer	| 400 -> 120									|
| RELU					| 												|
| Fully connected layer	| 120 -> 84										|
| RELU					| 												|
| Fully connected layer	| 84 -> 43										|


####3. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the sixth cell of the ipython notebook. 

To train the model, I tried out the following variables.

Optimizers tried: [AdamOptimizer, AdagradOptimizer]

learning_rates: [0.1, 0.01, 0.005, 0.001, 0.0005]

Learning rate of 0.001 worked out best for me. AdamOptimizer worked well for my model architecture.

####4. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the sixth cell of the Ipython notebook.

My final model results were:
* training set accuracy = 0.991
* validation set accuracy = 0.931
* test set accuracy = 0.910

If a well known architecture was chosen:
* What architecture was chosen?
    LeNet
* Why did you believe it would be relevant to the traffic sign application?
    LeNet architecture is said to be accurate for small amount of computation like in our case. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    The accuracy on the validation set is greater than 0.93, which give out a 100% accuracy in identifying new images.
 
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image, Speed limit (70km/h) sign, might be difficult to classify because it contains text in the middle with a specific speed limit, which may confused with other speed limit signs.

The last image may be difficult to classify because it is not so clear at the resolution 32x32.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the eighth cell of the Ipython notebook.

Here are the results of the prediction:

| Image					| Prediction						| 
|:---------------------:|:---------------------------------:| 
| Speed limit (70km/h)	| Speed limit (70km/h)				| 
| Stop					| Stop								|
| No entry				| No entry							|
| Ahead only			| Ahead only		 				|
| Roundabout mandatory	| Roundabout mandatory				|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

For the first image, the model is relatively sure that this is No. 4, Speed limit (70km/h) sign (probability of 1.0). The top five soft max probabilities were

| Probability			| Prediction No. of Labels			| 
|:---------------------:|:---------------------------------:| 
| 1.0					| No. 4								| 
| 0.0					| No. 1								|
| 0.0					| No. 8								|
| 0.0					| No. 2				 				|
| 0.0					| No. 5								|


For the second image, the model is relatively sure that this is No. 14, Stop sign (probability of 0.99). The top five soft max probabilities were

| Probability			| Prediction No. of Labels			| 
|:---------------------:|:---------------------------------:| 
| 1.0					| No. 14							| 
| 0.0					| No. 38							|
| 0.0					| No. 1								|
| 0.0					| No. 33			 				|
| 0.0					| No. 15							|


For the third image, the model is relatively sure that this is No. 17, No entry sign (probability of 1.0). The top five soft max probabilities were

| Probability			| Prediction No. of Labels			| 
|:---------------------:|:---------------------------------:| 
| 1.0					| No. 17							| 
| 0.0					| No. 14							|
| 0.0					| No. 33							|
| 0.0					| No. 26			 				|
| 0.0					| No. 36							|


For the fourth image, the model is relatively sure that this is No. 35, Ahead only sign (probability of 1.0). The top five soft max probabilities were

| Probability			| Prediction No. of Labels			| 
|:---------------------:|:---------------------------------:| 
| 1.0					| No. 35							| 
| 0.0					| No. 10							|
| 0.0					| No. 9								|
| 0.0					| No. 38			 				|
| 0.0					| No. 13							|


For the fifth image, the model is relatively sure that this is No. 40, Roundabout mandatory sign (probability of 0.). The top five soft max probabilities were

| Probability			| Prediction No. of Labels			| 
|:---------------------:|:---------------------------------:| 
| 1.0					| No. 40							| 
| 0.0					| No. 37							|
| 0.0					| No. 39							|
| 0.0					| No. 1				 				|
| 0.0					| No. 2								|