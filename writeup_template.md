# **Traffic Sign Recognition** 

## Writeup

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

[histo]: ./imgs/histogram.png "Histogam"
[samples]: ./imgs/samples.png "Samples"
[pre_orig]: ./imgs/pre_orig.png "Original Image"
[pre_post]: ./imgs/pre_post.png "Processed Image"
[new_samples]: ./imgs/new_samples.png "New samples"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Project code is included in this zip file.
Project code can be found in ['./Traffic_Sign_Classifier.ipynb']('./Traffic_Sign_Classifier.ipynb'). Example output of the iPython notebook can be found ['./Traffic_Sign_Classifier.html']('./Traffic_Sign_Classifier.html'). 

### Data Set Summary & Exploration

#### 1. Basic summary of the data set


* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

At first, the histogam of the training data is shown.
On the x-axis, the class id is shown. The bars indicate the number of training images for each class. It can be seen, that there are classes, with large number of training samples and there are classes with few number of training samples. 

![Histogram][histo]

The following figure contains some sample images from the training data set.

![Samples][samples]

It can be seen that there are some very bright and some very dark images. This issue will be adressed in the preprocessing step.

### Design and Test a Model Architecture


#### 1. Preprocessing step

In preprocessing, three steps are done. First, histogram equalization is applied to the image. This is resolve the issue, that there are some images very bright and others are very dark.
```python
import cv2

def convert_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img
```
The next figure shows a sample training image:

![Original Image][pre_orig]


The following figure shows the same image after preprocessing:

![Processed Image][pre_post]

Then the images were normalized:

```python
def scale(X):
    return (X-128.0)/128
```

In the third step, some classes are extended with some images.
During training it has been shown, that lots of images of class 21 were wrongly classified as class 31. The reason for this behaviour is probably the similarity of the two classes (double curve and deer passing) and their different number of trainings samples. This issue was addresed by duplicating images from class 21, so that the number ot training samples of class 21 and 31 are equal. An augmentation was not applied, only the images were copied. Better results may have been achieved by augmentation of those images.

```python
# add samples for classes with problems...
n_add = (y_train==31).sum() - (y_train==21).sum()
valids = y_train==21
v = np.array(range(len(valids)))
v = v[valids]
randy = np.random.randint(len(v), size=n_add)
y_train = np.append(y_train, y_train[v[randy]], axis=0)
X_train = np.append(X_train, X_train[v[randy]], axis=0)
```

The same was applied for classes 5 and 3.

#### 2. Final model architecture

The final model was very close to the original LeNet5 architecture.Input channels were extended to 3. The depth of the first convolutional layer was extended to reflect the increaed number of input channels. During training it has been found, that the training error was close to 100%, but validation error was close to 90%. To resolve this issue, dropout was added to both neural network layers.

The following table shows the architecture of the neural network:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	      	    | outputs 400 				|
| Fully connected		| outputs 120        									|
| RELU					|												|
| Dropout   			| Keep Probability 75%												|
| Fully connected		| Output 84        									|
| RELU					|												|
| Dropout   			| Keep Probability 75%												|
| Fully connected		| Output 43        									|
| Softmax				|         									|

#### 3. Training of the model

The parameters for training are close to the original parameters. Because of the dropout, more epochs were needed. It was also found, that smaller trainings rate would perform better.

Parameters are:

| Parameter | Value |
|:------:|:------------:| 
| EPOCHS  | 50 |
| BATCH_SIZE | 128 |
| rate | 0.0005 |
| Optimizer| AdamOptimizer |


#### 4. Approach to get the final results

The final results are:

| What | Accuracy |
|:------:|:------------:| 
| train | 0.995 |
| validate | 0.973 |
| test | 0.838 |

Following steps were taken to get to the final results:

* Starting with LeNet5 architecture
   * Output classes changed to 43
   * Input depth changed to 3
   * Depth of first convolutional layer changed to 18
* Analysis of first training
   * See comments in section 1. preprocessing
* Images added during preprocessing stage (as described in section 1.)
* Analysis of next training
   * Training result close to 100%
   * Validation result below 93%
* Dropout added to both full connected layers added to fix overfitting
* Analysis of next training
   * Training result less than 90%
   * Validation closer to training than before
* Increased number of epochs. Decreased learning rate
* Analysis of next training
   * Training result close to 100%
   * Validation result above 93%

### Test a Model on New Images

#### 1. Images

The new images are shown in the following figure

![new samples][new_samples]

All five images are good visible and should be easy to classify. But it could be possible, that the first image gets confused with another speed limit sign.  

#### 2.Predictions

The predicted classes are shown in the following table

| Image	     |     Prediction			| 
|:---------------------:|:-----------------------------:| 
| Speed limit (80km/h)      		| Speed limit (80km/h)		| 
| End of all speed and passing limits| End of all speed and passing limits 		|
| No vehicles		| No vehicles	|
| Stop      		| Stop	|
| Priority road	| Priority road |

All images are classified correctly. Accuracy is 1.0. This is clearly more than the accuracy of the test set. This is probably because of the very good five taken images. Another reason is probably the small number of additional test samples.

#### 3. Certainties of the new predictions

The results of the softmax probabilities are shown in the following tables:

Results for image 1

|Probability|Prediction|
|---|---|
|1.00000| Speed limit (80km/h)|
|0.00000| No passing for vehicles over 3.5 metric tons|
|0.00000| Speed limit (30km/h)|
|0.00000| Road work|
|0.00000| Speed limit (60km/h)|

Results for image 2

|Probability|Prediction|
|---|---|
|0.99918| End of all speed and passing limits|
|0.00054| End of no passing|
|0.00026| End of speed limit (80km/h)|
|0.00002| Priority road|
|0.00000| Keep right|

Results for image 3

|Probability|Prediction|
|---|---|
|0.99996| No vehicles|
|0.00004| Speed limit (70km/h)|
|0.00000| Yield|
|0.00000| Speed limit (50km/h)|
|0.00000| Speed limit (120km/h)|

Results for image 4

|Probability|Prediction|
|---|---|
|0.99996 |Stop|
|0.00003| Speed limit (80km/h)|
|0.00001| No vehicles|
|0.00000| No entry|
|0.00000| Speed limit (50km/h)|

Results for image 5

|Probability|Prediction|
|---|---|
|1.00000| Priority road|
|0.00000| End of all speed and passing limits|
|0.00000| Traffic signals|
|0.00000| No vehicles|
|0.00000| End of no passing|


Each image is correctly idenitified with a high probability. 
