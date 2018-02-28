# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


Track 1            |  Track 2
:-------------------------:|:-------------------------:
![alt text][image16]  |  ![alt text][image17]
[Youtube](https://youtu.be/DmBIX7k2Rws) / [File](video.mp4) | [Youtube](https://youtu.be/0o0WiMw92Nc) / [File](video_track2.mp4)


### Setup
* Download the simulator:
  * [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
  * [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
  * [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)
* Open the simulator and run `python drive.py model.h5`
* Get the sample data set of track 1 [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). 

[//]: # (Image References)

[image1]: ./examples/histogram.png "Data Distribution - Track 1"
[image2]: ./examples/cameras.png "Cameras"
[image3]: ./examples/angles.png "Steering Angles"
[image4]: ./examples/histogram_fake.png "Data Distribution with Fake Data"
[image5]: ./examples/multiple_cameras.png "Multiple Cameras"
[image6]: ./examples/flip.png "Flipping"
[image7]: ./examples/brightness.png "Brightness"
[image8]: ./examples/translate.png "Translate"
[image9]: ./examples/rgb.png "RGB"
[image10]: ./examples/hsv.png "HSV"
[image11]: ./examples/yuv.png "YUV"
[image12]: ./examples/normalization.png "Normalization"
[image13]: ./examples/cropped.png "Cropped"
[image14]: ./examples/histogram2.png "Data Distribution - Track 1 & 2"
[image15]: ./examples/nvidia.png "NVIDIA"
[image16]: ./examples/track1.gif "Track 1 Result"
[image17]: ./examples/track2.gif "Track 2 Result"

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the [NVIDIA model](https://arxiv.org/pdf/1604.07316.pdf), it consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 ([model.py lines 145-160](model.py#L145-L160)) 

The model includes RELU layers to introduce nonlinearity ([code line 150-154](model.py#L150-L154)), and the data is normalized in the model using a Keras lambda layer ([code line 147](model.py#L147)). 

#### 2. Attempts to reduce overfitting in the model

The model contains data augmentation in order to reduce overfitting ([model.py lines 32-94](model.py#L32-L94)). 

The model was trained and validated on different data sets to ensure that the model was not overfitting ([code line 21](model.py#L21)). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually ([model.py line 165](model.py#L165)).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving of track 1 and track 2. It's a set of center lane driving with images taken from 3 cameras mounted at the front of a car.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to go simple first, then add complexity as needed. The goal was train a model to predict the steering angles for a car to drive by itself in the simulator. 

A summary of the data set:
* The size of training set is 8335
* The size of the validation set is 2084
* The shape of a traffic sign image is (160, 320, 3)
* The range of steering angles in the data set is -1.0 to 1.0

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute over angles ranging from -1 to 1. Each bin represents certain range of angles and its height indicates how many examples in that range.

![alt text][image1]

The car is mounted with 3 cameras: center, left, and right. Here are the images taken from these cameras at the same point of time.

![alt text][image2]

Here are the images taken from center camera with its corresponding steering angle range.

![alt text][image3]

My first step was to use the convolution neural network model from [NVIDIA](https://arxiv.org/pdf/1604.07316.pdf) with samples from track 1. I thought this model might be appropriate because it was known that it works well for self driving car.

In order to gauge how well the model was working, I split the sample data into a training and validation set. I found that my first model had a low mean squared errors on both the training and the validation set, but performs poorly on the simulator. I had a highly unbalanced dataset. The dataset have an excessive amount of small angles(i.e. straight), and the model trained from these will be highly biased towards driving straight. Track 1 has relatively more straight parts, and hence the lower errors. Another reason might be that I was only validating with images from center camera. It assumed the car will always be perfectly at the center of the lane, and this does not reflect the real situation on simulator.

This implied that the model was overfitting. To combat the overfitting, I added more data from track 2 to the model.

I ran the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially at the curving parts. To improve the driving behavior in these cases, I further augmented the data set using multiple cameras as decribed in following sections.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. 

#### 2. Final Model Architecture

The final model architecture ([model.py lines 145-160](model.py#L145-L160)) consisted of a few additional Lambda and Cropping2D layers at the top on the NVIDIA architecture. Here's a visualization: 

![alt text][image15]

And the visualization of the final architecture
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
lambda_2 (Lambda)                (None, 160, 320, 1)   0           lambda_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 1)    0           lambda_2[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   624         cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
====================================================================================================
Total params: 980,619
Trainable params: 980,619
Non-trainable params: 0
____________________________________________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded an additional lap from track 2, then combined it with the provided sample dataset. 

Track 2 has more curves, and with a keyboard, it was harder for me to keep it perfectly at the center of the lane. In this way, I was able to train the model on handling curves and recovering from side of roads. Here's the new data distribution:

![alt text][image14]

To further train the model on recovering from the sides of the road back to center, I then randomly chose images from one of the cameras. For images from left/right camera, I used a steering angle generated by adding/subtracting from the center steering angle.

![alt text][image5]

I then preprocessed this data by normalization, cropping, and converting to HSV color space.

Here's an example of normalized input image:

![alt text][image12]

Here's a comparison of images under different color spaces:

![alt text][image9]
![alt text][image10]
![alt text][image11]

I decided to use S Channel for the input images. This reduced the data size and training time.

The cameras are mounted to the front of the car, and the road will always appear in the same general region in the image. By keeping the useful information and removing top(trees, hill, and sky) and bottom(hood of car) portions, the model might train faster.

![alt text][image13]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I trained the model over 3 epochs, and suprisingly, the car was able to successfully drive by itself around track 1. But it wasn't working well for track 2.

To conquor the challenge of track 2, I further augmented the dataset.

I balanced the dataset distribution by generating additional augmented data for steering angles ranging from -1 to 1. Here's the new dataset distribution:

![alt text][image4]

To augment the data set, I randomly flipped images and angles thinking that this would help the model to be more robust. For example, here is an image that has then been flipped:

![alt text][image6]

I also randomly adjusted the brightness of the images, to simulate various lighting condition at different times of a day.

![alt text][image7]

I also randomly translated the image and updated its steering angle accordingly.

![alt text][image8]

I trained the over 6 epochs, the car was able to autonomously drive around track 2 for a lap without leaving the lane. 

#### 4. Future Work
* Imporve the model to perform better on track 2.
* Train with different model architecture such as [SqueezeNet](https://arxiv.org/pdf/1602.07360.pdf)

