# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network (i have only free github service and i couldn't upload this file since this file is around 200 MB (even 142 MB after zipping). I'm expecting my fellow reviewer to show me a way to share this file, i.e. dropbox, google drive etc.) 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

For the final model, i have used a modified VGG model i have found on the net. The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

I have tried to use Comm.ai and NVIDIA NNs also, but it will take too much time on my machine (1 epoch/half a day, which is unaccaptable). Comm.ai seems faster, so i have tried that but i have had to make many trials, which means couple weeks more, to get to a meaningful result. So, i also gave up trying Comm.ai net. So "modified VGG" seems a good solution for my own case, because it is faster and it has enough layers for training as well as introducing nonlinearity.

### Model Architecture and Training Strategy

Unfortunately, with the modified VGG and Comm.ai networks i have used, i couldn't manage the simulator to drive all around the track-1. The reasons are:

1 - I didn't even know, before starting this course, that i will be needing some external resources like Amazon AWS. Unfortunately, right now i'm jobless living in another country with some living condition difficulties and 1000 Canadian dollars i have paid for the course, is already too too much for me. Now, i declined to use some other resources. Even they are for couple pennies, i'm refusing to do so. So, theres no way i will be using Amazon AWS or github pro or another external resource, i'm done with this (unless i can find some job next week).

2 - Actually, i have a good machine with GPU. I thought i was using GPU but in fact i couldn't because now i have learned that i have to install drivers, i have to run some scripts, adjustments and couple of startups etc etc. (i thought installing tf-gpu would suffice) Well, i'm afraid to do it to a working system because this is the only thing i can work on and what if something happens on the way. I'm sure Udacity will not pay if anything bad happens (like unstability or workstation/graphic card burn down etc). So, this is also not a choice for me for the time being. But, since i have 64 GBs of RAM, i can try trainings on my own machine (which i have been doing for over last 2 weeks).

3 - Now, i have to use my own machine, which is a good machine in fact. I have been doing the training for "Behavioral Cloning" project for over 2 weeks and the results given as follows. I have failed for the simulator to drive all around the track. What i have done so far will be discussed in next topics. I need some insight and solution from my fellow colleagues in Udacity.

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 to 64. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains Maxpooling and dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

The Udacity training data was chosen as a base. I have added some personal data on top of it in the later phases of the project. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

At first, i have to make sure that i can read the data and feed it to a basic 1 layer fully connected layer as it was suggested in Udacity training.

The overall strategy for deriving a model architecture was to start with training set given by Udacity and find a proper/well known model already tested and used widely (Googlenet, VGGNet, Nvidia etc). All i have to look, considering the situation explained before, the network should train fast, include Convolutional (network has to learn different characteristics of the track) and Fully Connected layers, layers to introduce nonlinearity (relu) and also layers for avoiding over&under fitting (Droput and Maxpooling layers). 

In order to test how well the model was working, I split my image and steering angle data into a training and validation set. I have used different split percentages, 20%, 25% and 30%. 

The solutions steps and what have i done in the process will be covered in the following in Section 3 of this writeup.

At the end of the process, the vehicle got off the track when it saw the paved way. I need extra suggestions from Udacity reviewers and fellows, considering the situation i have described at the beginning.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

Layer
Details

Cropping layer
Top Crop : 70
Bottom Crop : 25

Convolution Layer 1
Filters: 32
Stride: 3 x 3
Padding: SAME
Activation: relu

Convolution Layer 2
Filters: 32
Stride: 3 x 3
Padding: SAME
Activation: relu

Maxpooling layer
Pool Size : 2 x 2

Dropout layer (0.25)

Convolution Layer 3
Filters: 64
Stride: 3 x 3
Padding: SAME
Activation: relu

Convolution Layer 4
Filters: 64
Stride: 3 x 3
Padding: SAME
Activation: relu

Maxpooling layer
Pool Size : 2 x 2

Dropout layer (0.25)

Flatten layer

Fully Connected Layer 1
Neurons: 256
Activation: relu

Dropout layer (0.5)

Fully Connected Layer 2
Neurons: 10
Activation: relu

Fully Connected Layer 3
Neurons: 1


#### 3. Creation of the Training Set & Training Process

I have first used the training data given by Udacity and i've seen that the data is not enough for training. Also, training took so much time so i decided to crop images. As a result, the car can not even reach to the bridge. I have then introduced left and right cam images to the training data. With the model (VGG) i have used, it seems pretty stable at the beginning, but right after the bridge the car left the track whenever it sees the paved way. To augment the data set, I also flipped images and angles thinking that this would help training by adding more data to the set. But it didn't help, the car was still choosing the paved way.

To overcome this problem, i have recorded the specific section of the track where the car left the track, for once. And added this new data to the training set. Since this data is for only some specific part of the track, and it was added to end of the whole data set, i randomly shuffled the whole data before introducing it to network (considering this whole data at the end would be chosen as validation data if it wasn't shuffled). After some training, i have seen the same behaviour.

Then, i decided to have more data, with the specific section of the track. i have recorded 3-4 times passing this area, including reverse driving. But again with no luck, and actually things got worse and the car even couldn't pass the first bridge. (Each trial and error took at least half a day by the way).

Finally, i have decided that instead of a part of a track i have to add full track record to the existing data, which seems more logical because i maybe only adding some inconsistent, only a part of a track to add more noise instead of a useful data. I have added 1 full track-1 record to the data and i have run it. Now, after more than 2,5 weeks of trial and error, i'm at the same point. The car is leaving the path whenever it sees the paved way after the bridge.

For once though, the car left the track, drive all the way thru paved way and got back to track. I wish i have saved that model and sent it to you but i didn't because i thought things would have got better with the more data i have introduced.

##### 3.1 Observations

1 - Whether i shuffled the data or not, after 5 to 10 epochs validation loss stays same around 0.0150.
2 - The training loss also didn't change much after 20 epochs.
3 - Changing of batch_size has sometimes considerable effect on the result (batch_sizss of 32,64,128 and 256 tested)
4 - I also played with validation_split, which has not big impact on the results (tests done with 20%, 25% and 30%)
5 - The max number of input images was 60.000 which is a good number. If suggested, i can add more data but it seems no impact after this point, the network more or less behaved same. 


#### 4. What can be done next

1 - I'm expecting Udacity reviewer valuable ideas to go on, because now i'm stuck and i'm not learning anything anymore. If reviewer suggests more trial&error, i should state that i have no time/energy/motivation left to go on.

2 - I'm now considering to change the depths of convolution layers from 32-32-64-64 to 16-32-64-128 to extract more features out of the images. I might try that once or twice.

3 - There may be too much dropout going on in the existing structure. I may consider less dropouts (e.g. change ratio from 0.25 to 0.5), if fellow reviewer suggests to do so.

4 - I may consider a new structure if reviewer give me the whole structure. I don't have time and motivation to jump from one structure to another for another couple weeks with my existing capabilities.

Other than that, thanks.
