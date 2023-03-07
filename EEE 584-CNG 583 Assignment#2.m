%% EEE 584 Assignment # 2: CNN Based Object Recognition 

clc
clear all

%% The Dataset name is "101_ObjectCategories" and the object to be recognize is "airplanes" and Motorbikes".

rootFolder_dataset = '.\\101_ObjectCategories\\';

% The objects to recognize in the dataset.
categories = {'airplanes', 'Motorbikes'};

% Required Output Image Dimensions
Output_Image_Dim =[75 75 1]; 

%% The Images from the "airplanes" and "Motorbikes" are Combined in a Dataset from 1:800 and 801:1598 respectively
%  I took first 400 images from each according to the requirement. 

training_Images_airplane = [1:400];
train_Images_motorbike = [801:800+400];

test_Images_airplane = [401:800];
test_Images_motorbike = [1201:1200+398];


% Training & Testing Datasets
training_dataset = [training_Images_airplane train_Images_motorbike];
testing_dataset = [test_Images_airplane test_Images_motorbike];

% I Created an Image Datastor for Object using the following Function.
imds = imageDatastore(fullfile(rootFolder_dataset,categories), 'LabelSource', 'foldernames');

%% Displaying Images from Dataset

figure();
perm = randperm(1598,25);
for i = 1:25
subplot(5,5,i);
imshow(imds.Files{perm(i)});
end

%% I Splited the Dataset into Training and Testing by following Functions

train_set = subset(imds,training_dataset);
test_set =subset(imds,testing_dataset);

% Shuffle the Images in the Dataset
train_set.shuffle();                  
test_set.shuffle();       

%% I Resized and converted to Gray-Scale Images
% Create an augmentedImageDatastore object to resize and convert rgb2gray
train_gray_Images = augmentedImageDatastore(Output_Image_Dim,...
                       train_set,'ColorPreprocessing','rgb2gray');

test_gray_Images = augmentedImageDatastore(Output_Image_Dim,...
                       test_set,'ColorPreprocessing','rgb2gray');
                   
%% I defined my CNN Model Layers as shown in below Code.

layers = [
imageInputLayer(Output_Image_Dim)

convolution2dLayer(3,8)
batchNormalizationLayer   
reluLayer
maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,16)
batchNormalizationLayer   
reluLayer
maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,32)
batchNormalizationLayer   
reluLayer
maxPooling2dLayer(2,'Stride',2)

fullyConnectedLayer(2)
softmaxLayer 
classificationLayer 
];

%% Training and Testing the Network
  
options = trainingOptions('sgdm', ...
'InitialLearnRate' ,0.01, ...
'MaxEpochs', 6, ...
'MiniBatchSize', 20, ...
'Shuffle','every-epoch', ...
'Verbose', true, ...
'Plots','training-progress');

net = trainNetwork(train_gray_Images,layers,options);

[predictedLabels Score]=classify(net,test_gray_Images);

%% Confusion Matrix and Accuracy

Conf_Mat=confusionmat(test_set.Labels,predictedLabels);

figure()
confusionchart(Conf_Mat,categories);
title('Confusion Matrix');

Accuracy = sum(diag(Conf_Mat))/sum(sum(Conf_Mat))
