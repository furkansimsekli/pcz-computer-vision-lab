% % Download and load CIFAR10 dataset with the help of helper functions
% cifar10Data = "./datasets";
% url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
% helperCIFAR10Data.download(url, cifar10Data);
% [trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load(cifar10Data);
% 
% % % Size and categories
% % disp("Size:")
% % disp(size(trainingImages))
% % disp("Categories:")
% % disp(categories(trainingLabels));
% 
% % % Show sample images from dataset
% % figure
% % thumbnails = trainingImages(:,:,:,1:100);
% % montage(thumbnails)
% 
% % Create the image input layer for 32x32x3 CIFAR-10 images.
% [height,width,numChannels, ~] = size(trainingImages);
% imageSize = [height width numChannels];
% inputLayer = imageInputLayer(imageSize);
% 
% layers = [
%     imageInputLayer(imageSize)
%     convolution2dLayer(3, 16, 'Padding', 2)
%     reluLayer()
%     maxPooling2dLayer(2, 'Stride', 2)
%     convolution2dLayer(3, 32, 'Padding', 2)
%     reluLayer()
%     maxPooling2dLayer(2, 'Stride', 2)
%     convolution2dLayer(3, 64, 'Padding', 2)
%     reluLayer()
%     maxPooling2dLayer(2, 'Stride', 2)
%     convolution2dLayer(3, 128, 'Padding', 2)
%     reluLayer()
%     maxPooling2dLayer(2, 'Stride', 2)
%     convolution2dLayer(3, 256, 'Padding', 2)
%     reluLayer()
%     maxPooling2dLayer(2, 'Stride', 2)
%     fullyConnectedLayer(10)
%     softmaxLayer()
%     classificationLayer()
% ];
% 
% % Set the network training options
% cnnTrainingOptions = trainingOptions('sgdm', ...
%     'Momentum', 0.9, ...
%     'InitialLearnRate', 0.001, ...
%     'LearnRateSchedule', 'piecewise', ...
%     'LearnRateDropFactor', 0.1, ...
%     'LearnRateDropPeriod', 8, ...
%     'L2Regularization', 0.004, ...
%     'MaxEpochs', 40, ...
%     'MiniBatchSize', 128, ...
%     'Plots','training-progress', ...
%     'Verbose', false);
% 
% % mysupercnn = trainNetwork(trainingImages, trainingLabels, layers, cnnTrainingOptions);
load('models/mysupercnn1.mat');
% 
% % % Extract the first convolutional layer weights
% % w = mysupercnn.Layers(2).Weights;
% % 
% % % rescale the weights to the range [0, 1] for better visualization
% % w = rescale(w);
% % 
% % figure
% % montage(w)
% 
% % Run the network on the test set.
% YTest = classify(mysupercnn, testImages);
% 
% % Calculate the accuracy.
% accuracy = sum(YTest == testLabels)/numel(testLabels);
% disp(["Accuracy: ", num2str(accuracy)])


% Load the ground truth data
data = load('stopSignsAndCars.mat', 'stopSignsAndCars');
stopSignsAndCars = data.stopSignsAndCars;

% Update the path to the image files to match the local file system
visiondata = fullfile(toolboxdir('vision'),'visiondata');
stopSignsAndCars.imageFilename = fullfile(visiondata, stopSignsAndCars.imageFilename);


% Display a summary of the ground truth data
summary(stopSignsAndCars)

% Only keep the image file names and the stop sign ROI labels
stopSigns = stopSignsAndCars(:, {'imageFilename','stopSign'});

% Display one training image and the ground truth bounding boxes
% I = imread(stopSigns.imageFilename{1});
% I = insertObjectAnnotation(I,'Rectangle',stopSigns.stopSign{1},'stop sign','LineWidth',8);

% figure
% imshow(I)

% Set training options
rcnnTrainingOptions = trainingOptions('sgdm', ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 100, ...
    'MaxEpochs', 100, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train an R-CNN object detector.
% rcnn = trainRCNNObjectDetector(stopSigns, mysupercnn, rcnnTrainingOptions, ...
%     'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1]);

% load("rcnn2.mat")

load("models/rcnn.mat")

% Read test image
testImage = imread('datasets/stop-signs/test/image001.png');

% Detect stop signs
[bboxes,score,label] = detect(rcnn,testImage,'MiniBatchSize',128);

% Display the detection results
[score, idx] = max(score);

bbox = bboxes(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

outputImage = insertObjectAnnotation(testImage, 'rectangle', bbox, annotation);

figure
imshow(outputImage);
