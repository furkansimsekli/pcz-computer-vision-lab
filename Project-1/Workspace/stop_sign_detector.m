cifar10Data = './datasets';

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';

helperCIFAR10Data.download(url,cifar10Data);

[trainingImages,trainingLabels,validationImages, validationLabels, testImages,testLabels] = helperCIFAR10Data.load(cifar10Data);

% size(trainingImages)

numImageCategories = 10;
% categories(trainingLabels)

% Create the image input layer for 32x32x3 CIFAR-10 images.
[height,width,numChannels, ~] = size(trainingImages);

imageSize = [height width numChannels];

layers = [
    imageInputLayer(imageSize)
    convolution2dLayer(3, 16, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 128, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 256, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(10)
    softmaxLayer()
    classificationLayer()
];

% Set the network training options
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 128, ...
    'ValidationData', {validationImages, validationLabels}, ...
    'ValidationPatience', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);
% load('models/cifar10Netv2.mat');

% % Extract the first convolutional layer weights
% w = cifar10Net.Layers(2).Weights;
% 
% % rescale the weights to the range [0, 1] for better visualization
% w = rescale(w);
% 
% figure
% montage(w)

% Run the network on the test set.
YTest = classify(cifar10Net, testImages);

% Calculate the accuracy.
accuracy = sum(YTest == testLabels)/numel(testLabels);
disp(['cifar10Net Accuracy: ', num2str(accuracy)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the ground truth data
data = load('stopSignsAndCars.mat', 'stopSignsAndCars');
stopSignsAndCars = data.stopSignsAndCars;

% Update the path to the image files to match the local file system
visiondata = fullfile(toolboxdir('vision'),'visiondata');
stopSignsAndCars.imageFilename = fullfile(visiondata, stopSignsAndCars.imageFilename);

% % Display a summary of the ground truth data
% summary(stopSignsAndCars);

% Only keep the image file names and the stop sign ROI labels
stopSignsTrain = stopSignsAndCars(:, {'imageFilename','stopSign'});

% Apparently trainRCNNObjectDetector() doesn't support validation dataset.
% If in the future, developers get together their mind and stop being lazy
% you should pass stopSignsValidation as ValidationData in options.
% I hate MathWorks.
load('datasets/stop-signs/stopSignsValidation.mat');

% Set training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 100, ...
    'MaxEpochs', 100, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% Train an R-CNN object detector. This will take several minutes.    
rcnn = trainRCNNObjectDetector(stopSignsTrain, cifar10Net, options, ...
'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1]);
% load('models/rcnnv2.mat')

% Testing
load('datasets/stop-signs/stopSignsTest.mat');

total_iou = 0;

for i=1:size(stopSignsTest, 1)
    image = imread(stopSignsTest.imageFilename{i});
    [bboxes, score, label] = detect(rcnn, image, 'MiniBatchSize', 128);
    [score, idx] = max(score);
    bbox = bboxes(idx, :);

    gT = stopSignsTest.stopSign{i};
    iou = calculateIoU(gT, bbox);
    total_iou = total_iou + iou;

    % % Uncommand below lines, if you want to see the detection in action!
    % annotation = sprintf('%s: (Confidence = %f)', label(idx), score);
    % outputImage = insertObjectAnnotation(image, 'rectangle', bbox, annotation);
    % figure
    % imshow(outputImage);
end

test_accuracy = total_iou / size(stopSignsTest, 1);
disp(['RCNN Accuracy: ', num2str(test_accuracy)]);

function iou = calculateIoU(boxA, boxB)
    if isempty(boxA) || isempty(boxB)
        iou = 0;
        return;
    end

    xA = max(boxA(1), boxB(1));
    yA = max(boxA(2), boxB(2));
    xB = min(boxA(1) + boxA(3), boxB(1) + boxB(3));
    yB = min(boxA(2) + boxA(4), boxB(2) + boxB(4));

    intersectionArea = max(0, xB - xA + 1) * max(0, yB - yA + 1);
    boxAArea = boxA(3) * boxA(4);
    boxBArea = boxB(3) * boxB(4);
    unionArea = boxAArea + boxBArea - intersectionArea;
    iou = intersectionArea / unionArea;
end
