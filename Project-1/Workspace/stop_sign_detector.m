% Download and load CIFAR-10 dataset. Please note that, if dataset already
% exists in the given path, it doesn't download all over again.
cifar10Data = './datasets';
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
helperCIFAR10Data.download(url,cifar10Data);
[trainX, trainY, validationX, validationY, testX, testY] = helperCIFAR10Data.load(cifar10Data);

disp(['Train Dataset Size: ', num2str(size(trainX))]);
disp(['Validation Dataset Size: ', num2str(size(validationX))]);
disp(['Test Dataset Size: ', num2str(size(testX))]);
disp([categories(trainY)]);

[height,width,numChannels, ~] = size(trainX);
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

opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 128, ...
    'ValidationData', {validationImages, validationLabels}, ...
    'ValidationPatience', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);

% % If you want to load pretrained model, uncomment the line below, and
% % comment the trainNetwork line above.
% load('models/cifar10Net.mat');

% Testing
YTest = classify(cifar10Net, testImages);
cifar10NetAccuracy = sum(YTest == testLabels)/numel(testLabels);
disp(['cifar10Net Accuracy: ', num2str(cifar10NetAccuracy)]);

% -------------------------------------------------------------------------
% In the other half of the code, I transfered this classifier network to
% RCNN. New network will be trained for object detection, to be more
% specific will be trained for stop sign detection. Training dataset
% belongs to MATLAB Computer Vision Toolbox.
% -------------------------------------------------------------------------

% Load the ground truth training data
data = load('stopSignsAndCars.mat', 'stopSignsAndCars');
stopSignsAndCars = data.stopSignsAndCars;

% Update the path to the image files to match the local file system
visiondata = fullfile(toolboxdir('vision'),'visiondata');
stopSignsAndCars.imageFilename = fullfile(visiondata, stopSignsAndCars.imageFilename);

summary(stopSignsAndCars);

% Only keep the image file names and the stop sign ROI labels
stopSignsTrain = stopSignsAndCars(:, {'imageFilename','stopSign'});

% Apparently trainRCNNObjectDetector() doesn't support validation dataset.
% If in the future, developers get together their mind and stop being lazy
% you should pass stopSignsValidation as ValidationData in options.
% I hate MathWorks.
load('datasets/stop-signs/stopSignsValidation.mat');

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 100, ...
    'MaxEpochs', 1, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

rcnn = trainRCNNObjectDetector(stopSignsTrain, cifar10Net, options, ...
'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1]);

% % If you want to load pretrained model, uncomment the line below, and
% % comment the trainNetwork line above.
% load('models/rcnn.mat')

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

rcnnAccuracy = total_iou / size(stopSignsTest, 1);
disp(['RCNN Accuracy: ', num2str(rcnnAccuracy)]);

function iou = calculateIoU(boxA, boxB)
    % Calculates Intersection of Union from given two boxes.

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
