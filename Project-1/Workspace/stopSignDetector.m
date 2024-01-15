% Download and load CIFAR-10 dataset. Please note that, if dataset already
% exists in the given path, it doesn't download all over again.
cifar10Data = './datasets';
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
helperCIFAR10Data.download(url,cifar10Data);
[trainX, trainY, validationX, validationY, testX, testY] = helperCIFAR10Data.load(cifar10Data);

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
    'MaxEpochs', 40, ...
    'MiniBatchSize', 128, ...
    'ValidationData', {validationX, validationY}, ...
    'ValidationPatience', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

cifar10Net = trainNetwork(trainX, trainY, layers, opts);

% % If you want to load pretrained model, uncomment the line below, and
% % comment the trainNetwork line above.
% load('models/cifar10Net.mat');

% Testing
predY = classify(cifar10Net, testX);
cifar10NetAccuracy = sum(predY == testY)/numel(testY);
disp(['cifar10Net Accuracy: ', num2str(cifar10NetAccuracy)]);

% Plot example images with predictions and true labels
numExamples = 6;
figure;

for i = 1:numExamples
    subplot(ceil(numExamples / 2), 2, i);
    idx = randi(size(testX, 4));
    imshow(testX(:,:,:,idx));
    trueLabel = char(testY(idx));
    predictedLabel = char(predY(idx));
    title(['True: ', trueLabel, ' | Predicted: ', predictedLabel]);
end

% Confusion matrix
C = confusionmat(testY, predY);

figure;
confusionChart = confusionchart(C);

% TP, TN, FP, FN
numClasses = size(C, 1);
TP = zeros(1, numClasses);
TN = zeros(1, numClasses);
FP = zeros(1, numClasses);
FN = zeros(1, numClasses);

for i = 1:numClasses
    TP(i) = C(i, i);
    FP(i) = sum(C(:, i)) - TP(i);
    FN(i) = sum(C(i, :)) - TP(i);
    TN(i) = sum(C(:)) - TP(i) - FP(i) - FN(i);
end

accuracy = zeros(1, numClasses);
recall = zeros(1, numClasses);
specificity = zeros(1, numClasses);
precision = zeros(1, numClasses);
f1 = zeros(1, numClasses);
mcc = zeros(1, numClasses);

for i = 1:numClasses
    accuracy(i) = (TP(i) + TN(i)) / (TP(i) + TN(i) + FP(i) + FN(i));
    recall(i) = TP(i) / (TP(i) + FN(i));
    specificity(i) = TN(i) / (TN(i) + FP(i));
    precision(i) = TP(i) / (TP(i) + FP(i));
    f1(i) = (2 * TP(i)) / (2 * TP(i) + FP(i) + FN(i));
    mcc(i) = (TP(i) * TN(i) - FP(i) * FN(i)) / ...
        (sqrt((TP(i) + FP(i)) * (TP(i) + FN(i)) * ...
        (TN(i) + FP(i)) * (TN(i) + FN(i))));
end

% Create a table to display metrics
metricsTable = table(accuracy', recall', specificity', precision', f1', mcc', ...
    'VariableNames', {'Accuracy', 'Recall', 'Specificity', 'Precision', 'F1-score', 'MCC'}, ...
    'RowNames', arrayfun(@(x) sprintf('Class %d', x), 1:numClasses, 'UniformOutput', false));

disp('Metrics Table:');
disp(metricsTable);




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

% Plot example images with rectangles for the training set
numExamplesTrain = 6;
figure;

for i = 1:numExamplesTrain
    subplot(ceil(numExamplesTrain/2), 2, i);
    idx = randi(size(stopSignsTrain, 1));
    image = imread(stopSignsTrain.imageFilename{idx});
    groundTruth = stopSignsTrain.stopSign{idx};
    imageWithGroundTruth = insertShape(image, 'Rectangle', groundTruth, ...
        'Color', 'red', 'LineWidth', 4);
    imshow(imageWithGroundTruth);
end

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
    'MaxEpochs', 100, ...
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

% Plot example images with rectangles for RCNN
numExamples = 6;
figure;

for i = 1:numExamples
    subplot(ceil(numExamples/2), 2, i);
    idx = randi(size(stopSignsTest, 1));
    image = imread(stopSignsTest.imageFilename{idx});
    groundTruth = stopSignsTest.stopSign{idx};
    imageWithGroundTruth = insertShape(image, 'Rectangle',groundTruth, ...
        'Color', 'red', 'LineWidth', 4);
    [bboxes, ~, ~] = detect(rcnn, image, 'MiniBatchSize', 128);
    
    if ~isempty(bboxes)
        imageWithAnnotations = insertShape(imageWithGroundTruth, ...
            'Rectangle', bboxes(1, :), 'Color', 'green', 'LineWidth', 4);
        imshow(imageWithAnnotations);
    else
        imshow(imageWithGroundTruth);
    end
end

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
