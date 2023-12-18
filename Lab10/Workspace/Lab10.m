imds = imageDatastore('Baza3\',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7);

lenTrain = length(imdsTrain.Labels);
lenTest = length(imdsTest.Labels);

layers = [...
    imageInputLayer([28 28 1])
    convolution2dLayer(5, 20)
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer];

MiniBatchSize = 200;
options = trainingOptions('sgdm',...
    'MaxEpochs',20,...
    'MiniBatchSize',MiniBatchSize,...
    'InitialLearnRate',1e-4,...
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'Plots','training-progress');

net = trainNetwork(imdsTrain, layers, options);

[YPred, probs] = classify(net, imdsTest);
YTest = imdsTest.Labels;

% Confusion matrix
C = confusionmat(YTest, YPred);
disp('Confusion Matrix:');
disp(C);

% Confusion chart
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

disp(['TP for classes: ', num2str(TP)]);
disp(['TN for classes: ', num2str(TN)]);
disp(['FP for classes: ', num2str(FP)]);
disp(['FN for classes: ', num2str(FN)]);

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

disp(['Accuracy for classes: ', num2str(accuracy)]);
disp(['Recall for classes: ', num2str(recall)]);
disp(['Specificity for classes: ', num2str(specificity)]);
disp(['Precision for classes: ', num2str(precision)]);
disp(['F1-score for classes: ', num2str(f1)]);
disp(['MCC for classes: ', num2str(mcc)]);

% Show random images
numImages = 12;
randIndices = randperm(length(imdsTest.Files), numImages);

figure;
for i = 1:numImages
    img = readimage(imdsTest, randIndices(i));
    actualLabel = char(YTest(randIndices(i)));
    predictedLabel = char(YPred(randIndices(i)));
    subplot(4,4,i);
    imshow(img);
    title(['Actual: ' actualLabel,...
        'Predicted: ' predictedLabel]);
end

sgtitle('')