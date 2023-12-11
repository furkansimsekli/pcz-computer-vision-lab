imds = imageDatastore('Baza2\',...
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
    fullyConnectedLayer(2)
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
accuracy = sum(YPred == YTest)/numel(YTest);

% Show random images
numImages = 16;
randIndices = randperm(length(imdsTest.Files), numImages);
display(randIndices);

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