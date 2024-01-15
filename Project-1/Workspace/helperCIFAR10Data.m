% This is helper class to download and import the CIFAR-10 dataset. The
% dataset is downloaded from:
%
%  https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz
%
% References
% ----------
% Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of
% features from tiny images." (2009).
%
%
%
%
% 2024.01.15
% furkansimsekli
%   Split test dataset to half for validation.
%

classdef helperCIFAR10Data
    
    methods(Static)
        
        %------------------------------------------------------------------
        function download(url, destination)
            if nargin == 1
                url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
            end        
            
            unpackedData = fullfile(destination, 'cifar-10-batches-mat');
            if ~exist(unpackedData, 'dir')
                fprintf('Downloading CIFAR-10 dataset...');     
                untar(url, destination); 
                fprintf('done.\n\n');
            end
        end
        
        %------------------------------------------------------------------
        % Return CIFAR-10 Training and Test data.
        function [XTrain, TTrain, XValidation, TValidation, XTest, TTest] = load(dataLocation)         
            
            location = fullfile(dataLocation, 'cifar-10-batches-mat');
            
            [XTrain1, TTrain1] = loadBatchAsFourDimensionalArray(location, 'data_batch_1.mat');
            [XTrain2, TTrain2] = loadBatchAsFourDimensionalArray(location, 'data_batch_2.mat');
            [XTrain3, TTrain3] = loadBatchAsFourDimensionalArray(location, 'data_batch_3.mat');
            [XTrain4, TTrain4] = loadBatchAsFourDimensionalArray(location, 'data_batch_4.mat');
            [XTrain5, TTrain5] = loadBatchAsFourDimensionalArray(location, 'data_batch_5.mat');
            
            XTrain = cat(4, XTrain1, XTrain2, XTrain3, XTrain4, XTrain5);
            TTrain = [TTrain1; TTrain2; TTrain3; TTrain4; TTrain5];
            
            % Split the testing data into validation and testing
            [XTest, TTest] = loadBatchAsFourDimensionalArray(location, 'test_batch.mat');
            numTestSamples = size(XTest, 4);
            
            % Splitting ratio (adjust as needed)
            validationRatio = 0.5;
            numValidationSamples = round(validationRatio * numTestSamples);
            
            XValidation = XTest(:, :, :, 1:numValidationSamples);
            TValidation = TTest(1:numValidationSamples);
            
            XTest = XTest(:, :, :, numValidationSamples+1:end);
            TTest = TTest(numValidationSamples+1:end);
        end
    end
end

function [XBatch, TBatch] = loadBatchAsFourDimensionalArray(location, batchFileName)
load(fullfile(location,batchFileName));
XBatch = data';
XBatch = reshape(XBatch, 32,32,3,[]);
XBatch = permute(XBatch, [2 1 3 4]);
TBatch = convertLabelsToCategorical(location, labels);
end

function categoricalLabels = convertLabelsToCategorical(location, integerLabels)
load(fullfile(location,'batches.meta.mat'));
categoricalLabels = categorical(integerLabels, 0:9, label_names);
end
