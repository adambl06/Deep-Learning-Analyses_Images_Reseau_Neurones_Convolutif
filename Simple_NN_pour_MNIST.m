
close all

% Avant de lancer ce programme, il faut lancer MNIST_database.m

% Visualiser un sous-ensemble d'images de manière aléatoire
%warning off images:imshow:magnificationMustBeFitForDockedFigure
perm = randperm(numel(labelsTrain), 25);
subset = imgDataTrain(:,:,1,perm);
montage(subset)

% 4x1 Layer array with layers:
%
%    1   ''   Image Input             28x28x1 images with 'zerocenter' normalization
%    2   ''   Fully Connected         10 fully connected layer
%    3   ''   Softmax                 softmax
%    4   ''   Classification Output   crossentropyex
     
layers = [  imageInputLayer([28 28 1],'Name','input')
            fullyConnectedLayer(10,'Name','fullyconnected')
            softmaxLayer('Name','softmax')
            classificationLayer('Name','classification')   ];

% Set training options and train the network
miniBatchSize = 500;

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.1, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
     'MiniBatchSize', miniBatchSize,...
     'ValidationData',{imgDataTest,labelsTest},...
     'ValidationFrequency',40,...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imgDataTrain, labelsTrain, layers, options);


        