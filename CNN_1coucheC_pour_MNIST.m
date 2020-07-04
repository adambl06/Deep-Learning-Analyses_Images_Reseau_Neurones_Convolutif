
close all

% Avant de lancer ce programme, il faut lancer MNIST_database.m

% Visualiser un sous-ensemble d'images de manière aléatoire
%warning off images:imshow:magnificationMustBeFitForDockedFigure
perm = randperm(numel(labelsTrain), 25);
subset = imgDataTrain(:,:,1,perm);
montage(subset)

% 8x1 Layer array with layers:
%
%    1   ''   Image Input             28x28x1 images with 'zerocenter' normalization
%    2   ''   Convolution             20 5x5 convolutions with stride [1  1] and padding [0  0  0  0]
%    3   ''   BatchNormalisation      Normalisation des données d'entrée
%                                       par batch et pour chaque canal de convolution
%    4   ''   ReLU                    ReLU
%    5   ''   Max Pooling             2x2 max pooling with stride [2  2] and padding [0  0  0  0]
%    6   ''   Fully Connected         10 fully connected layer
%    7   ''   Softmax                 softmax
%    8   ''   Classification Output   crossentropyex
     
layers = [  imageInputLayer([28 28 1],'Name','input')
    
            convolution2dLayer(3,8,'Padding','same','Name','conv1')
            batchNormalizationLayer
            reluLayer('Name','relu')
            maxPooling2dLayer(2, 'Stride', 2,'Name','maxpooling')
  
            fullyConnectedLayer(10,'Name','fullyconnected')
            softmaxLayer('Name','softmax')
            classificationLayer('Name','classification')   ];
       
        
% Visualisation du réseau sous forme de graphes avec analyse détaillée de
% chaque couche
%figure
analyzeNetwork(layers)
% on peut aussi juste tracer le graphe du réseau (il faut que chaque couche
% ait un nom
% figure
% plot(layerGraph(layers))


% Set training options and train the network
miniBatchSize = 500;

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
     'MiniBatchSize', miniBatchSize,...
     'ValidationData',{imgDataTest,labelsTest},...
     'ValidationFrequency',40,...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imgDataTrain, labelsTrain, layers, options);


        