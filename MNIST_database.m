clear all
close all

% Pr�paration de la base d'apprentissage, de la base de test, des labels apprentissage et des labels tests

% Chargement de la base de donn�es MNIST
% 60 000 images de 28x28 pixels, chaque pixel cod� sur un octet
% 50 000 images dans la base d'apprentissage
% 10 000 images dans la base de test


[imgDataTrain, labelsTrain, imgDataTest, labelsTest] = prepareData;