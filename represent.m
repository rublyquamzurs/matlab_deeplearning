function [embedding]=represent(X,Y,Loc,K,L,varargin)
% paras.num_classes = 13;
% paras.embedding_size = 1024;
% paras.max_epoch = 500;
% paras.learning_rate = 0.001;


inputs = X';
labels = Y';
locations = Loc';
transform_matrix = L;

layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm',...
    'MaxEpochs',3, ...
    'ValidationData',valDigitData,...
    'ValidationFrequency',30,...
    'Verbose',false,...
    'Plots','training-progress');

net = trainNetwork(trainDigitData,layers,options);

predictedLabels = classify(net,valDigitData);
valLabels = valDigitData.Labels;

accuracy = sum(predictedLabels == valLabels)/numel(valLabels)

end


function select_triplets(embeddings, paras)


end

function train(net, layers, paras, traindata)

embeddings = net(traindata.data, layers, paras);

triplets = select_triplets(embeddings, paras)

for i = 0:minibatch
    loss = triplet_loss()
    applygradiant(layers)

end

end


function evaluate(net, layers, paras, validationdata)

end