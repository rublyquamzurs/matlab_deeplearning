<<<<<<< HEAD
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

=======
function [embedding]=represent(X,Y,Loc,K,L,varargin)

options.K=K;                 % number of nearest neighbours
options.tol = 1e-5;         % tolerance for convergence
options.verbose=true;   % screen output
options.depth=4;           % tree depth
options.ntrees=200;      % number of boosted trees
options.lr=1e-3;             % learning rate for gradient boosting
options.no_potential_impo=inf;
options.buildlayer = @buildlayer_sqrimpurity_openmp_multi;
options.XVAL=[];
options.YVAL=[];
options.LVAL=[];
options.XFULL=[];
options.YFULL=[];
options.LFULL=[];
options.valSetT = 1;
options.Ki = 50;
options=extractpars(varargin,options);

if ~isempty(options.XFULL)
    pred_full = options.XFULL;
end
outp = @normrnd(inp, mu, sigma);
weight = struct{'wc1', [3, 1, 10], 'wc2', [3, 10, 20], 'wc3', [9060, 1024], 'ce', [1024, num_classes], 'rp', [1024, embedding_size]};
biases = struct{'bc1', 10, 'bc2', 20, 'bc3', 1024, 'ce', num_classes, 'rp', embedding_size};
layers.wc1 = normrnd(0, )
snapshot
initwb

convolution2dLayer


net = AlexNet_1d();


>>>>>>> c37c00ce5982e3ae683fa806b6555b5e823fdcb6
end