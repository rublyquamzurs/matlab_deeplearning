function [embedding]=AlexNet_1d(inputs, varargin)
% Default parameter.
if nargin < 4
    layers = inputs;
end

dropout = 0.5;
num_classes = 1024;
embedding_size = 1024;

embedding = inference(inputs, dropout, num_classes, embedding_size, layers);
end


function outp=inference(input, dropout, num_classes, embedding_size, layers)
% 
conv1 = conventional(input, layers.weight.wc1, 2, 1, 0);

outp
end


function outp=random_normal(inp, mu, sigma)
%
outp = normrnd(inp, mu, sigma);

end