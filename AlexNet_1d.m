function [embedding]=AlexNet_1d(inputs, varargin)
% Default parameter.

dropout = 0.5;
num_classes = 1024;
embedding_size = 1024;

embedding = inference(inputs, dropout, num_classes, embedding_size, varargin);
end


function outp=inference(input, dropout, num_classes, embedding_size, varargin)
% 
weight = struct{'wc1', [3, 1, 10], 'wc2', [3, 10, 20], 'wc3', [9060, 1024], 'ce', [1024, num_classes], 'rp', [1024, embedding_size]};
biases = struct{'bc1', 10, 'bc2', 20, 'bc3', 1024, 'ce', num_classes, 'rp', embedding_size};
conv1 = conventional(input, random_normal)
outp
end


function outp=random_normal(inp, mu, sigma)
%
outp = normrnd(inp, mu, sigma);

end