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


end