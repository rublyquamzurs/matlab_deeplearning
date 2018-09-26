function [oup]=conventional(inp, kes, stride, dilation, padding)
% input, kernel size, stride, biases, dilation, padding 

assert(nargin == 5, 'Lack of parameters')

% N, D, INCHANNEL
[datan, datad, inchannel1] = size(inp);

% W, INCHANNEL, OUTCHANNEL
[kesize, inchannel2, outchannel] = size(kes);
assert(inchannel1 == inchannel2, 'Channel error')

% real_size = 1 + dilation * (kesize - 1)

if padding ~= 'VALID'
    extenal = floor((1 + dilation * (kesize - 1)) / 2);
    new_width = ceil(datad / 2);
    pads = ones(datan, extenal, inchannel1) * padding;
    temp = [pads, inp, pads];
    inp = temp;
    oup = zeros(datan, new_width, outchannel);
end

[datan, datad, ~] = size(inp);
walk = 1;
while walk + dilation*kesize <= datad
    multi_1 = inp(:, walk:dilation:walk+dilation*(kesize - 1), :);
    for i = 1:datan
        for j = 1:outchannel
            temp = multi_1(i, :, :) .* permute(kes(:, :, j), [2, 3, 1]);
            pixel = sum(sum(sum(temp)));
            oup(i, walk, j) = pixel;
        end
    end
    walk = walk + stride;
end

function [oup]=conventional(inp, kes, stride, biases, dilation, padding)
% input, kernel size, stride, biases, dilation, padding 
assert(nargin ~= 6, 'Lack of parameters')

embedding = net(X);
a = convn
trainsferLayer = net.Layers
a = [4 5; 2 6];
b = gpuArray(a);
c = b^3;
ans = gather(c);
net = newff()
disp(ans)
end

end