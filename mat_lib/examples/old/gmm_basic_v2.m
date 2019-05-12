function [model, y] = gmm_basic()
clear all
clc

n_samples = 100;

%% generate random data
mu_r = [1, 3];
sigma_r = [1.5, 1.5];
w_r = [1]; % has to sum to one

x = mvnrnd(mu_r, diag(sigma_r), n_samples);

plot(x(:,1),x(:,2),'x')

w_r = permute(w_r, [1 3 2] );

n_dim = 2;
n_kernels = 1;

%% define model
%model= cell();
model.layers{1}= GmmLayer(n_dim, n_kernels);

model.layers{1}.params.mu = rand(1, n_dim, n_kernels);
model.layers{1}.params.sigma = rand(1, n_dim, n_kernels);
model.layers{1}.params.w = 1; %rand(1, 1, n_kernels);

model.layers{1}.const.x = x;

model.layers{2}= NegLogLoss;

model.n_layers= length(model.layers);

model = get_num_params(model);

%% compute forward pass
[model, y] = forward(model);

%% compute backward pass
[model, y] = forward(model);

end



function loss(w)
    model = vect2params(model, w)
    
end

function dloss_dw(w)
    
    

end

% TODO: convert model into a class. Each layer must have an option for
% inputs, parameters and constants. 
% layer class should provide a forward and backward
% TODO: add number of input/outputs to each layer

function [model, y] = forward(model)
        
    y = cell(model.n_layers,1);
    layers = model.layers;
    
    if ~isempty(model.layers{1}.params) && ~isempty(model.layers{1}.const)
        params = struct2cell(model.layers{1}.params);
        const = struct2cell(model.layers{1}.const);
        
        y{1} = model.layers{1}.forward(params{:}, const{:});
        
    elseif ~isempty(model.layers{1}.params)
        params = struct2cell(model.layers{1}.params);
        
        y{1} = model.layers{1}.forward(params{:});
    end
        
    for l= 2:model.n_layers % change to number of layers
        if ~isempty(model.layers{l}.params) && ~isempty(model.layers{l}.const)
            params = struct2cell(model.layers{l}.params);
            const = struct2cell(model.layers{l}.const);

            y{l} = model.layers{l}.forward(y{l-1}, params{:}, const{:});

        elseif ~isempty(model.layers{l}.params)
            params = struct2cell(model.layers{l}.params);

            y{l} = model.layers{l}.forward(y{l-1}, params{:});
            
        else
            y{l} = model.layers{l}.forward(y{l-1});
        end
    
    end
    
end

function de_dw = backward(model)
    de_dw = zeros(model.n_params_cum(end), 1);
    idx=1;
    
    
    de_dinput = model.layers{end}.backward_inputs(1);
    
    for l= model.n_layers-1:-1:1
        % parameters gradient calculation
        if ~isempty(model.layers{l}.params)
            model.layers{l}.backward_params(de_dinput);
            
            dparams = struct2cell(model.layers{l}.dparams);
            dparams = cell2mat(cellfun(@(x) reshape(x,[],1), dparams, 'UniformOutput', false));
            
            de_dw(idx:model.n_params_cum(l)) = dparams;
            idx = model.n_params_cum(l);
        end
        % inputs gradient calculation
        if model.layers{l}.n_inputs ~= 0
            de_dinput = model.layers{end}.backward_inputs(de_dinput);
        end
            
    end
    
end

function w = params2vect(model)
        
    w = zeros(model.n_params_cum(end), 1);
    idx=1;
    for l= 1:model.n_layers 
        if ~isempty(model.layers{l}.params)
            params = struct2cell(model.layers{l}.params);
            params = cell2mat(cellfun(@(x) reshape(x,[],1), params, 'UniformOutput', false));
            
            w(idx:model.n_params_cum(l)) = params;
            idx = model.n_params_cum(l);
        end
    end
    
end

function model = vect2params(model, w)
    idx= 1;
    for l= 1:model.n_layers 
        if ~isempty(model.layers{l}.params)
            fields = fieldnames(model.layers{l}.params);
            
            for i = 1:numel(fields)
                param_i = model.layers{l}.params.(fields{i});
                model.layers{l}.params.(fields{i}) = reshape(w(idx:idx+numel(param_i)-1), size(param_i));
                idx = idx + numel(param_i);
            end
        end
    end
    
end

function model = get_num_params(model)
    model.n_params= zeros(model.n_layers, 1);
    
    for l= 1:model.n_layers 
        if ~isempty(model.layers{l}.params)
            params = struct2cell(model.layers{l}.params);
            model.n_params(l) = sum(cellfun(@numel, params));
        end
    end
    model.n_params_cum = cumsum(model.n_params);
end
