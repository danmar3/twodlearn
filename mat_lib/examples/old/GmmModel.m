% GmmModel: gaussian mixture model implementation
% 
% Created by: Daniel L. Marino (marinodl@vcu.edu)
%
%

classdef GmmModel < handle
    % The GMM model implemented here is for an independent set of
    % parameters. i.e. mu, sigma and w are supposed to be parameters, not
    % an output from, e.g. a neural network.
    %
    % Note: all vectors are asumed to be row vectors
    % 
    properties
        % n_dim: number of dimentions of the model
        % n_kernels: number of kernels
        % sigma_inv: stores the value of the inverse of sigma, which is
        %            encountered several times while computing the backward
        %            pass 
        % sigma_inv_x: stores the computation of sigma_inv*(x-mu) which is
        %              encountered several times 
        % gaussians: stores the computation of each one of the gaussians
        %
        
        n_dim 
        n_kernels 
        
        sigma_inv
        sigma_inv_x_mu  
        w
        sigma
        mu
        
        gaussians
        y
        
    end
    methods
        function obj = GmmModel(n_dim_in, n_kernels_in)
            if nargin == 2
                obj.n_dim = n_dim_in;
                obj.n_kernels = n_kernels_in;
                
                obj.sigma_inv = zeros([  obj.n_dim, obj.n_dim, obj.n_kernels]);
            end       
        end
        
        function Tout = batch_inv(Tin)
            if ismatrix(Tin)
                Tout = inv(Tin);
            elseif ndims(Tin)==3
                Tout = zeros(size(Tin));
                for k=1:size(Tin,3)
                    Tout(:,:,k) = inv(Tin(:,:,k));
                end
            end
        end
        
        function y = forward(obj, mu, sigma, w, x)
            % forward: forward pass, computes y=pi'*N(x,mu,sigma)
            %    - x: 2d tensor containing the inputs. Each row correspond
            %         to a different sample
            %    - mu: 3d tensor containing the means for the gaussians.
            %          the "depth" dimention (3rd) is used to index the
            %          gaussians.    [samples, dim ,kernel_id]
            %    - sigma: 3d tensor containing the covariance matrix of the
            %             gaussians. [samples, dim ,kernel_id]
            %    - w: vector in form of a 3d tensor containing the weights
            %         for each one of the gaussians. [samples, 1 ,kernel_id]
            % returns:
            %    - y: probability p(x|mu, sigma, w) given by the gmm model.
            %         [samples, 1]
            
            % check if sigma is given as diagonal matrices 
            obj.mu = mu;
            obj.sigma = sigma;
            
            if size(sigma,2)==obj.n_dim 
                y = forward_diagonal(obj, mu, sigma, w, x);
                return
            end
            
            % TODO: from here on, it is incorrect, the code should be
            % changed in a way that the first dimention represents
            % different inputs
            
            % make sure the sigma matrices are simetric:
            sigma = 0.5*(sigma + permute(sigma, [2,1]));
            
            % calculate inverse and determinant for each sigma of each
            % kernel: 
            norm_const= zeros(1, 1, obj.n_kernels); % normalization constant for each gaussian
            for k=1:obj.n_kernels
                obj.sigma_inv(:,:,k) = inv(sigma(:,:,k));
                norm_const(1,1,k) = 1/sqrt( ((2*pi)^obj.n_kernels) * det(sigma(:,:,k)) );
            end
            % compute (x-mu)
            x_mu = bsxfun(@minus, x, mu);
            
            % compute sigma_inv*(x-mu), Note: it is actually calculated as
            % (x-mu)*sigma_inv given that x-mu is oredered by rows.
            % remember also that sigma_inv is symetric
            for k=1:obj.n_kernels
                obj.sigma_inv_x_mu(:,:,k) = x_mu(:,:,k)*obj.sigma_inv(:,:,k);
            end
            
            % compute output
            obj.gaussians = bsxfun( @times, norm_const , exp( -0.5 * sum( x_mu .* obj.sigma_inv_x_mu ,2) ));
            
            obj.y = sum(bsxfun( @times, w , obj.gaussians), 3);
            y = obj.y;
            
        end
        
        function y = forward_diagonal(obj, mu, sigma, w, x)
            % forward: forward pass, computes y=pi'*N(x,mu,sigma), 
            %          optimized for diagonal sigma
            %    - x: 2d tensor containing the inputs. Each row correspond
            %         to a different sample
            %    - mu: 3d tensor containing the means for the gaussians.
            %          the "depth" dimention (3rd) is used to index the
            %          gaussians
            %    - sigma: 3d tensor containing the vectors (in row format)
            %             corresponding to the diagonal elements of the
            %             covariance matrix of the gaussians
            % 
            
            sigma= exp(sigma);  % modifier
            obj.sigma = sigma;
            
            if obj.n_kernels == 1
                w= 1;
            else
                w= exp(w); % modifier
                w= w/sum(w,3); % modifier
            end
            obj.w= w;
            
            % calculate inverse and determinant for each sigma of each
            % kernel: 
            norm_const= 1./sqrt( ((2*pi)^obj.n_kernels) * prod(sigma, 2) );
            obj.sigma_inv = 1./sigma;
            
            % compute (x-mu)
            x_mu = bsxfun(@minus, x, mu);
            
            % compute sigma_inv*(x-mu), Note: it is actually calculated as
            % (x-mu)*sigma_inv given that x-mu is oredered by rows.
            % remember also that sigma_inv is symetric
            obj.sigma_inv_x_mu = bsxfun( @times, x_mu, obj.sigma_inv);
            
            
            % compute output
            obj.gaussians = bsxfun( @times, norm_const , exp( -0.5 * sum( x_mu .* obj.sigma_inv_x_mu ,2) ));
            
            obj.y = sum(bsxfun( @times, w , obj.gaussians), 3);
            y = obj.y;
            
        end
        
        function de_dwin = de_dwin_eval(obj, de_dw)
            de_dwin = zeros(size(de_dw));
            
            if size(obj.w,1)==1 
                w = repmat(obj.w, [size(de_dw,1), 1, 1]); % simulate different input for different samples(this is very expensive)
            else
                w = obj.w;
            end
            for i=1:size(de_dw)
                w_i= squeeze(w(i,:,:)); % column vector
                de_dw_i= squeeze(de_dw(i,:,:)); % column vector
                
                de_dwin_i = (diag(w_i)-w_i*w_i')*de_dw_i;
                
                de_dwin(i,:,:)= de_dwin_i;
            end            
        end
        
        
        function [de_dmu, de_dsigma, de_dw] = backward(obj, de_dy)
            %   - e: represents the backpropagated Error, it also can be
            %        seen as the ordered derivative of the final function
            %        with respect the outputs of the Gmm model
            % return:
            %   - de_dsigma  
            %   - de_dw  
            %
            
            % w_gaussians: tensor of shape: [samples, 1 ,kernel]
            w_gaussians =  bsxfun( @times, obj.w, obj.gaussians); % 
            %% derivative with respect mu
            % dgmm_dmu: tensor of shape: [samples, dim ,kernel]
            dy_dmu = bsxfun( @times, w_gaussians , obj.sigma_inv_x_mu);  
            % de_dmu: tensor of shape: [samples, dim ,kernel]
            de_dmu = bsxfun( @times, de_dy, dy_dmu); 
            
            %% derivative with respect sigma
            % if diagonal matrix
            if size(obj.sigma_inv,2)==obj.n_dim 
                % dgmm_dmu: tensor of shape: [samples, dim ,kernel]
                dy_dsigma = bsxfun( @minus, obj.sigma_inv_x_mu.^2 , obj.sigma_inv);  
                dy_dsigma = 0.5 * bsxfun( @times, w_gaussians , dy_dsigma);  
                % de_dmu: tensor of shape: [samples, dim ,kernel]
                de_dsigma = bsxfun( @times, de_dy, dy_dsigma); 
            end
            
            %% derivative with respect w
            if numel(obj.w)==1
                de_dw = 0;
            else
                de_dw = bsxfun( @times, de_dy , obj.gaussians);  
            end
            
            %--------------------------------  derivatives of modifiers -----------------------------------%            
            %% derivative with respect sigma_in
            % if diagonal matrix
            if size(obj.sigma_inv,2)==obj.n_dim 
                de_dsigma= bsxfun(@times, de_dsigma, obj.sigma);
            end
            
            
            %% derivative with respect w_in
            de_dw = obj.de_dwin_eval(de_dw);
            
            
            %% if inputs are single parameters
            if size(obj.mu,1)==1 
                de_dmu = sum(de_dmu, 1);              
            end
            if size(obj.sigma_inv,1)==1 
                de_dsigma = sum(de_dsigma, 1);              
            end
            if size(obj.w,1)==1 
                de_dw = sum(de_dw, 1);              
            end
            
            
            
        end 
        
        
    end
end