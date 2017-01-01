classdef (Abstract) MCC
    %MCC: Base class for MCC related functions.
    %   Refer method specific help for more details. 
    %
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties
    end
    
    methods (Access = public, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ret] = gauss(f)
            %MCC_GAUSS: applies 'f' to a gauss distribution.
            %
            % Parameters
            % ----------
            % f : double
            %     Fraction (x^2/sigm^2).
            %
            
            ret = exp(-f/2);
        end  
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [sigma_2] = calc_sigma_q(err, n)
            %CALC_SIGMA_q: calculates sigma^2 (kernel bandwidth) based on quantiles.
            %
            % Parameters
            % ----------
            % err : double
            %     Error/Cost.
            %
            % n : double
            %     No. of samples.
            %
            
            % Calculate quantiles
            quantiles = quantile(err,3);
            q_range = quantiles(3) - quantiles(1);
            
            % Silverman's Rule
            sigma_2 = 1.06 * min(std(err), q_range/1.34)*(n^(-1/5));            
        end 
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       	function [sigma_2] = calc_sigma_s(err, s)
            %CALC_SIGMA_S: calculates sigma^2 (kernel bandwidth) based on simple criteria. 
            %
            % Parameters
            % ----------
            % err : double
            %     Error/Cost.
            %
            % s : double
            %     Sacling parameter.
            %
            
            % [REF. MCC_PCA Paper]
            sigma_2 = (1/(0.7*s)) * sum(err);
        end 
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      	function [sigmaM_2] = calc_sigma_dcc(err, n, n_per_class, cls_n)
            %CALC_SIGMA_DCC: calculates sigma^2 (kernel bandwidth) based on custom criteria.
            %   This calculation utilizes the calculated weights and std to provide the sigma. Each
            %   class may have different sigma's depending on its data distribution. 
            %
            % Parameters
            % ----------
            % err : double
            %     Error/Cost.
            %
            % n : double
            %     No. of samples.
            %
            % n_per_class : double
            %     No. of samples per class.
            %
            %
            % Returns
            % -------
            % sigmaM : 2D matrix -double
            %     Diagional matrix indicating sigmas.
            %
            
            % Initialize sigmaM matrix
            sigmaM_2 = zeros(n, n);

            % Set class start
            cls_st = 1;
            
            % Iterate the classes
            for i = 1:cls_n
                
                % Calculate range for class i
                range = cls_st:cls_st+n_per_class(i)-1;
                
                % Class specific error
                cls_error = err(range);

                % New Formulation (For Numerical Stability of Toy Example)
                tmp = cls_error;
                tmp(tmp < 1e-15) = max(cls_error);  % To handle the minumum 0 case
                w = exp(-cls_error./min(tmp));      % Calculate weights 
                std_val = std(cls_error, w);        % Weighted std.
                
                % For stability, when std is close to 0
                if (std_val < 1)
                    sigma = 1;
                else
                    sigma = std_val;
                end

                % sigma = std(cls_error)/s; % TODO: for symmectric error (-1, +1) std = 0

                % Set the diagonal matrix sigmaM
                sigmaM_2(range, range) = ...
                    diag(1./repmat(sigma, 1, n_per_class(i)));
                
                % Update cls_st for the next iteration
                cls_st = cls_st + n_per_class(i);
            end  

            mm = min(diag(sigmaM_2));
            sigmaM_2(sigmaM_2 > mm)= mm;
        end 
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
end
