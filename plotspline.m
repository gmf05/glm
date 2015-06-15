function [tau, y, y_lo, y_hi, covr] = plot_spline(knots, b, s, W, Z, do_mask)
%   plot_spline.m
%   [tau, y, y_lo, y_hi] = plot_spline(knots, b, s, W, Z)
%   knots: partition of the covariate axis
%   b: model parameters
%   s: tension parameter (usually 0.5)
%   W: covariance matrix assoc. w/ parameters b
%   Z: z-value associated w/ sig. level of confidence interval (usually 2 = 95%)

%   Given knots and spline coefficients, plots the interpolating
%   cubic spline.
%
    % checks::
    if nargin<3 || isempty(s), s=0.5; end;
    if nargin<4, do_conf_int = false; else do_conf_int = true; end;
    if nargin<5, Z = 2; end; % z-value associated w/ sig. level of confidence interval
    if nargin<6, do_mask = false; end;
    % standard axis partition is by 1 unit
    % unless range of axis < 1 -- then .01    
    
%     if  range(knots)>1000, dtau = 2;
%     elseif range(knots)>1, dtau = 1;
%     elseif range(knots) dtau = 1e-2; end;
    
    if range(knots)>1, dtau = 1;
    else dtau = 1e-2; end;
      
    % tension matrix
    s_coeff = [-s  2-s s-2  s; 2*s s-3 3-2*s -s; ...
               -s   0   s   0;   0   1   0   0];

    % covariate axis
    tau = knots(1):dtau:knots(end);
    N_knots = length(knots);
    NT = length(tau);
    
    % set X0 based on knots        
    X0 = zeros(NT,N_knots+2);
    
    % note: rounding below fixes occasional error where
    % 0:spacing(i)-1 turns out to be one element short (because spacing(i)
    % gets rounded down)
    spacing = round(diff(knots)*1/dtau); 
    count=1;
    for i=1:length(spacing)
        alphas = (0:spacing(i)-1)./spacing(i);
        X0(count:count+spacing(i)-1, i:i+3) = [alphas'.^3 alphas'.^2 alphas' ones(spacing(i),1)] * s_coeff;
        count = count+spacing(i);
    end
    X0(end, i:i+3) = [1 1 1 1] * s_coeff; % alpha = 1
    
    % get spline values
    y = X0*b;
    
    % get confidence intervals, if directed
    if do_conf_int
        y_hi = X0*b + 2*sqrt(diag(X0*W*X0'));
        y_lo = X0*b - 2*sqrt(diag(X0*W*X0'));
        covr = X0*W*X0';
        
        if do_mask
            ind = find(y_hi'>0 & y_lo'<0);
            y(ind) = 0;
        end
    end

end