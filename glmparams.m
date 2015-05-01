classdef glmparams
  properties
    covarNames
    covarChannels % channels corresponding to each covar
    covarKnots % partition of time/lag axis
    covarBases % list of 'spline' or 'indicator' for each covar
    covarInd % lists of indices corresponding to each covar
    response % channel being modeled
    fitMethod % 'glmfit', 'filt', or 'smooth'
    link % link function for glmfit (default 'log')    
    alpha % desired significance level for tests, etc. (default = 0.05)
    noise % noise (i.e. Kalman gain) parameters for filter/smoother
    s % tension parameter for spline interpolation (default s = 0.5)
    downsampleEst % factor for downsampling estimates (default 1)
    % TODO: delete the property below? 
%     window % (default window = [1 0.2])
    
  end
  
  methods
    % constructor
    function obj = glmparams()
      obj.fitMethod = 'glmfit';
      obj.link = 'identity';
      obj.s = 0.5;
      obj.alpha = 0.05;
    end
    
    function obj = addCovar(obj, name, channels, knots, basis)
      if nargin<1, name=''; end
      if nargin<2, basis = 'indicator'; end;
      obj.covarNames{end+1} = name;
      obj.covarChannels{end+1} = channels;
      obj.covarKnots{end+1} = knots;
      obj.covarBases{end+1} = basis;
      N = length(knots);
      N = N + 2*isequal(basis,'spline');
%       N = N - 1*isequal(basis,'indicator')*isequal(name,'rate');
      N = N - 1*isequal(basis,'indicator')*(channels==0);
      if isempty(obj.covarInd)        
        ind = 1:N;
      else
        ind = obj.covarInd{end}(end) + (1:N);
      end
      obj.covarInd{end+1} = ind;
    end
    
    function obj = delCovar(obj, i)
      if nargin<2, i = length(obj.covarNames); end
      obj.covarNames(i)=[];
      obj.covarInd(i)=[];
      obj.covarKnots(i)=[];
      obj.covarChannels(i)=[];
      obj.covarBases(i)=[];
    end
    
    function obj2 = getCovar(obj, i)
      obj2 = glmparams();
      obj2 = obj2.addCovar(obj.covarNames{i},obj.covarChannels{i},obj.covarKnots{i},obj.covarBases{i});
      obj2.fitMethod = obj.fitMethod; 
    end
    
    function burnin = getBurnIn(obj)      
      burnin = 0;
      % TODO: we're assuming covariate #1 is the intercept/rate term
      % is there a better way to handle this??
      for i = 2:length(obj.covarNames)
        burnin = max([burnin, obj.covarKnots{i}(end)]);
      end
    end

    function Xs = splineXi(obj, i)
      % i : covariate index -- makes a block of the spline matrix
      knots = obj.covarKnots{i};
      s = obj.s;
      
      if range(knots)>1, dtau = 1; else dtau = 0.01; end;
      tau = knots(1):dtau:knots(end);
      NT = length(tau);
      N = length(knots);
      onset = knots(1);
      offset = knots(end);
      intvls = diff(knots)*1/dtau;
      sCoeff = [-s  2-s s-2  s; 2*s s-3 3-2*s -s; ...
           -s   0   s   0;   0   1   0   0];
      
      if isequal(obj.covarBases{i}, 'indicator')
        switch obj.covarNames{i}
          case {'rate', 'intercept', 'dummy'}
            N = N-1; % n windows = n knots - 1
        end
        Xs = eye(N); return;
      end
      
      Xs = zeros(NT,N+2);
      count=1;
      for n=1:N-1
        I = intvls(n); % length of interval (num. of bins)
        alphas = (0:I-1)./I;
        Xs(count:count+I-1, n+(0:3)) = [alphas'.^3 alphas'.^2 alphas' ones(I,1)] * sCoeff;
        count = count+I;
      end
      Xs(end, N-1:N+2) = [1 1 1 1] * sCoeff; % alpha = 1
    end
    
    function Xs = splineX(obj,ind)
      s = obj.s;
      knots = obj.covarKnots{ind};
      N = length(knots);
      sCoeff = [-s  2-s s-2  s; 2*s s-3 3-2*s -s; ...
             -s   0   s   0;   0   1   0   0]; %#ok
      tau = knots(1):knots(end);
      NT = length(tau);
      intvls = diff(knots);
      Xs = zeros(NT,N+2);

      count=1;
      for n=1:N-1
        I = intvls(n); % length of interval (num. of bins)
        alphas = (0:I-1)./I;
        Xs(count:count+I-1, n+(0:3)) = [alphas'.^3 alphas'.^2 alphas' ones(I,1)] * sCoeff;
          count = count+I;
      end
      Xs(end, N-1:N+2) = [1 1 1 1] * sCoeff; % alpha = 1
    end
    
    function N = Ncovar(obj)
      N = obj.covarInd{end}(end);
    end
    
  end
end