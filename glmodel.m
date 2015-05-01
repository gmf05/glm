classdef glmodel
  properties
    coeff % coefficients
    cov % covariance matrix
    X
    y
    yEst
    link % link function
    fitMethod
    stats
    loglk % log-likelihood
    dev % deviance
    AIC % Akaike information criterion (penalized log-likelihood)    
  end
  
  methods
    function obj = fit(obj, d, p)
    % pp_model.fit(d, p)
    % INPUTS:
    % d -- point process data object
    % p -- point process params object
    %
    %
      warning(''); % clear last warning 
      fprintf(['\nFitting point process model...\n']);      
      obj.fitMethod = p.fitMethod;
      obj.link = p.link;
      obj.y = d.data(p.response,:)'; % response variable      
      
      % make design matrix
      Nparamtypes = length(p.covarNames);
      Nparams = p.covarInd{end}(end); % total # of covariates
      fsUpdateInd = p.covarInd{1}+1:Nparams;
      obj.X = ones(d.T,Nparams);
      fprintf(['Building design matrix...\n']);
      obj = obj.makeX(d,p);      
      fprintf(['Done!\n']);
      
      % trim burn-in period
      burnin = p.getBurnIn();
      burnin
      size(obj.y)
      obj.X = obj.X(burnin+1:end,:);
      obj.y = obj.y(burnin+1:end);         
      size(obj.y)
      
      fprintf(['Estimating parameters...']);
%       % MATLAB glmfit routine:
%       [b,dev,stats] = glmfit(obj.X,obj.y,'poisson','link',obj.link,'constant','off');

      % custom glmfit routine:
      [b,stats] = obj.glmfit0(obj.X,obj.y,obj.link);
      switch obj.fitMethod
        case 'glmfit'
          obj.coeff = b;
          obj.cov = stats.covb;
          obj.stats=stats;
          obj.yEst = obj.X * obj.coeff; % TODO: this must be passed through the link function!
        
        % TODO: this needs to be updated to be the classical kalman filter
        % rather than the point process version
        case 'filt'                    
          % initialize arrays
          NT = length(burnin:p.downsampleEst:d.T);
          bs = cell(1,NT);
          Ws = cell(1,NT);
%           obj.CIF = zeros(d.T - burnin, 1);
          Xbwd = flipud(obj.X);
          ybwd = fliplr(obj.y);
%           [bbwd,~,statsbwd] = glmfit(Xbwd,ybwd,'poisson','link',obj.link,'constant','off');
          [bbwd,statsbwd] = obj.glmfit0(Xbwd,ybwd,obj.link);
          Wbwd = statsbwd.covb;
          
          % noise (Kalman gain) matrix:
          noiseMtx = zeros(Nparams);
          for n = 1:Nparamtypes
            for i = p.covarInd{n}
              noiseMtx(i, i) = p.noise(n);
            end
          end
          noiseMtx = noiseMtx(fsUpdateInd,fsUpdateInd);
          
          fprintf(['\nFiltering backward...']);
          for t = 1:d.T-burnin
            dLdB = Xbwd(t,fsUpdateInd);
            lt = exp(Xbwd(t,:)*bbwd);
            A = Wbwd(fsUpdateInd,fsUpdateInd)+noiseMtx;
            U = dLdB';
            V = dLdB;
            C0 = lt;
            C1 = (1/C0+V*A*U);
            % matrix inversion lemma
            Wbwd(fsUpdateInd,fsUpdateInd) = A-A*U*(C1\V)*A;
            dB = Wbwd(fsUpdateInd,fsUpdateInd)*dLdB'*(obj.y(t) - lt);
            bbwd(fsUpdateInd) = bbwd(fsUpdateInd) + dB;
          end
%           b = bbwd;
%           b = rand(size(b));
          W = Wbwd;
          fprintf(['Done!\n']);
          
          % set initial values          
          bs{1} = b;
          Ws{1} = W;
          count = 2;
          
          fprintf(['\nFiltering forward...']);
          for t = 1:d.T-burnin
            dLdB = obj.X(t,fsUpdateInd);
            lt = exp(obj.X(t,:)*b);
            A = W(fsUpdateInd,fsUpdateInd)+noiseMtx;
            U = dLdB';
            V = dLdB;
            C0 = lt;
            C1 = (1/C0+V*A*U);
            % matrix inversion lemma:
            W(fsUpdateInd,fsUpdateInd) = A-A*U*(C1\V)*A;
            dB = W(fsUpdateInd,fsUpdateInd)*dLdB'*(obj.y(t) - lt);
            b(fsUpdateInd) = b(fsUpdateInd) + dB;
            obj.CIF(t) = exp(obj.X(t,:)*b);
            
            if mod(t, p.downsampleEst)==0
              bs{count} = b;
              Ws{count} = W;
              count = count+1;
            end
            
          end
          fprintf(['Done!\n']);
          
          obj.coeff = bs;
          obj.cov = Ws;
          
        case 'smooth'
          bs = cell(1,d.T-burnin);
          Ws = cell(1,d.T-burnin);
          NT = length(burnin:p.downsampleEst:d.T);
          bs0 = cell(1,NT);
          Ws0 = cell(1,NT);
          obj.CIF = zeros(d.T - burnin, 1);
          Xbwd = flipud(obj.X);
          ybwd = fliplr(obj.y);
          [bbwd,statsbwd] = obj.glmfit0(Xbwd,ybwd,obj.link);
          Wbwd = statsbwd.covb;
          
          % noise (Kalman gain) matrix:
          noiseMtx = zeros(Nparams);
          for n = 1:Nparamtypes
            for i = p.covariatee_ind{n}
              noiseMtx(i, i) = p.noise(n);
            end
          end
          noiseMtx = noiseMtx(fsUpdateInd,fsUpdateInd);
          
          fprintf(['Filtering backward...']);
          for t = 1:d.T-burnin
            dLdB = Xbwd(t,fsUpdateInd);
            lt = exp(Xbwd(t,:)*bbwd);
            A = Wbwd(fsUpdateInd,fsUpdateInd)+noiseMtx;
            U = dLdB';
            V = dLdB;
            C0 = lt;
            C1 = (1/C0+V*A*U);
            % matrix inversion lemma
            Wbwd(fsUpdateInd,fsUpdateInd) = A-A*U*(C1\V)*A;
            dB = Wbwd(fsUpdateInd,fsUpdateInd)*dLdB'*(obj.y(t) - lt);
            bbwd(fsUpdateInd) = bbwd(fsUpdateInd) + dB;
          end
          b = bbwd; W = Wbwd;
          fprintf(['Done!\n']);
          
          % or, instead of filtering backward,
          % initialize using glmfit
%           [b,dev,stats] = glmfit(obj.X,obj.y,'poisson','constant','off');
%           W = stats.covb;
          
          fprintf(['Filtering forward...']);
          for t = 1:d.T-burnin
            dLdB = obj.X(t,fsUpdateInd);
            lt = exp(obj.X(t,:)*b);
            A = W(fsUpdateInd,fsUpdateInd)+noiseMtx;
            U = dLdB';
            V = dLdB;
            C0 = lt;
            C1 = (1/C0+V*A*U);
            % matrix inversion lemma:
            W(fsUpdateInd,fsUpdateInd) = A-A*U*(C1\V)*A;
            dB = W(fsUpdateInd,fsUpdateInd)*dLdB'*(obj.y(t) - lt);
            b(fsUpdateInd) = b(fsUpdateInd) + dB;
            obj.CIF(t) = exp(obj.X(t,:)*b);            
            bs{t} = b;
            Ws{t} = W;
          end                              
          fprintf(['Done!\n']);
          
          fprintf(['Smoothing...']);
          b = bs{end}; W = Ws{end};
          bs0{end} = b; Ws0{end} = W;
          count = NT - 1;
          for t = d.T-burnin-1:-1:1
            W0 = Ws{t}(fsUpdateInd,fsUpdateInd);
            W1 = W0+noiseMtx;
            iW1 = inv(W1);
            b(fsUpdateInd) = bs{t}(fsUpdateInd) + W0*iW1*(b(fsUpdateInd) - bs{t}(fsUpdateInd));
            W(fsUpdateInd,fsUpdateInd) = W0 + W0*iW1*(W(fsUpdateInd,fsUpdateInd) - W1)*iW1*W0;
            lt = exp(obj.X(t,:)*b);
            
            if mod(t, p.downsampleEst)==0
              bs0{count} = b;
              Ws0{count} = W;
              count = count-1;
            end
          end
          fprintf(['Done!\n']);
          
          obj.coeff = bs0;
          obj.cov = Ws0;
      end
      fprintf(['Done!\n']);
      
      % TODO: add method for computing goodness-of-fit??
      obj = obj.calcGOF(); % goodness-of-fit
    end
    
    function obj = calcGOF(obj, p)      
      switch obj.link
        % NOTE: Commented lines are specific to poisson/point process
        % models
% %         case 'log'
% %           obj.loglk = sum(log(poisspdf(obj.y,obj.CIF)));
% %           obj.dev = 2*(sum(log(poisspdf(obj.y,obj.y))) - obj.loglk);
% %         case 'logit'
% %           obj.loglk = sum(log(binopdf(obj.y,ones(size(obj.y)),obj.CIF)));
% %           obj.dev = 2*(sum(log(binopdf(obj.y,ones(size(obj.y)),obj.y))) - obj.loglk);
        case 'identity'
          % TODO: check/fix formula for log-likelihood
          obj.loglk = sum(log(normpdf(obj.y, obj.X*obj.coeff, 1)));
          obj.dev = 2* (sum(log(normpdf(obj.y, obj.y, 1))) - obj.loglk);
      end
      obj.AIC = obj.dev+2*size(obj.coeff,1);
      
% %       % time rescaling & KS test (NOTE: Should only be for poisson
% %       % regression)
% %       obj = obj.rescaled_ISI(); % compute rescaled ISI, add to object properties
% %       [ks_stat,ks_ci,~,ks_p] = KStest(obj.y,obj.CIF); 
% % %       [ks_stat,ks_ci,ks_p] % uncomment to check internal KStest against
% % %       stat toolbox's kstest.m
% % %       obj.KS = [ks_stat,ks_ci,ks_p];    
% %       testx = 0:0.01:1; testcdf = unifcdf(testx,0,1); matcdf = [testx' testcdf'];
% %       [~,ks_p,ks_stat,ks_ci] = kstest(obj.rsISI,'CDF',matcdf,'Alpha',0.05);
% %       obj.KS = [ks_stat,ks_ci,ks_p];      
% % %       [ks_stat,ks_ci,ks_p] % uncomment to check internal KStest against
% % %       stat toolbox's kstest.m
      
    end
    
    
    function obj = makeX(obj, d, p)
      obj.X = zeros(d.T, p.Ncovar());
      for i = 1:length(p.covarNames)
        channels = p.covarChannels{i};
        basis = p.covarBases{i};     
        knots = p.covarKnots{i};
        ind = p.covarInd{i};        
        Xi = obj.makeXblock(d, channels, basis, knots, p.s);
        obj.X(:,ind) = Xi; clear Xi;
      end
%       obj.X = obj.X(p.getBurnIn()+1:end,:);
    end
    
    function Xi = makeXblock(obj, d, channels, basis, knots, s)
      if nargin<6, s = 0.5; end
      sCoeff = [-s  2-s s-2  s; 2*s s-3 3-2*s -s; ...
           -s   0   s   0;   0   1   0   0];
      N = length(knots);
      
      % make X for firing rate covariate
      if isequal(channels,0)
        bins_per_knot = round(diff(knots)*d.T);
        count = 0;
        
        switch basis
          case 'spline'
            Xi = zeros(d.T, N+2);
            for n = 1:N-1
              temp=bins_per_knot(n);
              alphas=1/temp*(1:temp);
              Xi(count + (1:temp), n+(0:3)) = Xi(count + (1:temp), n+(0:3)) + ...
                  [alphas'.^3 alphas'.^2 alphas' ones(temp, 1)]*sCoeff;
              count=count+bins_per_knot(n);
            end
          case 'indicator'
            Xi = zeros(d.T, N-1);
            for n = 1:N-1
              Xi(count+(1:bins_per_knot(n)),n) = 1;
              count = count+bins_per_knot(n);
            end
        end
        
      % make X for other covariates
      else
        
        if length(channels)<=1
          dat = double( d.data(channels,:) );
        else
          dat = double( sum(d.data(channels,:),1) );
        end
        
        switch basis
          case 'spline'
            if range(knots)>1, dtau = 1; else dtau = 0.01; end;
            tau = knots(1):dtau:knots(end);
            NT = length(tau);
            onset = knots(1);
            offset = knots(end);
            intvls = diff(knots)*1/dtau;                        
            
            Xi = zeros(d.T, N+2);
            Xs = zeros(NT,N+2);
            
            count=1;
            for n=1:N-1
              I = intvls(n); % length of interval (num. of bins)
              alphas = (0:I-1)./I;
              Xs(count:count+I-1, n+(0:3)) = [alphas'.^3 alphas'.^2 alphas' ones(I,1)] * sCoeff;
              count = count+I;
            end
            Xs(end, N-1:N+2) = [1 1 1 1] * sCoeff; % alpha = 1
        
            %%%%%%%%%%
%             
%             for t = 1:d.T - offset
%               Xi(t + (onset:offset),:) = ...
%                 Xi(t + (onset:offset),:) + dat(t) * Xs;
%             end
%             
%             for t = d.T - offset + (1:offset)
%               bins_to_end = d.T - t;
%               Xi(t + (onset:bins_to_end),:) = ...
%                 Xi(t + (onset:+bins_to_end),:) + dat(t) * Xs(1:bins_to_end-onset+1,:);
%             end
            
            %%%
            Xl = zeros(d.T, NT);
            for t = 1:NT    
              Xl(tau(t)+1:end, t) = dat(1:end-tau(t))';
            end
            Xi = Xl * Xs;
            %%%%%%%%%%
          case 'indicator'
            Xi = zeros(d.T, N);
            burnin = knots(end);
            for n = 1:N
              Xi(burnin+1:end, n) =  dat((burnin+1:end)-knots(n))';
            end
        end
      end
    end
    
    function obj = irls(obj, p)
      if nargin<2
        if isempty(obj.fitMethod), obj.fitMethod = 'glmfit'; end
        if isempty(obj.link), obj.link = 'identity'; end
      else
        obj.fitMethod = p.fitMethod;
        obj.link = p.link;
      end
      [b,stats] = glmfit0(obj,obj.X,obj.y,obj.link);
      obj.coeff = b;
      obj.cov = stats.covb;
      obj.stats = stats;
    end
    
    function [b,stats] = glmfit0(~, X, y_in, link)
      
      if size(y_in,2)>size(y_in,1), y = y_in';
      else y = y_in; end;

      % starting values:
% %       switch distr
% %       case 'poisson'
% %           mu = y + 0.25;
% %       case 'binomial'
% %           mu = (N .* y + 0.5) ./ (N + 1);
% %       case {'gamma' 'inverse gaussian'}
% %           mu = max(y, eps(class(y))); % somewhat arbitrary
% %       otherwise
% %           mu = y;
% %       end
      switch link
        case 'identity'
          linkFn = @(x) x;
          ilinkFn = @(x) x;
          linkFnprime = @(x) ones(size(x));
          sqrtvarFn = @(x) ones(size(x));
          mu = y; % initial conditions
        case 'log'
          linkFn = @(x) log(x);
          ilinkFn = @(x) exp(x);
          linkFnprime = @(x) 1./x;     
          sqrtvarFn = @(x) sqrt(x);
          mu = y + 0.25; % initial conditions
      end
      
      N = size(X,1);
      p = size(X,2);
      pwts = ones(N,1);
      b = ones(p,1);
%       b = randn(p,1);
      R = eye(p);
      eta = linkFn(mu);
      
      % convergence parameters
      eps = 1e-6;
      iterLim = 100;
      offset = 1e-3;

      for iter = 1:iterLim
        z = eta - offset + (y - mu) .* linkFnprime(mu);
        b_old = b;
        R_old = R;
        deta = linkFnprime(mu);
        sqrtirls = abs(deta) .* sqrtvarFn(mu);
        sqrtw = sqrt(pwts) ./ sqrtirls;

        % orthogonal (QR) decomposition of Xw
        % avoids forming the product Xw'*Xw
        zw = z .* sqrtw;
        Xw = X .* sqrtw(:,ones(1,p));
        [Q,R] = qr(Xw,0);
        b = R \ (Q'*zw);
              
        %-----
        % check convergence
        % if there's a problem with convergence:
%         if rcond(R)<1e-8, iter, b=b_old; R=R_old; break; end
%         if rcond(R)<1e-8, disp('Flat likelihood'), iter, break; end
        if rcond(R)<1e-8 || isnan(rcond(R)), warning('Flat likelihood'), iter; end
        if sum(isnan(b))>0, iter, b=b_old; R=R_old; break; end
        
        %
        % should we also add a function to diagnose convergence problems
        % and then take appropriate action???
        % 
        % stop if converged:
        if norm(b - b_old, inf) < eps
          fprintf(['Converged in ' num2str(iter) ' steps. ']);
          break;
        end
        %-----
        
        eta = offset + X*b;
        mu = ilinkFn(eta);
%         % plot to debug convergence issues:
%         clf, subplot(211), plot(b,'b-o'); subplot(212), plot(deta), pause;
%         save(['~/temp/irls' num2str(iter) '.mat'],'-v7.3','eta','mu','z','y','b','b_old','R','R_old','Q','Xw','zw','X','offset','N','p','pwts');
      end
      
      % glmfit covariance:
      RI = R\eye(p);
      C = RI * RI';
      % assumes no over/under-dispersion (s=1):
      % C=C*s^2;      
      stats.beta = b;
      stats.dfe = N-p;
      stats.sfit = [];
      stats.covb = C;
      stats.s = 1;
      stats.estdisp = 0;
      stats.se = sqrt(diag(stats.covb));
      stats.t = b ./ stats.se;
      stats.p = 2 * normcdf(-abs(stats.t));
      % stats.wts = diag(W);
    end
    
    function plot(obj, d, p)
      
      global PLOT_COLOR;
      global DO_CONF_INT;
      global DO_MASK;
      
      Z = 2;
      Nparamtypes = length(p.covarNames);
      burnin = p.getBurnIn();
      % NOTE: ASSUMES 'rate' is first covariate
      % and all other types ('self-hist', 'ensemble', etc)
      % come afterwards
      
      % RATE----------------------
      subplot(Nparamtypes,1,1); hold on;
      T0 = length(p.covarKnots{1});
      ind = p.covarInd{1};
      switch obj.fitMethod
        case 'glmfit'
          b1 = obj.coeff(ind);
          if DO_CONF_INT, W1 = obj.cov(ind,ind); end;
        case {'filt','smooth'}
          b1 = obj.coeff{1}(ind);
          if DO_CONF_INT, W1 = obj.cov{1}(ind,ind); end;
      end
      
      switch p.covarBases{1}
        case 'spline'
          if DO_CONF_INT
            [t_axis,Y,Ylo,Yhi] = plotspline(p.covarKnots{1},b1,p.s,W1,Z);
            t_axis = t_axis*(d.time(end)-d.time(1)) + d.time(1); % convert to secs
            L = Y'; Llo = Ylo'; Lhi = Yhi'; % TODO: pass these through link function!
%             plot(t_axis,L,PLOT_COLOR,t_axis,Lhi,[PLOT_COLOR '--'],t_axis,Llo,[PLOT_COLOR '--']);
            shadedErrorBar(t_axis,L,[Lhi-L; L-Llo],{'Color',PLOT_COLOR});
          else
            [t_axis,Y] = plotspline(p.covarKnots{1},obj.coeff(ind),p.s);
            t_axis = t_axis*(d.time(end)-d.time(1)); % convert to secs
            L = Y'; % TODO: pass these through link function!
            plot(t_axis,L,'Color',PLOT_COLOR,'linewidth',2);
%             plot(p.covarKnots{1}*t_axis(end),exp(b1(2:end-1))/d.dt,'Color',[PLOT_COLOR 'o'],'linewidth',2);
          end
        case 'indicator'
          t_axis = p.covarKnots{1} * (d.time(end)-d.time(1)) + d.time(1);
          gca(); hold on;
          if DO_CONF_INT
            Y = b1;
            % NOTE: NEED TO MODIFY HOW Ylo, Yhi
            % are computed
            Ylo = Y - Z*sqrt(diag(W1)); Yhi = Y + Z*sqrt(diag(W1)); % 1d case
            L = Y'; Llo = Ylo'; Lhi = Yhi'; % TODO: pass these through link function!
            for t = 1:T0-1
%               plot(t_axis,L,PLOT_COLOR,t_axis,Lhi,[PLOT_COLOR '--'],t_axis,Llo,[PLOT_COLOR '--']);
              shadedErrorBar([t_axis(t),t_axis(t+1)],L(t)*ones(1,2),[(Lhi(t)-L(t))*ones(1,2); (L(t)-Llo(t))*ones(1,2)],{'Color',PLOT_COLOR},1);              
            end
          else
            for t = 1:T0-1
              L = b1(t); % TODO: pass this through link function!!
              plot([t_axis(t),t_axis(t+1)],L*ones(1,2),PLOT_COLOR,'linewidth',2);
            end
          end
      end
      xlabel('time [s]');
%       ylabel('--');
      title(p.covarNames{1});
      
      % OTHER COVARIATES----------
      for covarNum = 2:Nparamtypes        
        subplot(Nparamtypes,1,covarNum); hold on;
        ind = p.covarInd{covarNum};
        switch obj.fitMethod
          case 'glmfit'
            switch p.covarBases{covarNum}
              case 'spline'
                if DO_CONF_INT
                  [lagaxis,Y,Ylo,Yhi] = plotspline(p.covarKnots{covarNum},obj.coeff(ind),p.s,obj.cov(ind,ind),Z);
                  lagaxis = lagaxis*d.dt*1e3; % convert from bins to ms
                  L = Y'; Llo = Ylo'; Lhi = Yhi';  % TODO: Pass these through link!
                  shadedErrorBar(lagaxis,L,[Lhi-L; L-Llo],{'Color',PLOT_COLOR},1);
                else
                  [lagaxis,Y] = plotspline(p.covarKnots{covarNum},obj.coeff(ind),p.s);
                  lagaxis = lagaxis*d.dt*1e3; % convert from bins to ms
                  L = Y'; % TODO: pass this through link!!
                  plot(lagaxis,L,PLOT_COLOR,'linewidth',2);
%                   plot(p.covarKnots{covarNum},exp(obj.coeff(ind(2:end-1))),[PLOT_COLOR 'o'],'linewidth',2);
                end
              case 'indicator'
                lagaxis = p.covarKnots{covarNum};
                if DO_CONF_INT                  
                  Y = exp(obj.coeff(ind));                  
                  error('write more code!');
                  % assign Y, Ylo, Yhi based on covariance structure
                  % TODO: pass these through link!!
                  L = Y'; Llo = Ylo'; Lhi = Yhi';
                  shadedErrorBar(lagaxis,L,[Lhi-L; L-Llo],{'Color',PLOT_COLOR},1);
                else
                  plot(lagaxis,exp(obj.coeff(ind)'),PLOT_COLOR,'linewidth',2);                  
                end
            end
            plot([lagaxis(1) lagaxis(end)],[0 0],'k--');
            xlabel('lag time [ms]');
%             ylabel('--');

          case {'filt', 'smooth'}
            tInd = burnin+1:p.downsampleEst:d.T; % modified time axis
            NT = length(tInd);
            switch p.covarBases{covarNum}
              case 'spline'                
                lagaxis = p.covarKnots{covarNum}(1):p.covarKnots{covarNum}(end);
                lagaxis = lagaxis*d.dt*1e3; % covert from bins to ms
                allL = zeros(length(lagaxis), NT);            
                for t = 1:NT
                  if DO_MASK
                    [~,Y,Ylo,Yhi] = plotspline(p.covarKnots{covarNum},obj.coeff{t}(ind),p.s,obj.cov{t}(ind,ind),Z);
                    % mask y:
                    goodInd = [find(Yhi>0)', find(Ylo<0)'];
                    badInd = setdiff(1:length(lagaxis), goodInd);
                    Y(badInd) = 0;
                    allL(:,t) = exp(Y');
                  else
                    [~,Y] = plotspline(p.covarKnots{covarNum},obj.coeff{t}(ind),p.s);
                    allL(:,t) = exp(Y');                
                  end
                end
              case 'indicator'
                lagaxis = p.covarKnots{covarNum}(1):p.covarKnots{covarNum}(end);
                lagaxis = lagaxis*d.dt*1e3; % covert from bins to ms
                allL = zeros(length(lagaxis), NT);
                for t = 1:NT
                  if DO_MASK
                    Y = exp(obj.coeff{tInd(t)}(ind));
                    % assign Ylo, Yhi
                    error('write more code!');                    
                    % mask Y:
                    goodInd = [find(Yhi>0), find(Ylo<0)];
                    badInd = setdiff(1:length(lagaxis), goodInd);
                    Y(badInd) = 0;
                    allL(:,t) = exp(Y');
                  else
                    allL(:,t) = exp(obj.coeff{tInd(t)}(ind));
                  end
                end
            end
            imagesc(d.time(tInd), lagaxis, allL);
            xlabel('time [s]');
            ylabel('lag time [ms]');            
        end
        xlim(round([p.covarKnots{covarNum}(1),p.covarKnots{covarNum}(end)*0.8]*d.dt*1e3));
        title(p.covarNames{covarNum});
      end
    end
  end
end