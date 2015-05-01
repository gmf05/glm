classdef glmdata
  properties
    name % name of data set
    labels % name of each channel
    data 
    Nchannels % number of channels
    time % time axis
    dt % time step size
    Fs % sampling rate [Hz] = 1/dt
    T % number of time points
  end
  
  methods
    % constructor
    function obj = glmdata(data,time,varargin)
      obj.data = data;
      obj.Nchannels = size(data,1);      
      if nargin<2
        obj.T = size(data,2);
        obj.time = 1:obj.T;
      else
        obj.time = time;
        obj.T = length(time);
      end
      obj.dt = obj.time(2) - obj.time(1);
      obj.Fs = 1/obj.dt;
     
      % parse varargin for name, labels if provided
      for n = 1:2:length(varargin)
        switch varargin{n}
          case 'name', obj.name = varargin{n+1};
          case 'labels', obj.labels = varargin{n+1};
        end
      end          
    end
    
    % get a portion of the data (i.e. over a subset of the time axis)
    function obj = subtime(obj,varargin)
      % keeps all marks rather than distinguishing
      % which are related to the new time window
      % this approach is meant for quick and dirty
      % applications
      if length(varargin)==1
        ind = varargin{1};
      else
        beg_ind = getclosest(obj.time,varargin{1});
        end_ind = getclosest(obj.time,varargin{2});
        ind = beg_ind:end_ind;
      end
      
      obj.data = obj.data(:,ind);
      obj.time = obj.time(ind);
      obj.T = length(obj.time);
    end
    
    % get a portion of the data (i.e. over a subset of the channels)
    function obj = subdata(obj,ind)
      if ~isempty(obj.labels), labels = {obj.labels{ind}}; else labels = {}; end %#ok
      obj = glmdata(obj.data(ind,:), obj.t, 'name', obj.name, 'labels', labels); %#ok
    end
    
    function obj = timereset(obj)
      obj.time = (1:obj.T)*obj.dt;
    end
    
    function obj = downsample(obj, dsfactor)
      T0 = ceil(obj.T/dsfactor);
      d = zeros(obj.Nchannels, T0);
      for t = 1:T0-1
        d(:,t) = mean(obj.data(:, (t-1)*dsfactor + (1:dsfactor)), 2);
      end
      d(:,end) = mean(obj.data(:, (T0-1)*dsfactor+1 : end), 2);
      obj.data = d;
      obj.time = obj.time(1:dsfactor:end);
      obj.T = T0;
      
%       obj.data = obj.data(:, 1:dsfactor:end);
%       obj.time = obj.time(1:dsfactor:end);
%       obj.T = length(obj.time);
      obj.dt = obj.dt*dsfactor;
      obj.Fs = 1/obj.dt;
    end
    
    % concatenate over time
    function obj = concat(obj,obj2)
      % a couple general checks between old & new objects
      if abs(obj.dt-obj2.dt)>1e-10
        error('ERROR: data objects must have the same time resolution dt');
      elseif size(obj.data,1)~=size(obj2.data,1)
        error('ERROR: data objects must contain the same number of dimensions');
      end
      obj.data = [obj.data obj2.data];
      obj.time = [obj.time obj2.time];
      obj.T = length(obj.time);
      % TODO: check labels to ensure they match??
    end
    
    % concatenate over channels
    function obj = addchannels(obj,obj2)
      obj.data = [obj.data; obj2.data];
      obj.Nchannels = obj.Nchannels+size(obj2.data,1);
      obj.labels = [obj.labels obj2.labels]; % TODO: matlab suggested this fix. double check that it works
    end
  end
end