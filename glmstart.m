global GLM PLOT_COLOR DO_CONF_INT DO_MASK
GLM = '~/Code/repos/glm';
addpath(GLM)
addpath([GLM '/helper']);
addpath('~/Code/repos/mgh/classes')
addpath(genpath('~/Code/repos/mgh/helper'));
PLOT_COLOR ='b'; % 
DO_CONF_INT = true; % plot confidence intervals?
DO_MASK = false; % mask insignificant filter/smoother estimates?