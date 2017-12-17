function [pl_svm,pref] = train_hog3d_model(pos_data,neg_data,pref,method,numIters,si,sj,st)

%%  Input:
%       pos_data: array or positive samples: NxM, where N is the number of samples and M = w*h*t of the spatio-temporal cube
%       neg_data: array or negative samples: NxM
%       pref:     some label added to the saved file (can be set arbitrary)
%       method:   either 'svm' or 'btr' (for boosted trees detector)
%       numIters: number of iterations for the Boosted trees algorithm, (only used if you choose 'btr' before otherwise set it to [])
%       si:       height of the st-cube
%       sj:       width of the st-cube
%       st:       temporal depth of the st-cube

%%  Output:
%       pl_svm:   trained model of either 'svm' or 'btr' type
%       pref:     some label added to the saved file


addpath(genpath('~/path/to/liblinear'));                % <-- modify this
addpath(genpath('~/path/to/Piotr_s_Dollars_toolbox'));  % <-- modify this

if(nargin < 1)
    disp(' ');
    disp('[pl_svm,pref] = train_svm_hog3d(plane_db,neg_plane_db,pref,method,numIters);');
    disp(' ');
    return;
end

if(nargin < 5)||(isempty(numIters))
    numIters = 400;
end

if(nargin < 6)||(isempty(si))
    si = 40;
end
if(nargin < 7)||(isempty(sj))
    sj = 40;
end
if(nargin < 8)||(isempty(st))
    st = 4;
end

psz = size(pos_data,1);
nsz = size(neg_data,1);

switch(method)
    case 'svm'
        group = [ones(psz,1); zeros(nsz,1)];
    case 'btr'
        group = [ones(psz,1); -1*ones(nsz,1)];
    otherwise
        group = [ones(psz,1); zeros(nsz,1)];
end
plane_data = [pos_data; neg_data];
plane_data = double(plane_data);

[nsamples,~] = size(plane_data);

display('extracting HoG..');

rsize = [16 16 2];
numcell = [2 2 2];

for i = 1:nsamples
    plane_data(i,:) = plane_data(i,:)./255;
end

time1 = tic;
tmp = myhog3d(reshape(plane_data(1,:),si,sj,st),rsize,numcell,1,0);
svm_data = zeros(nsamples,numel(tmp));
for i = 1:nsamples
    svm_data(i,:) = myhog3d(reshape(plane_data(i,:),si,sj,st),rsize,numcell,0,0);
   if(toc(time1) > 2)
       clc; fprintf('%0.1f%% finished\n',100*i/nsamples);
       time1 = tic;
   end
end

switch(method)
    case 'svm'
        % training SVM
        pl_svm = train(group, sparse(double(svm_data)),'-B 1');
        save(strcat('svm_',date,pref,'.mat'),'pl_svm','rsize','numcell','pref','-v7.3');
    case 'btr'
        opts = [];
        opts.loss = 'exploss'; % can be logloss or exploss
        opts.shrinkageFactor = 0.1;
        opts.subsamplingFactor = 0.5;
        opts.maxTreeDepth = uint32(2);  % this was the default before customization
        opts.randSeed = uint32(rand()*1000);

        model = SQBMatrixTrain(single(svm_data), group, uint32(numIters), opts);
        save(strcat('btr_',date,pref,'.mat'),'model','rsize','numcell','pref','-v7.3');
        pl_svm = model;
    otherwise
        pl_svm = train(group, sparse(double(svm_data)),'-B 1');
        save(strcat('svm_',date,pref,'.mat'),'pl_svm','rsize','numcell','pref','-v7.3');
end
