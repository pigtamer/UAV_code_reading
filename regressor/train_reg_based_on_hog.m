function train_reg_based_on_hog(data,annot_data,si,sj,hbin,todo,sh_hor,sh_vert,sh_sc,numIters,suff)

if(nargin < 1)
    disp('-------------------- HELP ------------------------');
    disp('INPUT:');
    disp('data:                  input array with shifted images');
    disp('annot_data:            annotations for the array of input images');
    disp('si[40]:                vertical size of the patch');
    disp('sj[40]:                horizontal size of the patch');
    disp('hbin[4]:               bin of the histogram of gradients');
    disp('todo[1 1 1]:           vector of execution flags:');
    disp('                             1** - create vlfeat HoG representation for every image');
    disp('                             *1* - train the regressors on the data');
    disp('                             **1 - test on training data');
    disp('sh_hor[0.3]:           shrinkage in horizontal direction');
    disp('sh_vert[0.3]:          shrinkage in vertical direction');
    disp('sh_sc[0.3]:            shrinkage in scale domain');
    disp('numIters[30000]:       number of iterations to be done for every regressor');
    disp('suff:                  suffix that is added to the name of the output file');
    disp(' ');
    disp('usage:                 train_reg_based_on_hog(data,annot_data,si,sj,hbin,todo,sh_hor,sh_vert,sh_sc,numIters,suff);');
    disp('example:               train_reg_based_on_hog(reg_data,reg_annot,40,40,5,[1 1 1],0.3,0.3,0.3,5000,''_plane'');');
    disp('------------------ HELP END -----------------------');
    return;
end


%% loading data

train_images = data;

if(nargin < 10)
    suff = '';
end

addpath('~/matlab_code/my_try/_suppl_functions');
addpath(genpath('~/matlab_code/vlfeat'));

if(~exist('train_images'))
    display('Error: Training data is not loaded properly: exiting');
    return;
end

if(~exist('annot_data'))
    display('Error: Annotation data is not loaded properly: exiting');
    return;
end

%% parameters

fldr = '/cvlabdata1/cvlab/forArtem/reg_tmp';
if(exist(fldr) == 0)
    mkdir(fldr);
end

if(exist(sprintf('%s/tmp_hog.mat',fldr)) ~= 0)
    load(sprintf('%s/tmp_hog.mat',fldr));
else
    traindata = [];
end

if(nargin < 6)||(isempty(todo))
    todo = [1 1 1];                             % 1** - create vlfeat HoG representation for every image
end                                             % *1* - train the regressors on the data
                                                % **1 - test on training data

sz = size(train_images,1);

if(nargin < 3)||(isempty(si))
    si = 40;
end
if(nargin < 4)||(isempty(sj))
    sj = 40;
end

if(nargin < 5)||(isempty(hbin))
    hbin = 4;
end

if(todo(1) == 1)

%% preparing the dataset

    time1 = tic;
    if(~isempty(traindata))
        st = size(traindata,1)+1;
    else
        st = 1;
    end

    count = 1;
    for i = st:sz
    % feature extraction
        im = reshape(train_images(i,:),si,sj);
        hog = vl_hog(im2single(im),hbin);
        traindata = [traindata; hog(:)'];
	count = count+1;
        if((toc(time1) > 10)&&(count >= 1000))
            clc; display(sprintf('HOG extraction: %d/%d(%0.2f%%) is finished',i,sz,100*i/sz));
            warning('check');
	    save(sprintf('%s/tmp_hog.mat',fldr),'traindata');
            if(~strcmp(lastwarn,'check'))
                save(sprintf('%s/tmp_hog.mat',fldr),'traindata','-v7.3');
            end
            time1 = tic;
	    count = 1;
        end
    end
end

save(sprintf('%s/tmp_hog.mat',fldr),'traindata');

%% clean data

fprintf('data cleanning ..\n');
i = 1;
time1 = tic;
while(i <= size(annot_data,1))
    if(isnan(annot_data(i,1)))||(isnan(annot_data(i,2)))
	traindata(i,:) = [];
	annot_data(i,:) = [];
    else
        i = i+1;
    end
    if(toc(time1) > 3)
        fprintf('checked %d/%d \n',i,size(annot_data,1));
        time1 = tic;
    end
end

if(todo(2) == 1)
%% training

    opts = [];
    opts.loss = 'squaredloss'; % can be logloss or exploss
% this has to be not too high (max 1.0)
    if(nargin < 7)||(isempty(sh_hor))
        opts.shrinkageFactor = 0.3;
	else
	opts.shrinkageFactor = sh_hor;
    end
    opts.subsamplingFactor = 0.2;
    opts.maxTreeDepth = uint32(2);  % this was the default before customization
    opts.randSeed = uint32(rand()*1000);

    if(nargin < 10)||(isempty(numIters))
        numIters = 30000;
    end

    tic;

    display('training horisonal regressor..');
    motion_regressor_hor = SQBMatrixTrain( single(traindata), annot_data(:,2), uint32(numIters), opts );

    if(nargin < 8)||(isempty(sh_vert))
        opts.shrinkageFactor = 0.3;
	else
	opts.shrinkageFactor = sh_vert;
    end

    display('training vertical regressor..');
    motion_regressor_vert = SQBMatrixTrain( single(traindata), annot_data(:,1), uint32(numIters), opts );

    if(size(annot_data,2) > 2)
        if(nargin < 7)||(isempty(sh_hor))
            opts.shrinkageFactor = 0.3;
	else
	    opts.shrinkageFactor = sh_sc;
        end
    	display('training scale regressor..');
    	motion_regressor_scale = SQBMatrixTrain( single(traindata), annot_data(:,3), uint32(numIters), opts );
    end

    time = toc;
    min_time = floor(time/60); 
    sec_time = time - min_time*60;
    fprintf('trained in %.0f minutes %.2f seconds \n',min_time,sec_time);

    if(size(annot_data,2) > 2)
    	save(sprintf('/home/rozantse/matlab_code/Regressors/motion_regressor_with_scale_%s%s.mat',date,suff),'motion_regressor_vert','motion_regressor_hor','motion_regressor_scale','si','sj','hbin');
    else
	save(sprintf('/home/rozantse/matlab_code/Regressors/motion_regressor_%s%s.mat',date,suff),'motion_regressor_vert','motion_regressor_hor','si','sj','hbin');
    end
end

%% test on training data

if(todo(3) == 1)
    time2 = tic;
    fprintf('testing on the training data ...........................................');
    pred_vert = SQBMatrixPredict( motion_regressor_vert, single(traindata));
    pred_hor = SQBMatrixPredict( motion_regressor_hor, single(traindata));
    if(size(annot_data,2) > 2)
    	pred_scale = SQBMatrixPredict( motion_regressor_scale, single(traindata));
    end

    if(size(annot_data,2) > 2)
    	score_on_td = sum((sqrt((annot_data - [pred_vert pred_hor pred_scale]).^2)));
    else
	score_on_td = sum((sqrt((annot_data - [pred_vert pred_hor]).^2)));
    end

    time = toc(time2);
    min_time = floor(time/60); 
    sec_time = time - min_time*60;
    fprintf('done in %.0f minutes %.2f seconds \n',min_time,sec_time);

    if(size(annot_data,2) > 2)
    	save(sprintf('/home/rozantse/matlab_code/Regressors/motion_regressor_with_scale_%s%s.mat',date,suff),'motion_regressor_vert','motion_regressor_hor','motion_regressor_scale','score_on_td','si','sj','hbin');
    else
	save(sprintf('/home/rozantse/matlab_code/Regressors/motion_regressor_%s%s.mat',date,suff),'motion_regressor_vert','motion_regressor_hor','score_on_td','si','sj','hbin');
    end
    display(sprintf('sum squared error: vert = %.02f, hor = %.02f',score_on_td(1),score_on_td(2)));

end
