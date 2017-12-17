function [resulting_annot, original_annot, ground_truth] = test_on_data_from_file_HBT(videos, vid_type, db_path, model_name, ref_model_name, save_flag, rand_ann, res_annot, m_iter, r_iter, Iroi)
    
%% parametes
% db_path -- path to the data folder

addpath(genpath('/path/to/vlfeat/'));           % <-- modify this
addpath('/path/to/Gradient/Boosted/Trees/');    % <-- modify this
pth_code = '/path/to/the/root/folder/';         % <-- modify this
hbin = 4;

%% other parameters

if((nargin < 1)||isempty(videos))
    fprintf('\n\nusage:        resulting_annot = test_on_data_from_file(videos, vid_type, db_path, model_name, ref_model_name, save_flag)\n\n');
    fprintf('\nRunning with the default parameters:\n\n');
    %%%% parameters for planes
    % videos = {'14','16','17'};
    %%% parameters for drones
    videos = {'49'};
end
v = char(videos);
fprintf('Video numbers            : %s\n', v');
if((nargin < 2)||isempty(vid_type))
    %%%% parameters for planes
%     vid_type = 'av';
    %%% parameters for drones
    vid_type = 'rexp';
    
end
fprintf('Video type               : %s\n', vid_type);

switch(vid_type)
    case 'rexp'
        reg_path = fullfile(pth_code,'code_cvpr15','_mc_reg','drones','motion_regressor_08-Sep-2014_v1.mat');
        load(reg_path);
        motion_regressor_hor  = motion_regressor_hor(1:5000);
        motion_regressor_vert = motion_regressor_vert(1:5000);
    case 'av'
        reg_path = fullfile(pth_code,'code_cvpr15','_mc_reg','planes','motion_regressor_30-Oct-2014_plane_all.mat');
        load(reg_path);
        motion_regressor_hor  = motion_regressor_hor(1:5000);
        motion_regressor_vert = motion_regressor_vert(1:5000);
end

if((nargin < 3)||isempty(db_path))
    db_path = pth_code;
end
fprintf('Path to data             : %s\n', db_path);
if((nargin < 4)||isempty(model_name))
    %%%% parameters for planes
    % model_name = 'planes_40_pp_0.25';
    %%% parameters for drones
    model_name = 'drones_40_pp_0.25';
end
fprintf('Model Name               : %s\n', model_name);
if((nargin < 5)||isempty(ref_model_name))
    %%%% parameters for planes
    % ref_model_name = 'planes_40_md_0.10';
    %%% parameters for drones
    ref_model_name = 'drones_40_md_0.15';
end
fprintf('Refinement Model Name    : %s\n', ref_model_name);
fprintf('\n');
if((nargin < 6)||isempty(save_flag))
    save_flag = 0;
end
fprintf('Save flag                : %d\n', save_flag);
if((nargin < 7)||isempty(rand_ann))
    rand_ann = [];
end
if(isempty(rand_ann))
    fprintf('Random init annotations  : Yes\n');
else
    fprintf('Random init annotations  : No: picking value from %d position\n',rand_ann);
end

if((nargin < 8)||(isempty(res_annot)))
    fprintf('Use provided annotations : No: Read from files\n');
    read_flag = 1;
else
    read_flag = 0;
    fprintf('Use provided annotations : Yes\n');
end


fprintf('\nOutput variable          : resulting_annot\n\n');

%% outside params

run_path = pwd;

cd(run_path);

if(read_flag == 0)
    prev = 0;
    annot_counter = 1;
end


main_model_name = model_name;
resulting_annot = [];           % collection of the resulting annotation data from all the videos
original_annot = [];            % collection of the original annotation data from all the videos

if(save_flag == 1)
    if(numel(videos) > 1)
        wr = VideoWriter(strcat('testing_on_',vid_type,'_videos.avi'));
    else
        video_number = char(videos);
        wr = VideoWriter(strcat('testing_on_',vid_type,'_video_',video_number,'.avi'));
    end
    wr.FrameRate = 4;
    wr.open();
end

for v_num = videos
%% loading videos 1 by 1
video_number = char(v_num);

switch(vid_type)
    case 'av'
        vid_path = fullfile(db_path,'videos','planes','mat',sprintf('video_%s.mat',video_number));
        annot_path = fullfile(db_path,'annotations','planes',sprintf('Video_%s.txt',video_number));
    case 'rexp'
        vid_path = fullfile(db_path,'videos','drones','mat',sprintf('video_%s.mat',video_number));
        annot_path = fullfile(db_path,'annotations','drones',sprintf('Video_%s.txt',video_number));
    otherwise
        fprintf('Error: video type is not correct: finishing.');
        return;
end

%% printing paths

fprintf('--- READING -----------------\n');
fprintf(' Video from       : %s\n', vid_path);
fprintf(' Annotations from : %s\n', annot_path);
fprintf('-----------------------------\n');

%% detection parameters

prop = 0.25;
msz = 40;
szi = msz;
szj = msz;
max_dist = 0.5;
if(nargin < 9) || isempty(m_iter)
    m_iter = 3;
end
if(nargin < 10) || isempty(r_iter)
    r_iter = 5;
end
eps = 0.001;

if(isempty(rand_ann))
    nd_pf = 2;      % number of distortions per frame of the video
else
    nd_pf = 1;
end

data_name = strcat('video_',video_number,'_res');
if(prop > 0)
    data_name = sprintf('%s_pp_%.02f',data_name,prop);
end

%% loading data
if (nargin < 11)||(numel(videos) > 1)||(isempty(Iroi))
    load(vid_path);
end

%% counting annotated frames in the video
clear annot;

if(read_flag == 1)
    fid = fopen(annot_path);
    tline = fgets(fid);
    tline = strtrim(tline);
    nfr_count = 1;
    while(tline ~= -1)
        tline = strtrim(tline);
        if(tline(1) ~= '#')
            tline = tline(13:end);
            timestamp = sscanf(tline,'%d',[1 Inf]);
            tline = tline(14+numel(num2str(timestamp)):end);
            a = sscanf(tline,'(%d,%d,%d,%d), ',[1 Inf]);
            for i = 1:size(a,1)
                annot(nfr_count:nfr_count+nd_pf-1,1) = timestamp;
                annot(nfr_count:nfr_count+nd_pf-1,2) = a(i,1);
                annot(nfr_count:nfr_count+nd_pf-1,3) = a(i,2);
                annot(nfr_count:nfr_count+nd_pf-1,4) = a(i,3);
                annot(nfr_count:nfr_count+nd_pf-1,5) = a(i,4);
                annot(nfr_count:nfr_count+nd_pf-1,6) = msz/(a(i,3)-a(i,1));
                annot(nfr_count:nfr_count+nd_pf-1,7) = str2double(video_number);
                nfr_count = nfr_count+nd_pf;
            end
        end
        tline = fgets(fid);
    end
    fclose(fid);
else
    ii = 1;
    while(prev <= res_annot(annot_counter,1))
        annot(ii,:) = res_annot(annot_counter,:);
        ii = ii+1;
        prev = res_annot(annot_counter,1);
        annot_counter = annot_counter+1;
        if(annot_counter > size(res_annot,1))
            break;
        end
    end
    prev = 0;
end

original_annot = [original_annot; annot];
if(read_flag == 1)
    ground_truth = original_annot;
else
    ground_truth = -1;
end

%% generate random shifts for each detection

if(isempty(rand_ann))
    di = zeros(size(annot,1),1);
    dj = zeros(size(annot,1),1);
    for i = 1:size(annot,1)
        di = round(rand()*szi*max_dist-szi*max_dist/2);
        dj = round(rand()*szj*max_dist-szj*max_dist/2);
        annot(i,2) = annot(i,2)+round(di/annot(i,6));
        annot(i,4) = annot(i,2)+round(msz/annot(i,6));
        annot(i,3) = annot(i,3)+round(dj/annot(i,6));
        annot(i,5) = annot(i,3)+round(msz/annot(i,6));
    end
else
    annot(:,1) = annot(:,1)+rand_ann;
    ind = find(annot(:,1) <= 0);
    annot(ind,:) = [];
    ind = find(annot(:,1) > size(Iroi,3));
    annot(ind,:) = [];
end

dannot = annot;         % saving initial annotations
original_annot = dannot;

%% main iterations

for iter = 1:(m_iter+r_iter)

prop = 0;
    
% building test_x and test_y data from the video: Iterative
clear test_x test_y;

i = 0;
dispstat('','init');
time1 = tic;

indexes = [];

test_x = [];
test_y = [];

while i < size(annot,1)
    
    i = i+1;
    im = Iroi(:,:,annot(i,1));
    ca = annot(i,:);
    im2 = imresize(im,ca(6),'bilinear');
    ca(2:5) = round(ca(2:5).*ca(6));
    
    ca(2) = ca(2)-msz*prop;
    ca(4) = ca(2)+round(msz*(1+2*prop));
    ca(3) = ca(3)-msz*prop;
    ca(5) = ca(3)+round(msz*(1+2*prop));
    
	if ca(2) < 0
        ca(2) = 0;
        ca(4) = round(msz*(1+2*prop));
	end
	if ca(3) < 0
        ca(3) = 0;
        ca(5) = round(msz*(1+2*prop));
	end
	if ca(4) >= size(im2,1)
        ca(2) = size(im2,1)-round(msz*(1+2*prop));
        ca(4) = size(im2,1);
	end
	if ca(5) >= size(im2,2)
        ca(3) = size(im2,2)-round(msz*(1+2*prop));
        ca(5) = size(im2,2);
	end
    
    try
        detection = im2(ca(2)+1:ca(4),ca(3)+1:ca(5));
        indexes = [indexes; i];
    catch
        fprintf('the aircraft is too close to the boundary of the image to extract the needeed area skipping\n');
        continue;
    end
    detection = single(detection);
    detection = detection-min(detection(:));
    detection = detection./(max(detection(:))+eps);
    
	test_x = [test_x, detection(:)];
    test_y = [test_y, [rand(1);rand(1)]];          % this line is not important - so filling up with random variables
    
    if(toc(time1) > 5)
        dispstat(sprintf('%.02f%% finished..',100*i/size(annot,1)));
        time1 = tic;
    end
end

%% running HBT on data

res = [];

fprintf('\nprediction  ');
si = 40;
sj = 40;
time2 = tic;

dispstat('','init');

for i = 1:size(test_x,2)
    cut_im = reshape(test_x(:,i),si,sj);
    vhog = vl_hog(im2single(imresize(cut_im,[si,sj],'bilinear')),hbin);
    pred_vert = SQBMatrixPredict( motion_regressor_vert, (vhog(:)'));
    pred_hor = SQBMatrixPredict( motion_regressor_hor, (vhog(:)'));
    res = [res, [pred_vert; pred_hor]];
    if(toc(time2) > 5)
        dispstat(sprintf(' %.02f%% finished..',100*i/size(test_x,2)));
        time2 = tic;
    end
%     if(mod(i,200) == 0)
%         fprintf('.');
%         time2 = tic;
%     end
end
fprintf('  [done]\n');

res = reshape(res,2,[]);

oannot = annot;
nannot = annot;

for i = 1:numel(indexes)
    di = round(res(1,i)*msz*(1+2*prop));
    dj = round(res(2,i)*msz*(1+2*prop));
    
    nannot(indexes(i),2) = min(max(annot(indexes(i),2)-round(di/annot(indexes(i),6)),prop*round(msz/annot(indexes(i),6))+1),size(Iroi,1)-(1+prop)*round(msz/annot(indexes(i),6)));
    nannot(indexes(i),4) = nannot(indexes(i),2)+round(msz/annot(indexes(i),6));
    nannot(indexes(i),3) = min(max(annot(indexes(i),3)-round(dj/annot(indexes(i),6)),prop*round(msz/annot(indexes(i),6))+1),size(Iroi,2)-(1+prop)*round(msz/annot(indexes(i),6)));
    nannot(indexes(i),5) = nannot(indexes(i),3)+round(msz/annot(indexes(i),6));
end
annot = nannot;

if(iter == m_iter)
    prop = 0;
    model_name = ref_model_name;
end

end

%% show data

if(save_flag == 0)
%     for i = 1:size(annot,1)
%         im = Iroi(:,:,annot(i,1));
%         det = im(dannot(i,2):dannot(i,4),dannot(i,3):dannot(i,5));
%         det = single(det);
%         det = det-min(det(:));
%         det = det./(max(det(:))+eps);
%         det = imresize(det,[40 40],'bilinear');
%     
%         comp_det = im(nannot(i,2):nannot(i,4),nannot(i,3):nannot(i,5));
%         comp_det = single(comp_det);
%         comp_det = comp_det-min(comp_det(:));
%         comp_det = comp_det./(max(comp_det(:))+eps);
%         comp_det = imresize(comp_det,[40 40],'bilinear');
%     
%         imshow([det comp_det]);
%         w = waitforbuttonpress;
%         if(w ~= 0)
%             close all;
%             break;
%         end
%     end
elseif(save_flag == 1)
    for i = 1:size(annot,1)
        im = Iroi(:,:,annot(i,1));
        det = im(dannot(i,2):dannot(i,4),dannot(i,3):dannot(i,5));
        det = single(det);
        det = det-min(det(:));
        det = det./(max(det(:))+eps);
        det = imresize(det,[40 40],'bilinear');
    
        comp_det = im(nannot(i,2):nannot(i,4),nannot(i,3):nannot(i,5));
        comp_det = single(comp_det);
        comp_det = comp_det-min(comp_det(:));
        comp_det = comp_det./(max(comp_det(:))+eps);
        comp_det = imresize(comp_det,[40 40],'bilinear');
        
        % output
        
        difftext = imresize(Other_MEM_Text2Im('Results with iterative CNN',10,10),1/2,'bilinear');
        imtext_dist = imresize(Other_MEM_Text2Im(' distorted ',10,10),1/2,'bilinear');
        imtext_comp = imresize(Other_MEM_Text2Im('compensated',10,10),1/2,'bilinear');
        difftext    = repmat(1 - difftext,[1 1 3]);
        imtext_dist = repmat(1 - imtext_dist,[1 1 3]);
        imtext_comp = repmat(1 - imtext_comp,[1 1 3]);

        ctemp = repmat(det,[1,1,3]);
        cout = repmat(comp_det,[1,1,3]);
        oz1 = round((430-size(difftext,2))/2);
        oz2 = 430-oz1-size(difftext,2);

        im = [zeros(10,10,3),zeros(10,200,3),zeros(10,10,3),zeros(10,200,3),zeros(10,10,3); ...
              zeros(10,oz1,3),difftext,zeros(10,oz2,3); ...
              zeros(10,10,3),zeros(10,200,3),zeros(10,10,3),zeros(10,200,3),zeros(10,10,3); ...
              zeros(200,10,3),imresize(ctemp,[200,200],'nearest'),zeros(200,10,3),imresize(cout,[200,200],'nearest'),zeros(200,10,3); ...
              zeros(10,10,3),zeros(10,200,3),zeros(10,10,3),zeros(10,200,3),zeros(10,10,3); ...
              zeros(10,10,3),zeros(10,51,3),imtext_dist,zeros(10,50,3),zeros(10,10,3),zeros(10,50,3),imtext_comp,zeros(10,51,3),zeros(10,10,3); ...
              zeros(10,10,3),zeros(10,200,3),zeros(10,10,3),zeros(10,200,3),zeros(10,10,3); ...
              ];

        writeVideo(wr,im);   
    end
end

resulting_annot = [resulting_annot; nannot];
switch(vid_type)
    case 'av'
        save(sprintf('annot_after_CNN_ref_drone_v%s.mat',video_number),'resulting_annot');
    case 'rexp'
        save(sprintf('annot_after_CNN_ref_plane_v%s.mat',video_number),'resulting_annot');
end

end

if(save_flag == 1)
    close(wr);
end
