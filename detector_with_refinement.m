% clc, clear
close all
%% ---- Parameters / Data Preparations ----
vid_type = 'rexp';
video_number = '1';
% --- Read in video sequence as 3-d array ---
VIDEO_SOURCE = VideoReader('D:\Proj\UAV\dataset\drones\Video_19.avi');
Iroi = read(VIDEO_SOURCE, [1, Inf]);

% thresh = 0.1;                                   %% if score is more then 0 then output result on the image

if(~exist('timestamp')||(~exist('stack_of_loc')))

    %% libraries and toolboxes
    addpath('./_supp_func/');
    %% paths to libraries that needs to be installed
%     addpath('~/path/to/SQB files/from/Carlos Becker (point 3 in README)');
    addpath('X:\UAV\toolbox\sqb-0.1');
    addpath(genpath('X:\UAV\toolbox\piotr-toolbox\matlab'));
    addpath(genpath('X:\UAV\toolbox\vlfeat-0.9.20'));

    load('./_mc_reg/drones/motion_regressor_10-Sep-2014.mat'); % Using pre-trained model for motion regression
    % reducing the number of trees to speed up (normal size is 30000 trees)
    motion_regressor_vert = motion_regressor_vert(1:5000);
    motion_regressor_hor = motion_regressor_hor(1:5000);
    
    save_flag = 1;
    start = 0;
    time_step = 0.5; % -- Time step for marking.
    ml_type = 'btr';
    
    switch(vid_type)
        case 'av'
            fixed_camera_params = 1;
            cam_motion_comp     = 0;
            todouble = 1;
        case 'rexp'
            todouble = 1;
            switch(video_number)
                case '47'
                    fixed_camera_params = 0;
                    cam_motion_comp     = 0;
                case '48'
                    fixed_camera_params = 1;
                    cam_motion_comp     = 1;
                case '49'
                    fixed_camera_params = 1;
                    cam_motion_comp     = 0;
                case '55'
                    fixed_camera_params = 1;
                    cam_motion_comp     = 1;
                otherwise
                    fixed_camera_params = 1;
                    cam_motion_comp     = 0;
            end
        otherwise
            fprintf('Error: Unknown video type: %s',vid_type);
            return;
    end
    timestamp = datestr(datevec(now()), 'dd_mm_yy-HH_MM_SS');
    data_name = strcat('refinement_for_detection_',video_number,'_',ml_type,'_',timestamp);

    iter = 3;
    prop = 0.25;
    sc_m_iter = 5;
    sc_r_iter = 5;
    sc_prop = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75];
    st = 4;
    si = 40;
    sj = 40;

%    thresh = 0.2;                                   %% if score is more then 0 then output result on the image
%    timestamp = datestr(datevec(now()), 'dd_mm_yy-HH_MM_SS');

    overlap = 0.8;
    max_det_overlap = 0.5;
    pnms = 3;                                       %% 4 pix non-max supression
    occupied_perc = 0.2;                           %% area which needs to be non-uniform to trigger deteciton process

    d_rate_col = 1;
    range_filt_threshold = 3;
    
    fprintf('Video number              : %s\n',video_number);
    fprintf('overlap on sliding window : %0.2f\n',overlap);
    fprintf('time step                 : %0.2f\n',time_step);
    fprintf('machine learning          : %s\n',ml_type);
    
    switch(vid_type)
        case 'av'
            
            switch(video_number)
                case '20'
                    scales = [20, 30, 40];
                otherwise
                    scales = [20, 30, 40, 50, 60, 80, 100, 120];
            end
            
            switch(ml_type)
                case 'btr';
                    btr_st = 4;
                    btr_si = 28;
                    btr_sj = 28;
                    st = btr_st;
                    load(sprintf('./_ml/planes/btr_CNN_mc_s%d_t%d.mat',btr_si,btr_st));
                    thresh = 0;
                otherwise
                    disp('not finished for planes yet');
                    return;
            end
        case 'rexp'
            scales = [20, 25, 30, 35, 40, 45, 50];
            switch(ml_type)
                case 'btr'
                    numcell = [2,2,2];
                    btr_st = 4;
                    btr_si = 40;
                    btr_sj = 40;
                    switch(btr_st)
                        case 4
                            load(sprintf('./_ml/drones/btr_CNN_mc_s%d_t4.mat',btr_si));      % 4 fr
                        case 7
                            load(sprintf('./_ml/drones/btr_CNN_mc_s%d_t7.mat',btr_si));      % 7 fr
                        case 11
                            load(sprintf('./_ml/drones/btr_CNN_mc_s%d_t11.mat',btr_si));     % 11 fr
                        otherwise
                            btr_st = 4;
                            load('./_ml/drones/btr_29-Jan-2015_CNN_dr_49_2500iter.mat');     % 4 fr
                    end
                    st = btr_st;
                    thresh = 0.5;
                otherwise
                    disp('Error: Unknown type of machine learning method: Exiting..');
                    return;
            end
        otherwise
            scales = [40];
    end
    
    ini_rate = scales(1)/si;
    rates = si./scales;

    stack_of_loc = struct([]);
else
    if(~isempty(stack_of_loc))
        start = stack_of_loc(end).t;
    end
end
    
data_name = strcat('refinement_for_detection_',video_number,'_',ml_type,'_',num2str(si),'_',num2str(st));


%% variables
[di,dj,len_vid] = size(Iroi);


% t = 34;

traj_ind = 1;

for t = start:ceil(st*time_step):(len_vid-st-1)
    fprintf('\nStarting at t = %d\n',t);
    clear block;
    block = Iroi(:,:,t+1:t+st);

    %% compensate for camera motion

    matrix = zeros(3,3,st);
    for i = 1:st
        matrix(:,:,i) = eye(3);
    end
    
    if(cam_motion_comp == 1)
        for ali_count = 1:floor(st/2)
            if(ali_count < st/2)
            	[temp,matrix(:,:,ceil(st/2)-ali_count)] = motion_from_im_feature(block(:,:,ceil(st/2)),block(:,:,ceil(st/2)-ali_count));
            	block(:,:,ceil(st/2)-ali_count) = temp;
            end
            [temp,matrix(:,:,ceil(st/2)+ali_count)] = motion_from_im_feature(block(:,:,ceil(st/2)),block(:,:,ceil(st/2)+ali_count));
	    block(:,:,ceil(st/2)+ali_count) = temp;
        end
        disp('Camera motion compensated     ->      [ok]');
    end


    if(fixed_camera_params == 0)
        temporary = double(block);
        for block_ct = 1:st
            temporary(:,:,block_ct) = temporary(:,:,block_ct)./255;
            temporary(:,:,block_ct) = temporary(:,:,block_ct).*0.6/mean(mean(temporary(:,:,block_ct)));
            temporary(:,:,block_ct) = temporary(:,:,block_ct).*255;
        end
        block = uint8(temporary);
    end
    
    %% filtering out uniform patches
    if(range_filt_threshold > 0)
        im = double(block(:,:,ceil(st/2)));
        [imx,imy] = gradient(gaussSmooth(im,[1 1],'smooth'));
        im = sqrt(imx.^2+imy.^2);
        mask = ones(di,dj);
            i = max(rates);
            h = fspecial('average',round(si/i));
            m = imfilter(im,h,'replicate');
            a = (range_filt_threshold < abs(im-m)+(im > 0)) > 0;
            h = fspecial('average',5);
            b = imfilter(a,h,'replicate');
            mask2 = zeros(size(im));
            mask2(b > 0) = 1;
            if(i == ini_rate)
                mask = mask2;
            end
            mask = mask.*mask2;
    end

    %% using previous step for speeding up the performance
    
    dif_thr = 0.10;
    
    if(t > ceil(st/2))
        im = Iroi(:,:,t-ceil(st/2));
        im2 = Iroi(:,:,t+ceil(st/2));
        [recovered, matr] = motion_from_im_feature( im2, im, 1);
        a = gaussSmooth(im2double(im2),[1 1],'smooth');
        b = gaussSmooth(im2double(recovered),[1 1],'smooth');
        c = histeq(b,imhist(a));
        dif = abs(c-a);
        dif(dif > dif_thr) = 1;
        dif(dif <= dif_thr) = 0;
        % using prev det
        try
            ind = find([stack_of_loc.t] == t-ceil(st/2));
            im = zeros(di,dj);
        for idx = 1:numel(ind)
            indx = ind(idx);
            strti = floor(max(stack_of_loc(indx).si,1));
            strtj = floor(max(stack_of_loc(indx).sj,1));
            psi = stack_of_loc(indx).fi - stack_of_loc(indx).si;
            psj = stack_of_loc(indx).fj - stack_of_loc(indx).sj;
            im(floor(strti+psi/4):floor(strti+3*psi/4),floor(strtj+psj/4):floor(strtj+3*psj/4)) = 1;
        end
        
        agt = estimateGeometricTransform;
        agt.OutputImagePositionSource = 'Property';
        [h, w] = size(im);
        agt.OutputImagePosition = [1 1 w h];
        im = step(agt, im2single(im), single(matr));
        im(im > 0.25)  = 1;
        im(im <= 0.25) = 0;
        catch
            im = ones(di,dj);
        end
        
        im = im(1:di,1:dj);
        
        pvf_msk = max(dif,im);
        pvf_msk = gaussSmooth(pvf_msk,[1 1],'smooth');
        pvf_msk(pvf_msk > 0) = 1;
        
        dmask = pvf_msk;
        mask = mask.*dmask;
    end
    
    %% prep for refinement
    positions = [];
	for crate = 1:numel(rates)
        sti = round(scales(crate)*overlap);
        stj = round(scales(crate)*overlap);
        coli = round((di-scales(crate))/sti);
        colj = round((dj-scales(crate))/stj);

        for ij = 1:coli*colj
            [posi,posj] = ind2sub([coli,colj],ij);
            ca = [(posi-1)*sti+1, (posj-1)*stj+1,(posi-1)*sti+scales(crate), (posj-1)*stj+scales(crate), rates(crate), ij];
            uni = mask(ca(1):ca(3),ca(2):ca(4));

            ind = find(uni > 0);
            if(~isempty(ind))
                [I,J] = ind2sub(size(uni),ind);
                mI = mean(I);
                mJ = mean(J);
                mvi = floor(mI-size(uni,1)/2);
                mvj = floor(mJ-size(uni,2)/2);
                if (ca(1)+mvi > 0)   && ...
                   (ca(2)+mvj > 0)   && ...
                   (ca(3)+mvi <= di) && ...
                   (ca(4)+mvj <= dj)
                    ca(1) = ca(1)+mvi;
                    ca(3) = ca(3)+mvi;
                    ca(2) = ca(2)+mvj;
                    ca(4) = ca(4)+mvj;
                    uni = mask(ca(1):ca(3),ca(2):ca(4));
                end
            end
            
            if(mean(uni(:)) < occupied_perc)
                continue;
            end
            
%             im = block(:,:,ceil(st/2));
%             im = repmat(im,[1 1 3]);
%             im(:,:,2) = mask.*255;
%             imshow(im);
%             rectangle('position',[ca(2),ca(1),ca(4)-ca(2)+1,ca(3)-ca(1)+1]);
%             waitforbuttonpress;
            
            positions = [positions; ca];
        end
    end
    
    if(isempty(positions))
        fprintf('No positions to check found at the curent frame.\n Proceeding to the next one.\n');
%         continue;
    end
    
    %% refinement
    rs = {};
    
    [annot] = refine_pos_v1( [positions,ones(size(positions,1),1)], block(:,:,ceil(st/2)), motion_regressor_vert , motion_regressor_hor, iter, si, sj);
    rs{ceil(st/2)} = annot(:,1:6);
    
%   [~,ind,~] = unique([round(rs{ceil(st/2)}(:,1:4)/pnms), rs{ceil(st/2)}(:,5)],'rows');
    [~,ind,~] = unique([round(rs{ceil(st/2)}(:,1:4)/pnms)],'rows');
    rs{ceil(st/2)} = rs{ceil(st/2)}(ind,:);
   
    [~,ind] = sort(rs{ceil(st/2)}(:,5));
    rs{ceil(st/2)} = rs{ceil(st/2)}(ind,:);
    for i = 1:st
        rs{i} = rs{ceil(st/2)};
    end
    
    rs2 = rs;
    
    for i = 1:ceil(st/2)-1
        
        rs{ceil(st/2)-i} = rs{ceil(st/2)-i+1};
        rs{ceil(st/2)+i} = rs{ceil(st/2)+i-1};
        
        annot = [rs{ceil(st/2)-i}, (ceil(st/2)-i)*ones(size(rs{ceil(st/2)},1),1); ...
                 rs{ceil(st/2)+i}, (ceil(st/2)+i)*ones(size(rs{ceil(st/2)},1),1);];
         
        [new_annot] = refine_pos_v1( annot, block, motion_regressor_vert, motion_regressor_hor, iter, si, sj);
        
        ind = find(round(new_annot(:,7)) == (ceil(st/2)-i));
        rs{ceil(st/2)-i} = new_annot(ind,1:6);
        ind = find(round(new_annot(:,7)) == (ceil(st/2)+i));
        rs{ceil(st/2)+i} = new_annot(ind,1:6);
    end
    
    rs4 = rs;
    
    disp('\nMotion of objects compensated     ->      [ok]');
    
    %% smoothing the detection
    
    if(st > 5)&&(cam_motion_comp == 1)
    
        for i = 1:size(rs{1},1)
            traj = [];
            for j = 1:st
                traj = [traj; [rs{j}(i,1),rs{j}(i,2),rs{j}(i,3),rs{j}(i,4)]];
            end
            
            traj = gaussSmooth(traj,[1 0],'smooth');
            for j = 1:st
                rs{j}(i,1) = traj(j,1);
                rs{j}(i,2) = traj(j,2);
                rs{j}(i,3) = traj(j,3);
                rs{j}(i,4) = traj(j,4);
            end
        end
        
        for_nms = [];
        for j = 1:st
        	for_nms = [for_nms,floor(rs{j}(:,1:4)/5)];
        end
        
        [~,ind,~] = unique(for_nms,'rows');
        for j = 1:st
            rs{j} = rs{j}(ind,:);
        end
    end
    %% extract data for testing
    det = zeros(si,sj,st);
    tst_data = [];

    i = 1;
    while( i <= size(rs{1},1))
        try
            for j = 1:st
                ca = round(rs{j}(i,1:4));
                det(:,:,j) = imresize(block(ca(1)+1:ca(3),ca(2)+1:ca(4),j),[si,sj],'bilinear');
            end
            tst_data = [tst_data; uint8(det(:)')];
            i = i+1;
        catch
            for j = 1:st
                rs{j}(i,:) = [];
            end
        end
    end
    nsamples = size(tst_data,1);

%% machine learning
    
    switch(ml_type)            
        case 'btr'
            if(numel(tst_data) > 0)
                tst_data = reshape(tst_data,[],si,sj,st);
                tst_data = reshape(permute(tst_data,[2 3 4 1]),si,sj,[]);
                tst_data = imresize(tst_data,[btr_si,btr_sj],'bilinear');
                tst_data = reshape(reshape(tst_data,btr_si,btr_sj,btr_st,[]),btr_si*btr_sj*btr_st,[])';
                
                tmp = myhog3d(reshape(tst_data(1,:),btr_si,btr_sj,btr_st),rsize,numcell,1,0);
                if(todouble == 1)&&(max(tst_data(:)) > 1)
                    tst_data = double(tst_data)/255;
                end
                ml_data = zeros(nsamples,numel(tmp));
        
                for i = 1:nsamples
                    ml_data(i,:) = myhog3d(reshape(tst_data(i,:),btr_si,btr_sj,btr_st),rsize,numcell,1,0);
                end
                decision_values = SQBMatrixPredict( model, single(ml_data) );
                decision_values = (decision_values+1)/2;
            else
                decision_values = [];
            end
        otherwise
            disp('Unknown machine learning technique')
            
    end

    %% clean repeating detections
    
    ind = find(decision_values > thresh);
    mfa = rs{ceil(st/2)}(ind,:);
    
    for ij = 1:numel(ind)
       traj_ind = traj_ind+1;
       for j = 1:st
            i = ind(ij);

            if(cam_motion_comp == 1)
                sz_i = rs{j}(i,3)-rs{j}(i,1);
                sz_j = rs{j}(i,4)-rs{j}(i,2);

                e = [rs{j}(i,2),rs{j}(i,1),1]/matrix(:,:,j);
                rs{j}(i,2) = floor(e(1)/e(3));
                rs{j}(i,1) = floor(e(2)/e(3));
                rs{j}(i,3) = rs{j}(i,1) + sz_i;
                rs{j}(i,4) = rs{j}(i,2) + sz_j;
            end
            stack_of_loc = [stack_of_loc; struct('t',t+j,'si',rs{j}(i,1),'sj',rs{j}(i,2),'fi',rs{j}(i,3),'fj',rs{j}(i,4),'score',decision_values(i),'det_count',traj_ind)];
        end
    end

    if(save_flag == 1) 
        switch(ml_type)
            case 'btr'
                 save(sprintf('./results/_vid_%s_%s_stack_of_loc_sc_%d_%s_%s_%d_%d.mat',video_number,vid_type,numel(scales),timestamp,ml_type,btr_si,btr_st),'stack_of_loc','st','si','sj','btr_si','btr_sj');
        end
    end
end

fprintf('TERMINATED NORMALLY\n');
