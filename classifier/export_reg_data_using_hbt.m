%% inputs

db_path = '/path/to/the/root/folder/';         % <-- modify this
object = 'drones';
data2gen = 'pos';

%% parameters

switch(object)
    case 'planes'
        videos = {'9','11','13','14','16','17'};
        vid_type = 'av';
        model_name = 'planes_40_pp_0.25';
        ref_model_name = 'planes_40_md_0.10';
        fprintf('Running for planes..\n\n'); 
        di = 540;
        dj = 720; 
    case 'drones'
        videos = {'48','49','53'};
        vid_type = 'rexp';
        model_name = 'drones_40_pp_0.25';
        ref_model_name = 'drones_40_md_0.15';
        fprintf('Running for drones..\n\n'); 
        di = 480;
        dj = 752; 
    otherwise
        disp('Error: Unknown object: exiting..');
        return;
end

switch(data2gen)
    case 'pos'
        pos_flag = 1;
    case 'neg'
        pos_flag = 0;
    otherwise
        disp('Error: Unknown type of data: exiting..');
        return;
end

si = 40;
sj = 40;
st = 4;
neg_per_frame = 30;

extr_data = [];

%% prepare annotations for negative examples
% renerate random indexes
annot = [];
for v_num = videos
% loading videos 1 by 1
    video_number = char(v_num);
    switch(vid_type)
        case 'av'
            annot_path = fullfile(db_path,'annotations','planes',sprintf('Video_%s.txt',video_number));
        case 'rexp'
            annot_path = fullfile(db_path,'annotations','drones',sprintf('Video_%s.txt',video_number));
        otherwise
            fprintf('Error: video type is not correct: finishing..');
    	return;
    end
    % read annot file
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
            a = reshape(a,4,[])';
            for i = 1:size(a,1)
                annot = [annot; [timestamp, a(i,1), a(i,2), a(i,3), a(i,4), si/(a(i,3)-a(i,1)), str2double(video_number)]];
                nfr_count = nfr_count+1;
            end
        end
        tline = fgets(fid);
    end
    fclose(fid);
        
    fprintf('annotations for video %s are imported..\n',video_number);
        
    if(pos_flag == 0)
        neg_annot = [];
        
        for i = 1:size(annot,1)
            for ii = 1:neg_per_frame
                while(1)
                    sc = normrnd(1.5,0.5);
                    sz = round(si/sc);
                    sti = round(rand(1)*(di-sz*1.5)+0.25*sz+1);
                    fini = sti+sz;
                    stj = round(rand(1)*(dj-sz*1.5)+0.25*sz+1);
                    finj = stj+sz;
                    if(((annot(i,2) < fini) || (annot(i,4) < sti) || (annot(i,3) > finj) || (annot(i,5) < stj))) && ...
                      (((sti-0.25*sz-1 > 0)   && (stj-0.25*sz-1 > 0)  && (fini+0.25*sz+1 < di) && (finj+0.25*sz+1 < dj))) && ...
                      (sc > 0.5) && (sc < 4)
                        neg_annot = [neg_annot; [annot(i,1),sti,stj,fini,finj,si/sz,annot(i,7)]];
                        break;
                    end
                end
            end
        end
        fprintf('video %s is finished ..\n\n',video_number);
    end
end

%% prepare the annotation data for frame from the spatio-temporal cube:

if(pos_flag == 1)
    [resulting_annot{2}, original_annot] = test_on_data_from_file_HBT(videos, vid_type, db_path, model_name, ref_model_name, 2, 0, annot);
else
    [resulting_annot{2}, original_annot] = test_on_data_from_file_HBT(videos, vid_type, db_path, model_name, ref_model_name, 2, 0, neg_annot);
end
[resulting_annot{1}, ~] = test_on_data_from_file_HBT(videos, vid_type, db_path, model_name, ref_model_name, 2, -1,resulting_annot{2});
[resulting_annot{3}, ~] = test_on_data_from_file_HBT(videos, vid_type, db_path, model_name, ref_model_name, 2, 1,resulting_annot{2});
[resulting_annot{4}, ~] = test_on_data_from_file_HBT(videos, vid_type, db_path, model_name, ref_model_name, 2, 2,resulting_annot{2});

%% putting the results together
rs = resulting_annot;

%% saving the results of the motion compensation

switch(vid_type)
    case 'av'
        fnam = 'plane_';
    case 'rexp'
        fnam = 'drone_';
    otherwise
        disp('Error: unknown video type: please check the initial parameters');
end

%% here you need to put the path to the folder where resultsing data will be stored
fnam = strcat(fnam,'CNN_mc');

a = char(videos);
b = a';
fnam = strcat(fnam,'_',data2gen);
fnam = strcat(fnam,'_',b(:)');
fnam = strcat(fnam,'.mat');

fname = fullfile(db_path,fnam);

save(fname,'resulting_annot','rs');

%% combine these cubes into training and testing data

prev_ind = Inf;
vid_counter = 0;

for i = 1:size(rs{1},1)
    if(rs{1}(i,1) < prev_ind)
        vid_counter = vid_counter+1;
        video_number = char(videos{vid_counter});
        switch(vid_type)
            case 'av'
                vid_path = fullfile(db_path,'videos','planes','mat',sprintf('video_%s.mat',video_number));
            case 'rexp'
                vid_path = fullfile(db_path,'videos','drones','mat',sprintf('video_%s.mat',video_number));
            otherwise
                fprintf('Error: video type is not correct: finishing.');
        end
        load(vid_path);
    end
    
    stcube = uint8(zeros(si,sj,st));
    for j = 1:st
        ca = round(reshape(rs{j}(i,:),1,[]));
        im = Iroi(:,:,ca(1));
        det = im(ca(2):ca(4),ca(3):ca(5));
        det = imresize(det, [si sj],'bilinear');
        stcube(:,:,j) = uint8(det);
    end
    extr_data = [extr_data; stcube(:)'];
    prev_ind = rs{1}(i,1);
end

%% export in the appropriate format

save(fname,'resulting_annot','extr_data','rs');