function [data,annot_data] = get_reg_data_from_annot_Iroi(Iroi,video_number,si,sj,spf,video_path,fin,save_flag,add_original,prop)

% this function extracts data from the video sequence for training the regression model

if(nargin < 1)
    disp('-------------------- HELP --------------------');
    disp(' ');
    disp('Iroi:             video sequence in the 3D aray format');
    disp('video_number:     number of the video, which is going to be processed');
    disp('si[40]:           size of the detection in pixels (vertical)');
    disp('sj[40]:           size of the detection in pixels (horizontal)');
    disp('spf[10]:          number of samples for each detection');
    disp('video_path:       ''drones'' or ''planes''');
    disp('fin[Inf]:         last frame to be processed');
    disp('save_flag[1]:     write results to file');
    disp('add_original[0]:  add original bounding box as a separate channel to extracted data');
    disp('prop[0]:          extension of the original size of the patch by this proportion');
    disp(' ');
    disp('------------------ END HELP ------------------');
    return;
end
%% parameters
if(nargin < 3)||(isempty(si))
    si = 40;
end
if(nargin < 4)||(isempty(sj))
    sj = 40;
end

if(nargin < 5)||(isempty(spf))
    spf = 10;                           % 10 randmly generated sample from each detection
end

if(nargin < 6)||(isempty(video_path))
    video_path = 'drones';
end

if(nargin < 7)||(isempty(fin))
    fin = Inf;
end

if(nargin < 8)||(isempty(save_flag))
    save_flag = 1;
end

if(nargin < 9)||(isempty(add_original))
    add_original = 0;
end

if(nargin < 10)||(isempty(prop))
    prop = 0;
end

data = [];
annot_data = [];

[gx,gy,gt] = size(Iroi);

freq = 2;                               % every # of frames to do the extraction

scale_thr = 0;
max_scale = 0;

max_dist = 0.50;

max_dist = max_dist./(1+2*prop);

%% additional variables

fannot = fopen(sprintf('../../videos/%s/Video_%s.txt',video_path,video_number),'r');

annot_max = 0;
tline = fgets(fannot);
while(tline ~= -1)
    annot_max = annot_max+1;
    tline = fgets(fannot);
end
fclose(fannot);

fannot = fopen(sprintf('../../annotations/%s/Video_%s.txt',video_path,video_number),'r');

%% extraction

lst = min(annot_max,fin);

tline = fgets(fannot);

count = 0;

for t = 1:freq:lst
    
% getting data
    
%     if(isempty(tline))
%         display('annotation file is finished');
%         break;
%     end
    if(tline == -1)
        display('annotation file is finished');
        break;
    end
    
%     clc; display(tline);
    
    tline = tline(1,13:end);
    
    tt = sscanf(tline,'%d')';

    tline = tline(1,numel(num2str(tt))+14:end);
    a = sscanf(tline,'(%d, %d, %d, %d), ',[4 Inf])';
    
    ra = a;

    if(~isempty(a))
        for j = 1:size(a,1)
                a = ra;
                rszi = a(j,3)-a(j,1);
                rszj = a(j,4)-a(j,2);

		ra(j,1) = ra(j,1) - round(prop*rszi);
		ra(j,2) = ra(j,2) - round(prop*rszj);
		ra(j,3) = ra(j,3) + round(prop*rszi);
		ra(j,4) = ra(j,4) + round(prop*rszj);
		
            for i = 1:spf
                a = ra;
		
                szi = rszi;
                szj = rszj;
		
                ext = round((rand()*scale_thr)*(2*max_scale)-scale_thr*max_scale);
                if(scale_thr > 0)
                    exti = ext*rszi/scale_thr;
                    extj = ext*rszj/scale_thr;
                else
                    exti = 0;
                    extj = 0;
                end
                a(j,1) = max(a(j,1)-exti,1);
                a(j,2) = max(a(j,2)-extj,1);
                a(j,3) = min(a(j,3)+exti,gx);
                a(j,4) = min(a(j,4)+extj,gy);
                szi = a(j,3)-a(j,1);
                szj = a(j,4)-a(j,2);
                
                di = round(rand()*szi*max_dist-szi*max_dist/2);
                dj = round(rand()*szj*max_dist-szj*max_dist/2);
                
                iter = 0;
                
                while((a(j,4)+dj > gy)||(a(j,3)+di > gx)||(a(j,2)+dj < 1)||(a(j,1)+di < 1))&&(iter < 100)
                    di = round(rand()*szi*max_dist-szi*max_dist/2);
                    dj = round(rand()*szj*max_dist-szj*max_dist/2);
                    ext = round((rand()*scale_thr)*(2*max_scale)-scale_thr*max_scale);
                    exti = ext*rszi/scale_thr;
                    extj = ext*rszj/scale_thr;
                    iter = iter+1;
                end
                
                
		try
%		imargin = round(prop*szi);
%		jmargin = round(prop*szj);
		imargin = round(prop*si);
		jmargin = round(prop*sj);
                if(add_original == 1)
                    data = [data; reshape(imresize(Iroi(a(j,1):a(j,3),a(j,2):a(j,4),t),[si+2*imargin,sj+2*jmargin],'bilinear'),1,[]), ...
                                  reshape(imresize(Iroi(a(j,1)+di:a(j,3)+di,a(j,2)+dj:a(j,4)+dj,t),[si+2*imargin,sj+2*jmargin],'bilinear'),1,[])];
                else
                    data = [data; reshape(imresize(Iroi(a(j,1)+di:a(j,3)+di,a(j,2)+dj:a(j,4)+dj,t),[si+2*imargin,sj+2*jmargin],'bilinear'),1,[])];
                end
		if(scale_thr > 0)
                    annot_data = [annot_data; [di/szi dj/szj ext/scale_thr]];
                else
                    annot_data = [annot_data; [di/szi dj/szj]];
                end
		
        catch
            disp('some problem occured, presumably due to the image boundaries');
            continue;
        end
%                display(sprintf('a(%d,1) = %d, a(%d,2) = %d, a(%d,3) = %d, a(%d,4) = %d, di = %d, dj = %d, szi = %d, szj = %d',j,a(j,1),j,a(j,2),j,a(j,3),j,a(j,4),di,dj,szi,szj));
        end
        end
    end
    
    for i = 1:freq
        tline = fgets(fannot);
    end
    
    clc; display(sprintf('%d/%d(%.02f%%) is finished', t,lst,t/lst*100)); 

    count = count+1;
    if(save_flag == 1)&&(count == 200)
%        save(sprintf('./%s/data_for_regression_with_scale.mat',video_path),'data','annot_data');
    	if(add_original == 1)
            save(sprintf('./%s/data_for_regression_%d_pp_%.02f_ori.mat',video_path,si,prop),'data','annot_data');
    	else
            save(sprintf('./%s/data_for_regression_%d_pp_%.02f.mat',video_path,si,prop),'data','annot_data');
    	end
	count = 0;
    end
end

fclose(fannot);

if(save_flag == 1)
%        save(sprintf('./%s/data_for_regression_with_scale.mat',video_path),'data','annot_data');
    if(add_original == 1)
        save(sprintf('./%s/data_for_regression_%d_pp_%.02f_ori.mat',video_path,si,prop),'data','annot_data');
    else
        save(sprintf('./%s/data_for_regression_%d_pp_%.02f.mat',video_path,si,prop),'data','annot_data');
    end
end
