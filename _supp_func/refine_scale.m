function [ positions ] = refine_scale( positions, block, main_model_name, ref_model_name, data_name, m_iter, r_iter, prop, si, sj)

eps = 0.001;

[szi,szj,szt] = size(block);

for iter = 1:(m_iter+r_iter)

if(iter == 1)
	model_name = main_model_name;
end
if(m_iter == 0)&&(iter == 1)
    prop = [-0.25, 0, 0.25];
    si = 40;
    sj = 40;
    model_name = ref_model_name;
end

clear test_x test_y;

i = 0;
dispstat('','init');
time1 = tic;

while i < size(positions,1)
    i = i+1;
    ca = positions(i,:);
	im = block(:,:,ca(7));      
%     im2 = imresize(im,ca(6),'bilinear');
    [szi,szj] = size(im);
    dsi = ca(3)-ca(1);
    dsj = ca(4)-ca(2);
    
    cdet = [];
    
    rca = ca;
    for pps = 1:numel(prop)
        ca = rca;
        ca(1) = ca(1)-floor(dsi*prop(pps)/2);
        ca(3) = ca(3)+floor(dsi*prop(pps)/2);
        ca(2) = ca(2)-floor(dsj*prop(pps)/2);
        ca(4) = ca(4)+floor(dsj*prop(pps)/2);
        
        if(ca(1) >= 0) && (ca(2) >= 0) && (ca(3) < szi) && (ca(4) < szj)
            detection = im(round(ca(1))+1:round(ca(3)),round(ca(2))+1:round(ca(4)));
            detection = single(detection);
            detection = detection-min(detection(:));
            detection = detection./(max(detection(:))+eps);
            detection = imresize(detection,[si,sj],'bilinear');
            cdet = [cdet; detection(:)];
        else
            positions(i,:) = [];
            i = i-1;
            cdet = [];
            break;
        end
    end
    if(isempty(cdet))
        continue;
    end
    cdet = reshape(cdet,si,sj,numel(prop));
    cdet = permute(cdet,[3 1 2]);
	test_x(:,i) = cdet(:);
    test_y(1,i) = rand(1);          % this line is not important - so filling up with random variables
    
    if(toc(time1) > 5)
        dispstat(sprintf('%.02f%% finished..',100*i/size(positions,1)));
        time1 = tic;
    end
end

rnd = round(rand(1)*1000000);

save(sprintf('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/my_data/data_%s_%d.mat',data_name,rnd),'test_x','test_y','-v7.3');

%% running CNN on the data
or_path = pwd;
cd('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/');

comp_f = sprintf('~/tmp/comp_%d',rnd);
mkdir(comp_f);

fprintf('\nStarted iteration #%d\n\n',iter);
system(strcat('unset LD_LIBRARY_PATH; THEANO_FLAGS=''compiledir=''',sprintf('%s',comp_f),' python try.py -p''./my_data/'' -n "',sprintf('%s_%d',data_name,rnd),'" -m "',model_name,'" -c "conf_',model_name,'"'));

rmdir(comp_f,'s');

cd(or_path);

%% loading the resulting data

res = importdata(strcat('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/my_data/tst_predicted_values_',sprintf('%s_%d',data_name,rnd),'.txt'));
delete(strcat('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/my_data/tst_predicted_values_',sprintf('%s_%d',data_name,rnd),'.txt'));
delete(sprintf('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/my_data/data_%s_%d.mat',data_name,rnd));
load(strcat('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/my_data/data_',model_name,'_params.mat'));

res = res.*maxi;
res = res+mini;
res = reshape(res,1,[]);

nannot = positions;

for i = 1:size(positions,1)
    ext = round(res(1,i)*(positions(i,4)-positions(i,2))/(1+2*res(1,i)));
    
    nannot(i,1) = max(positions(i,1)+ext,1);
    nannot(i,3) = min(positions(i,3)-ext,szi);
    nannot(i,2) = max(positions(i,2)+ext,1);
    nannot(i,4) = min(positions(i,4)-ext,szj);
end
positions = nannot;

if(iter == m_iter)
    prop = [-0.25, 0, 0.25];
    si = 40;
    sj = 40;
    model_name = ref_model_name;
end

end


end

