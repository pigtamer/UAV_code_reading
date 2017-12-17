function [ positions ] = refine_scale_v0( positions, block, main_model_name, ref_model_name, data_name, m_iter, r_iter, prop, si, sj)

eps = 0.001;

[szi,szj,szt] = size(block);

for iter = 1:(m_iter+r_iter)

if(iter == 1)
	model_name = main_model_name;
end
clear test_x test_y;

i = 0;
dispstat('','init');
time1 = tic;

while i < size(positions,1)
    i = i+1;
    im = block(:,:,positions(i,7));
    ca = positions(i,:);
    im2 = imresize(im,ca(5),'bilinear');
    [szi2,szj2] = size(im2);
    ca(1:4) = round(ca(1:4).*ca(5));
    
    ca(1) = ca(1)-si*prop;
    ca(3) = ca(1)+round(si*(1+2*prop));
    ca(2) = ca(2)-sj*prop;
    ca(4) = ca(2)+round(sj*(1+2*prop));
    if(ca(1) > 0) && (ca(2) > 0) && (ca(3) <= szi2) && (ca(4) <= szj2)
        detection = im2(ca(1)+1:ca(3),ca(2)+1:ca(4));
        detection = single(detection);
        detection = detection-min(detection(:));
        detection = detection./(max(detection(:))+eps);
    else
        positions(i,:) = [];
        i = i-1;
        continue;
    end
	test_x(:,i) = detection(:);
    test_y(1,i) = [rand(1)];          % this line is not important - so filling up with random variables
    
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
system(strcat('unset LD_LIBRARY_PATH; THEANO_FLAGS=''compiledir=''',sprintf('%s',comp_f),' python try.py -p''./my_data/'' -n "',sprintf('%s_%d',data_name,rnd),'" -m "',model_name,'"'));

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
    di = round(res(1,i)*si*(1+2*prop));
    
    nannot(i,1) = max(positions(i,1)+round(di/positions(i,5)),1);
    nannot(i,3) = min(positions(i,3)-round(di/positions(i,5)),szi);
    nannot(i,2) = max(positions(i,2)+round(di/positions(i,5)),1);
    nannot(i,4) = min(positions(i,4)-round(di/positions(i,5)),szj);
end
positions = nannot;

if(iter == m_iter)
    prop = 0;
    model_name = ref_model_name;
end

end


end

