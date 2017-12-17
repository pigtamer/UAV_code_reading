function [ positions ] = refine_pos( positions, block, main_model_name, ref_model_name, data_name, m_iter, r_iter, prop, si, sj)

eps = 0.001;

[szi,szj,szt] = size(block);

for iter = 1:(m_iter+r_iter)

if(iter == 1)
	model_name = main_model_name;
end
clear test_x test_y;

i = 0;
time1 = tic;

while i < size(positions,1)
    i = i+1;
    im = block(:,:,positions(i,7));
    ca = positions(i,:);
    im2 = imresize(im,ca(5),'bilinear');
    ca(1:4) = round(ca(1:4).*ca(5));
    
    ca(1) = ca(1)-si*prop;
    ca(3) = ca(1)+round(si*(1+2*prop));
    ca(2) = ca(2)-sj*prop;
    ca(4) = ca(2)+round(sj*(1+2*prop));
    if(ca(1) >= 0) && (ca(2) >= 0) && (ca(3) <= szi*ca(5)) && (ca(4) <= szj*ca(5))
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
    test_y(1:2,i) = [rand(1),rand(1)];          % this line is not important - so filling up with random variables
    
    if(toc(time1) > 10)
        fprintf(sprintf('%.02f%% -> ',100*i/size(positions,1)));
        time1 = tic;
    end
end

fprintf('\n');

rnd = round(rand(1)*1000000);

save(sprintf('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/my_data/data_%s_%d.mat',data_name,rnd),'test_x','test_y','-v7.3');

%% running CNN on the data
or_path = pwd;
cd('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/');

comp_f = sprintf('~/tmp/comp_%s_%d',data_name,rnd);
mkdir(comp_f);

fprintf('\nStarted iteration #%d\n\n',iter);
system(strcat('unset LD_LIBRARY_PATH; THEANO_FLAGS=''compiledir=''',sprintf('%s',comp_f),' python try_bd.pyc -p''./my_data/'' -n "',sprintf('%s_%d',data_name,rnd),'" -m "',model_name,'"'));
cd('Utils/tools_matlab');

rmdir(comp_f,'s');

cd(or_path);

%% loading the resulting data

res = importdata(strcat('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/my_data/tst_predicted_values_',sprintf('%s_%d',data_name,rnd),'.txt'));
delete(strcat('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/my_data/tst_predicted_values_',sprintf('%s_%d',data_name,rnd),'.txt'));
delete(sprintf('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/my_data/data_%s_%d.mat',data_name,rnd));
load(strcat('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/my_data/data_',model_name,'_params.mat'));

res = res.*maxi;
res = res+mini;
res = reshape(res,2,[]);

nannot = positions;

for i = 1:size(positions,1)
    di = round(res(1,i)*si*(1+2*prop));
    dj = round(res(2,i)*sj*(1+2*prop));
    
    nannot(i,1) = min(max(positions(i,1)-round(di/positions(i,5)),0.25*round(si/positions(i,5))+1),szi-1.25*round(si/positions(i,5)));
    nannot(i,3) = nannot(i,1)+round(si/positions(i,5));
    nannot(i,2) = min(max(positions(i,2)-round(dj/positions(i,5)),0.25*round(sj/positions(i,5))+1),szj-1.25*round(sj/positions(i,5)));
    nannot(i,4) = nannot(i,2)+round(sj/positions(i,5));
end
positions = nannot;

if(iter == m_iter)
    prop = 0;
    model_name = ref_model_name;
end

end


end

