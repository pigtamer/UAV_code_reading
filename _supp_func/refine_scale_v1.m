function [ positions ] = refine_scale_v1( positions, block, main_model_name, ref_model_name, data_name, m_iter, r_iter, prop, si, sj)

positions = [positions(:,7), ...
             positions(:,1), ...
             positions(:,2), ...
             positions(:,3), ...
             positions(:,4), ...
             positions(:,5), ...
             positions(:,6)];
         
fprintf('\n');


rnd = round(rand(1)*10000);
if(size(block,3) == 1)
    block = repmat(block,[1 1 2]);
end
Iroi = block;
save(sprintf('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/temp_fold/block_%s_%d.mat',data_name,rnd),'Iroi','positions','si','sj','-v7.3');

%% running CNN on the data
or_path = pwd;
cd('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/');

comp_f = sprintf('/home/rozantse/tmp/comp_%s_%d',data_name,rnd);
mkdir(comp_f);
system(strcat('unset LD_LIBRARY_PATH; ' , ...
              ' export PATH="/home/rozantse/anaconda/bin:/usr/local/cuda-6.5/bin/:$PATH";', ...
              ' export LD_LIBRARY_PATH="/usr/local/cuda-6.5/lib/:/usr/local/cuda-6.5/lib64/:$LD_LIBRARY_PATH";', ...
              ' THEANO_FLAGS=''compiledir="',sprintf('%s',comp_f),'",device=cpu,tensor.cmp_sloppy=0'' python try_iter.py', ...
              ' -n ''',sprintf('block_%s_%d',data_name,rnd),'''', ...                                                   % Name of the block of data
              ' -m ''',main_model_name,'''', ...                                                                        % main model file
              ' -z',sprintf(' %0.2f',prop(end)), ...                                                                        % proportion of additional data
              ' -i',sprintf(' %d',m_iter), ...                                                                              % number of iterations
              ' -r ''',ref_model_name,'''', ...                                                                         % refinement model file
              ' -x',sprintf(' %d',0.25), ...                                                                                % proportion of additional data
              ' -j',sprintf(' %d',r_iter), ...                                                                              % number of iterations
              ' -v ''',sprintf('%s_%d',data_name,rnd),''''));                                                           % random version of the build       
          
cd('Utils/tools_matlab');

rmdir(comp_f,'s');
cd(or_path);

%% loading the resulting data

positions = importdata(strcat('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/temp_fold/npos_',sprintf('%s_%d',data_name,rnd),'.txt'));
delete(strcat('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/temp_fold/npos_',sprintf('%s_%d',data_name,rnd),'.txt'));
delete(sprintf('/cvlabdata1/cvlab/forArtem/third_party/CNN/CNN_regressor/temp_fold/block_%s_%d.mat',data_name,rnd));

end

