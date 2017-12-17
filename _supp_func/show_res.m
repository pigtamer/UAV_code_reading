function show_res(block,res,st_flag,st)

if(isempty(st_flag) || nargin < 3)
    st_flag = 0;
end

if(isempty(st) || nargin < 4)
    st = 1;
end

leng = size(res{1},1);
dlen = numel(res);

for i = st:leng
    for j = 1:dlen
        subplot(1,dlen,j); imshow(block(res{j}(i,1):res{j}(i,3),res{j}(i,2):res{j}(i,4),j));
    end
    subplot(1,dlen,1); title(sprintf('detection #%d',i));
    if(st_flag == 0)
        pause(0.25);
    else
    w = waitforbuttonpress;
    if(w ~= 0)
       break; 
    end
    end
end

