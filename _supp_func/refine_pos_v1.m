function [ positions ] = refine_pos_v1( positions, block, vert_reg, hor_reg, iter, si, sj)

positions = [positions(:,7), ...
             positions(:,1), ...
             positions(:,2), ...
             positions(:,3), ...
             positions(:,4), ...
             positions(:,5), ...
             positions(:,6)];
         
if(size(block,3) == 1)
    block = repmat(block,[1 1 2]);
end


%% motion compensation
eps = 0.001;

annot = positions;
nannot = annot;
hbin = 4;

vhog = vl_hog(im2single(rand(40,40)),hbin);

vhogs = zeros(size(nannot,1),size(vhog(:)',2));

for j = 1:iter
    
    for i = 1:size(nannot,1)
    
        ca = nannot(i,:);
        msz = ca(4)-ca(2);
    
        if ca(2) < 1
            ca(2) = 1; ca(4) = msz;
        end
        if ca(3) < 1
            ca(3) = 1; ca(5) = msz;
        end
        if ca(4) >= size(block,1)
            ca(2) = size(block,1)-msz;ca(4) = size(block,1);
        end
        if ca(5) >= size(block,2)
            ca(3) = size(block,2)-msz;ca(5) = size(block,2);
        end
        
        cut_im = block(ca(2):ca(4),ca(3):ca(5),ca(1));
    
%         cut_im = imresize(cut_im,[si,sj],'bilinear');
%         cut_im = single(cut_im);
%         cut_im = cut_im-min(cut_im(:));
%         cut_im = cut_im./(max(cut_im(:))+eps);
        
        vhog = vl_hog(im2single(imresize(cut_im,[si,sj],'bilinear')),hbin);
        vhogs(i,:) = vhog(:)';
        clc; fprintf('%d/%d',i,size(annot,1));
    end
    
    pred_vert = SQBMatrixPredict( vert_reg, single(vhogs));
    pred_hor = SQBMatrixPredict( hor_reg, single(vhogs));
    fprintf(' -> [done]')
    res = [pred_vert(:)'; pred_hor(:)'];
    res = reshape(res,2,[]);
    
    di = round(res(1,i)*msz);
    dj = round(res(2,i)*msz);
    
    nannot(i,2) = min(max(annot(i,2)-round(di),1),size(block,1)-msz);
    nannot(i,4) = nannot(i,2)+msz;
    nannot(i,3) = min(max(annot(i,3)-round(dj),1),size(block,2)-msz);
    nannot(i,5) = nannot(i,3)+msz;
    
end

positions = [positions(:,2), ...
             positions(:,3), ...
             positions(:,4), ...
             positions(:,5), ...
             positions(:,6), ...
             positions(:,7), ...
             positions(:,1)];

end

