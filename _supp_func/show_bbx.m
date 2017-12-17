function [ oim ] = show_bbx( im, array, sz, col)

    oim = im;
    
    szd = floor(sz/2);
    array = round(array);
%     szd = 0;
    
    if(size(im,3) == 1)
        oim = repmat(im,[1 1 3]);
    elseif(size(im,3) ~= 3)
        disp('Error with an input image');
        return;
    end

    for i = 1:size(array,1)
        si = array(i,1);
        sj = array(i,2);
        fi = array(i,3);
        fj = array(i,4);
        
        if(si < 1) si = 1; fi = array(i,3)-array(i,1); end
        if(sj < 1) sj = 1; fj = array(i,4)-array(i,2); end
        
        if(fi > size(im,1)) fi = size(im,1); si = size(im,1)-array(i,3)+array(i,1); end
        if(fj > size(im,2)) fj = size(im,2); sj = size(im,2)-array(i,4)+array(i,2); end
        
%         fi = int32(fi);
%         si = int32(si);
        
        oim(si:fi,[sj-szd+1:sj-szd+sz,fj-szd+1:fj-szd+sz],1:3) = repmat(reshape(col,1,1,3),[fi-si+1,numel([sj-szd+1:sj-szd+sz,fj-szd+1:fj-szd+sz]),1]);
        oim([si-szd+1:si-szd+sz,fi-szd+1:fi-szd+sz],sj:fj,1:3) = repmat(reshape(col,1,1,3),[numel([si-szd+1:si-szd+sz,fi-szd+1:fi-szd+sz]),fj-sj+1,1]);
    end
end

