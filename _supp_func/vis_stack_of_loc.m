function vis_stack_of_loc(Iroi,stack_of_loc,thresh,st,inter,save_flag)

if(save_flag == 0)
    figure(1);
    close(1);
    figure(1);
else
    mkdir('./temp');
    dfile = fopen('./temp/_detections_from_vsl.txt','w');
end

sv_pth = './temp/ims/';
mkdir(sv_pth);

if(save_flag == 1)
    delete(strcat(sv_pth,'*.jpg'));
end

% NMS params
opx = 4;
osc = 0.05;

stack_of_loc = stack_of_loc([stack_of_loc.score] >= thresh);
stack_of_loc = stack_of_loc([stack_of_loc.fi] - [stack_of_loc.si] >= 10);
stack_of_loc = stack_of_loc([stack_of_loc.fi] - [stack_of_loc.si] <= 50);
% stack_of_loc = stack_of_loc([stack_of_loc.fi] - [stack_of_loc.si] <= 120);

max_t = max([stack_of_loc.t]);

for i = st:inter:max_t

    ind = find([stack_of_loc.t] == i);
    oim = Iroi(:,:,i);
    
    array = [[stack_of_loc(ind).si]',[stack_of_loc(ind).sj]',[stack_of_loc(ind).fi]',[stack_of_loc(ind).fj]',[stack_of_loc(ind).score]'];
    
    if(~isempty(array))
        a = array(:,3)-array(:,1);
        [~,ind] = sort(a,'descend');
        array = array(ind,:);
    
    
        % do NMS
        j = 1;
        while j <= size(array,1)
            k = j+1;
            while (k <= size(array,1))
                % pick the largest
                if(array(j,1) <= array(k,1)+opx) && (array(j,3) >= array(k,3)-opx) && ...
                  (array(j,2) <= array(k,2)+opx) && (array(j,4) >= array(k,4)-opx) && ...
                  (array(j,5) >= array(k,5)-osc)
                  array(k,:) = [];
                  continue;
                end
                if(array(k,1) <= array(j,1)+opx) && (array(k,3) >= array(j,3)-opx) && ...
                  (array(k,2) <= array(j,2)+opx) && (array(k,4) >= array(j,4)-opx) && ...
                  (array(k,5) >= array(j,5)-osc)
                  array(j,:) = [];
                  j = j-1;
                  break;
                end
                if(array(j,1) <= array(k,1)+opx) && (array(j,3) >= array(k,3)-opx) && ...
                  (array(j,2) <= array(k,2)+opx) && (array(j,4) >= array(k,4)-opx) && ...
                  (array(k,5) >= array(j,5)+osc)
                  array(j,:) = [];
                  j = j-1;
                  break;
                end
                k = k+1;
            end
            j = j+1;
        end
        oim = show_bbx( oim, array, 2, [0 255 0]);
    end
    if(save_flag == 1)
        imwrite(oim,sprintf('%sim_%06d.jpg',sv_pth,i));
        clc;fprintf('frame: %d / %d\n',i,max_t);
        
        fprintf(dfile,'time_layer: %d detections:',i);
        for j = 1:size(array,1)
            fprintf(dfile,' (%d, %d, %d, %d, %.05f),',round(array(j,1)),round(array(j,2)),round(array(j,3)),round(array(j,4)),array(j,5));
        end
        fprintf(dfile,'\n');
    else
        imshow(oim);title(sprintf('frame: %d / %d',i,max_t));
        pause(0.01);
    end
end

if(save_flag == 1)
    fclose(dfile);
end

