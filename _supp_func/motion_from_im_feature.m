function [recovered, tform_matrix] = motion_from_im_feature( original, distorted, show_flag)

tic;

if(nargin < 3)
    show_flag = 0;
end
% ---------------
show_flag = 0;
% ---------------
if(show_flag == 1)
    imshow(original);
end

ptsOriginal  = detectSURFFeatures(original);
ptsDistorted = detectSURFFeatures(distorted);

[featuresIn,   validPtsIn]  = extractFeatures(original,  ptsOriginal);
[featuresOut, validPtsOut]  = extractFeatures(distorted, ptsDistorted);

index_pairs = matchFeatures(featuresIn, featuresOut);

matchedPtsOriginal  = validPtsIn(index_pairs(:,1));
matchedPtsDistorted = validPtsOut(index_pairs(:,2));

if((size(index_pairs,1) < 2)||(matchedPtsOriginal.Count < 3)) 
    recovered = distorted;
    tform_matrix = -1*eye(3);
    return;
end

if(show_flag == 1)
    close all
    figure(1)
    ax = axes;
    showMatchedFeatures(original,distorted,matchedPtsOriginal,matchedPtsDistorted,'Parent',ax);
    legend(ax,'Matched points 1','Matched points 2');
    title('Putatively matched points (including outliers)');
end



[gte,inlierPtsDistorted,inlierPtsOriginal] = ...
    estimateGeometricTransform(matchedPtsDistorted,matchedPtsOriginal,...
    'similarity');

tform_matrix = gte.T;

if(show_flag == 1)
    figure(2),
    ax = axes;
    showMatchedFeatures(original,distorted,inlierPtsOriginal,inlierPtsDistorted);
    legend(ax,'Matched points 1','Matched points 2');
    title('Matching points (inliers only)');
    
end

% tform_matrix = cat(2,tform_matrix,[0 0 1]'); % pad the matrix
% 
% if(tform_matrix(3,1) > size(original,1)/2)&&(tform_matrix(3,2) > size(original,2)/2)
%     tform_matrix = eye(3);
% end

% if(size(inlierIdx,1) < 5)
%    tform_matrix = -1*eye(3);
%    recovered = original;
%    return;
% end

% if(sum(sum(tform_matrix > size(original,1)/4)) > 0)
%     tform_matrix = -1*eye(3);
% end

% size(inlierIdx)
% 
% tform_matrix


% --- DEPRECATED SINCE 2016A---
% agt = vision.GeometricTransformer;
% agt.OutputImagePositionSource = 'Property';


recovered = imwarp(distorted, gte, 'OutputView', imref2d(size(original)), ...
    'Interp', 'nearest');
margin_idx = uint8(recovered) == 0;
recovered(margin_idx) = original(margin_idx);


time = toc;

if(show_flag == 1)
    figure(3),
    imshow(recovered), title('Recovered Image')
    figure(4),
    imshow([original recovered abs(original - recovered)]);
    figure(5),
    mesh(abs(original - distorted)), title('Distorted gradient')
    figure(6),
    mesh(abs(original - recovered)), title('Recovered gradient')
    pause(0.01)
%     fprintf('processed for in %.4f seconds \n',time);   
end

end

