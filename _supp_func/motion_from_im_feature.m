function [recovered, tform_matrix] = motion_from_im_feature( original, distorted, show_flag)

epsilon = 0.1;

tic;

if(nargin < 3)
    show_flag = 0;
end

if(show_flag == 1)
    imshow(original);
end

ptsOriginal  = detectSURFFeatures(original);
ptsDistorted = detectSURFFeatures(distorted);
[featuresOriginal,validPtsOriginal] = ...
    extractFeatures(original,ptsOriginal);
[featuresDistorted,validPtsDistorted] = ...
    extractFeatures(distorted,ptsDistorted);
index_pairs = matchFeatures(featuresOriginal,featuresDistorted);
matchedPtsOriginal  = validPtsOriginal(index_pairs(:,1));
matchedPtsDistorted = validPtsDistorted(index_pairs(:,2));

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


% [tform_matrix inlierIdx] = step(gte, matchedPtsDistorted.Location, matchedPtsOriginal.Location);
tform_matrix = gte.T;
% inlierIdx = 

if(show_flag == 1)
    figure(2),
    ax = axes;
    
%     showMatchedFeatures(original,distorted,matchedPtsOriginal(inlierIdx),matchedPtsDistorted(inlierIdx),'Parent',ax);
    
    showMatchedFeatures(original,distorted,inlierPtsOriginal,inlierPtsDistorted);
    legend(ax,'Matched points 1','Matched points 2');
    title('Matching points (inliers only)');
end

% tform_matrix = cat(2,tform_matrix,[0 0 1]'); % pad the matrix

if(tform_matrix(3,1) > size(original,1)/2)&&(tform_matrix(3,2) > size(original,2)/2)
    tform_matrix = eye(3);
end

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



%  Use the size of the original image to set the output size.
% [h, w] = size(original);
% agt.OutputImagePosition = [1 1 w h];
% recovered = step(agt, im2single(distorted), single(tform_matrix));
% 
% temp = zeros(2*size(original,1),2*size(original,2));
% temp(1:size(original,1),1:size(original,2)) = original;
% 
% recovered(recovered == 0) = im2double(original(recovered == 0));
% recovered = uint8(recovered*255);

recovered = imwarp(distorted, gte, 'OutputView', imref2d(size(original)));
figure(3),
imshow(recovered),




time = toc;

if(show_flag == 1)
    figure,
    imshow([original recovered abs(original - recovered)]);

    fprintf('processed for in %.4f seconds \n',time);   
end

end

