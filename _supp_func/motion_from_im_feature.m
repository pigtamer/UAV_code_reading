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

[featuresIn   validPtsIn]  = extractFeatures(original,  ptsOriginal);
[featuresOut validPtsOut]  = extractFeatures(distorted, ptsDistorted);

index_pairs = matchFeatures(featuresIn, featuresOut);

matchedOriginal  = validPtsIn(index_pairs(:,1));
matchedDistorted = validPtsOut(index_pairs(:,2));

if((size(index_pairs,1) < 2)||(matchedOriginal.Count < 3)) 
    recovered = distorted;
    tform_matrix = -1*eye(3);
    return;
end

if(show_flag == 1)
    ax = axes;
    showMatchedFeatures(original,distorted,matchedOriginal,matchedDistorted,'Parent',ax);
    legend(ax,'Matched points 1','Matched points 2');
    title('Putatively matched points (including outliers)');
end

gte = vision.GeometricTransformEstimator; % defaults to RANSAC
gte.Transform = 'Affine'; % 'Nonreflective similarity';
gte.NumRandomSamplingsMethod = 'Desired confidence';
gte.MaximumRandomSamples = 1000;
gte.DesiredConfidence = 99.8;
% Compute the transformation from the distorted to the original image.

[tform_matrix inlierIdx] = step(gte, matchedDistorted.Location, matchedOriginal.Location);

if(show_flag == 1)
    ax = axes;
    showMatchedFeatures(original,distorted,matchedOriginal(inlierIdx),matchedDistorted(inlierIdx),'Parent',ax);
    legend(ax,'Matched points 1','Matched points 2');
    title('Matching points (inliers only)');
end

tform_matrix = cat(2,tform_matrix,[0 0 1]'); % pad the matrix

if(tform_matrix(3,1) > size(original,1)/2)&&(tform_matrix(3,2) > size(original,2)/2)
    tform_matrix = eye(3);
end

if(size(inlierIdx,1) < 5)
   tform_matrix = -1*eye(3);
   recovered = original;
   return;
end

% if(sum(sum(tform_matrix > size(original,1)/4)) > 0)
%     tform_matrix = -1*eye(3);
% end

% size(inlierIdx)
% 
% tform_matrix

agt = vision.GeometricTransformer;
agt.OutputImagePositionSource = 'Property';
%  Use the size of the original image to set the output size.
[h, w] = size(original);
agt.OutputImagePosition = [1 1 w h];
recovered = step(agt, im2single(distorted), single(tform_matrix));

temp = zeros(2*size(original,1),2*size(original,2));
temp(1:size(original,1),1:size(original,2)) = original;

recovered(recovered == 0) = im2double(original(recovered == 0));
recovered = uint8(recovered*255);

time = toc;

if(show_flag == 1)
    figure, imshow([original recovered abs(original - recovered)]);
    fprintf('processed for in %.4f seconds \n',time);   
end

end

