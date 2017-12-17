function [V, numblocks] = vol2col(A,bsize,kind,shift)
%extension of Matlabs im2col to volumes

if strcmp(kind, 'distinct')
    numblocks = floor(size(A)./bsize);
    
    V = zeros(prod(bsize),prod(numblocks));
    x = zeros(prod(bsize),1);
    
    rows = 1:bsize(1); cols = 1:bsize(2); slices = 1:bsize(3);
    
    for k=0:numblocks(3)-1
        for j=0:numblocks(2)-1
            for i=0:numblocks(1)-1
                x(:) = A(bsize(1)*i+rows,bsize(2)*j+cols,bsize(3)*k+slices);
                V(:,i + j*numblocks(1) + k*numblocks(1)*numblocks(2) + 1) = x;
            end
        end
    end
    
elseif strcmp(kind,'sliding')

    if nargin < 4
        shift = 0;
    end
    switch shift
        case 0
            [ma,na,oa] = size(A);
        case 1
            [dump, ma,na,oa] = size(A);        
        otherwise
            error('unknown shift')
    end
        
        
    m = bsize(1); n = bsize(2); o = bsize(3);
    
    if any([ma na oa] < [m n o]) % if neighborhood is larger than image
       b = zeros(m*n*o,0);
       return
    end
    
    % Create Hankel-like indexing sub matrix.
    mc = bsize(1); nc = ma-m+1; nn = na-n+1; no = oa-o+1;
    cidx = (0:mc-1)'; ridx = 1:nc;
    t = cidx(:,ones(nc,1)) + ridx(ones(mc,1),:);    % Hankel Subscripts
    tt = zeros(mc*n,nc);
    rows = 1:mc;
    for i=0:n-1,
        tt(i*mc+rows,:) = t+ma*i;
    end
    ttt = zeros(mc*n,nc*nn);
    cols = 1:nc;
    for j=0:nn-1,
        ttt(:,j*nc+cols) = tt+ma*j;
    end
    tttt = zeros(m*n*o,nc*nn);
    slices = 1:m*n;
    for k=0:o-1,
        tttt(k*m*n+slices,:) = ttt+ma*na*k;
    end
    ttttt = zeros(m*n*o,nc*nn*no);
    slices2 = 1:nc*nn;
    for l=0:no-1,
        ttttt(:,l*nc*nn+slices2) = tttt+ma*na*l;
    end
    
    switch shift
        case 0
            V = A(ttttt);
        case 1
            V = A(:,ttttt(:));
            V = reshape(V, [], size(ttttt,2));
    end
        
    numblocks = [nc nn no];
    
end