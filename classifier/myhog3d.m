function [myhog, params, hg] = myhog3d(Iroi,rsize,numcell,sigma,tau,bintype)

smooth = 1;

if nargin < 2 || isempty(rsize)
    rsize = [16 16 4]; % support region size (sbin)
end
if nargin < 3 || isempty(numcell)
    numcell = [2 2 2]; % number of cells per suport region
end
cellsize = rsize./numcell; %size of individual histogram cells

if nargin < 4 || isempty(sigma)
    sigma = 1;
end
if nargin < 5 || isempty(tau)
    tau = 1;
end
if nargin < 6 || isempty(bintype)
    bintype = 'dodeca';
end
sigmas = [sigma sigma tau];

overlaptype = 'sliding'; % {'sliding', 'sliding'}
filtertype = 'block'; % {'gauss', 'block'}
normalize = 1;

params.sigmas = sigmas; params.rsize = rsize; params.numcell = numcell; ...
    params.smooth = smooth; params.cellsize = cellsize;


%%% setup polyhedron
th = (1 + sqrt(5))/2;
switch bintype
    case 'dodeca'
        dodeca_half = [0 1 th; 0 -1 th; 1 th 0;  -1 th 0; th 0 1;  -th 0 1];
        dodeca_half = dodeca_half./repmat(sqrt(sum(dodeca_half.^2,2)),1,3);
        hmax = dodeca_half(1,:)*dodeca_half(3,:)';
        bin_orients = dodeca_half;
    case 'icosa'
        icosa_half = [1 1 1; 1 1 -1; 1 -1 1; -1 1 1; 0 1/th th; 0 -1/th th; ...
            1/th th 0; -1/th th 0; th 0 1/th; th 0 -1/th];
        icosa_half = icosa_half./repmat(sqrt(sum(icosa_half.^2,2)),1,3);
        hmax = icosa_half(1,:)*icosa_half(5,:)';
        bin_orients = icosa_half;
end
hdim = length(bin_orients);
params.hdim = hdim;

%%% get space-time grads %%%%%%%%%

V = gaussSmooth( Iroi, sigmas, 'smooth' );
% dx = [-1 0 1]; dy = dx'; dt = cat(3, cat(3,-1,0), 1);
% Vx = convnFast(V, dx, 'valid'); Vy = convnFast(V, dy, 'valid'); Vt = convnFast(V, dt, 'valid');
[Vx,Vy,Vt] = gradient(V);

newsize = min(min(size(Vx),size(Vy)),size(Vt));
Vx = arrayToDims( Vx, newsize ); Vy = arrayToDims( Vy, newsize ); Vt = arrayToDims( Vt, newsize );

norma = sqrt(Vx.^2 + Vy.^2 + Vt.^2);
Vx = Vx./max(norma,eps); Vy = Vy./max(norma,eps); Vt = Vt./max(norma,eps);

%%% assign grads to hisogram bins
if smooth
    Q = abs((bin_orients * ([Vx(:)'; Vy(:)'; Vt(:)']))) - hmax;
    Q(Q<0) = 0;
    %     Q = Q./repmat(max(sqrt(sum(Q.^2,1)),eps),hdim,1); %normlize L2 norm
    Q = Q.*repmat(1./max(sum(Q,1),eps),hdim,1); %normalize L1 norm
    Q = Q.*repmat(norma(:)',hdim,1);
    Q = reshape(Q',[size(norma) hdim]);
else
    Q = abs((bin_orients * ([Vx(:)'; Vy(:)'; Vt(:)'])));
    [mx, best_o] = max(Q,[],1);
    Qb = zeros(size(Q));
    Qb(sub2ind(size(Qb),best_o,1:size(Qb,2))) = norma(:);
    Q = reshape(Qb',[size(norma) hdim]);
end

switch filtertype
    case 'block'
        sampletmp = localSum(Q, [cellsize 1], 'block');
        sampletmp = reshape(sampletmp, floor(size(Q)./[cellsize 1]));
    case 'gauss'
        G = filterGauss([cellsize]);
        tmp = convnFast(Q, G, 'smooth');
        sampletmp = tmp(1:cellsize(1):end,1:cellsize(2):end,1:cellsize(3):end,:);
end

if normalize == 1
    cutvalue = 0.2; %histogram values larger than cutvalue will be cut
    eps_reg = 0.0000001;
    [Qblocks, numblocks] = vol2col(shiftdim(sampletmp,3), numcell, overlaptype, 1);
    Qblocks(Qblocks<0) = 0;
    qnorma = 1./sqrt(sum(Qblocks.^2,1)+eps_reg^2);
    Qblocks = Qblocks.*repmat(qnorma,size(Qblocks,1),1);
    Qblocks(Qblocks>cutvalue) = cutvalue;
    Qblocks(Qblocks<0) = 0;
    qnorma = 1./sqrt(sum(Qblocks.^2,1)+eps_reg^2);
    Qblocks = Qblocks.*repmat(qnorma,size(Qblocks,1),1);
    Qblocks= reshape(Qblocks,hdim*prod(numcell),[]);
    hg = reshape(Qblocks, [], numblocks(1), numblocks(2), numblocks(3));
    myhog = Qblocks(:)';
elseif normalize == 2
    eps_reg = 0.0000001;
    smpnorm = 1./sqrt(abs(localSum(sum(sampletmp.^2,4), [2 2 1], 'valid'))+eps_reg^2);
    myhog = cat(4, ...
        sampletmp(2:end-1,2:end-1,:,:).*repmat(smpnorm(1:end-1,1:end-1,:),[1,1,1,hdim]), ...
        sampletmp(2:end-1,2:end-1,:,:).*repmat(smpnorm(1:end-1,1:end-1,:),[1,1,1,hdim]), ...
        sampletmp(2:end-1,2:end-1,:,:).*repmat(smpnorm(1:end-1,2:end,:),[1,1,1,hdim]), ...
        sampletmp(2:end-1,2:end-1,:,:).*repmat(smpnorm(1:end-1,2:end,:),[1,1,1,hdim]), ...
        sampletmp(2:end-1,2:end-1,:,:).*repmat(smpnorm(2:end,1:end-1,:),[1,1,1,hdim]), ...
        sampletmp(2:end-1,2:end-1,:,:).*repmat(smpnorm(2:end,1:end-1,:),[1,1,1,hdim]), ...
        sampletmp(2:end-1,2:end-1,:,:).*repmat(smpnorm(2:end,2:end,:),[1,1,1,hdim]), ...
        sampletmp(2:end-1,2:end-1,:,:).*repmat(smpnorm(2:end,2:end,:),[1,1,1,hdim]));
    myhog = shiftdim(myhog,3);
else
    myhog = shiftdim(sampletmp,3);
end

