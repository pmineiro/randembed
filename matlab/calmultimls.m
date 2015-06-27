function [wr,b,ww]=calmultimls(xt,yt,varargin)
    [n,k]=size(yt);
    [~,d]=size(xt);

    if (exist('dmsm','file') == 3 && ...
        exist('chofactor','file') == 3 && ...
        exist('chosolve','file') == 3 && ...
        exist('fastexpprob','file') == 3)
      havemex=true;
    else
      havemex=false;
      if (~exist('OCTAVE_VERSION','builtin'))
        warning('rembed:nomex', 'MEX acceleration not available, have you compiled the mex?');
      end
    end
    
    [lambda,rfac,f,fbs,kernel,logisticiter,...
     monfunc,bsfac,decay,eta,alpha,shrink]=parseArgs(k,varargin{:});
    
    [wr,b]=randfeats(xt,rfac,f,kernel);
    
    % Calibration
    if (fbs >= f)
        xn=zeros(n,f+1,'single');
        xn(:,1:f)=xt*wr; xn(:,1:f)=bsxfun(@plus,xn(:,1:f),b); xn=cos(xn); % cos(0)=1
        c=xn'*xn;
        tr=sum(diag(c));
        diagidx=sub2ind(size(c),1:size(c,1),1:size(c,1));
        c(diagidx)=c(diagidx)+(lambda*tr/(f+1));
        if (havemex)
          chofactor(c);
        else
          c=chol(c);
        end
        if (issparse(yt))
          if (havemex)
            xnticyt=dmsm(xn',yt); % TODO: form xn' in tic-d form
          else
            xnticyt=single(double(xn)'*yt); % :(
          end
        else
          xnticyt=xn'*yt;
        end
        if (shrink < 1)
          xnticyt=(2*shrink-1)*xnticyt;
          sumxn=sum(xn,1);
          xnticyt=bsxfun(@plus,xnticyt,(1-shrink)*sumxn');
          clear sumxn;
        end
        clear xn;
    else
        xnticyt=zeros(f+1,size(yt,2),'single');
        xnticxn=zeros(f+1,f+1,'single');
        tr=0;
        for off=1:fbs:f
            offend=min(f,off+fbs-1);
            xntmp=xt*wr(:,off:offend);
            xntmp=bsxfun(@plus,xntmp,b(off:offend));
            xntmp=cos(xntmp);
            if (issparse(yt))
              if (havemex)
                xnticyt(off:offend,:)=dmsm(xntmp',yt); % TODO: form tic-d
              else
                xnticyt(off:offend,:)=single(double(xntmp)'*yt); % :(
              end
            else
              xnticyt(off:offend,:)=xntmp'*yt;
            end
            sumxntmp=sum(xntmp,1);
            if (shrink < 1)
              xnticyt(off:offend,:)=(2*shrink-1)*xnticyt(off:offend,:);
              xnticyt(off:offend,:)=...
                bsxfun(@plus,xnticyt(off:offend,:),(1-shrink)*sumxntmp');
            end
            xnticxn(1+f,off:offend)=sumxntmp;
            xnticxn(off:offend,1+f)=sumxntmp;
            clear sumxntmp;
            tr=tr+sum(sum(xntmp.*xntmp));
            for off2=off:fbs:f
              off2end=min(f,off2+fbs-1);
              xntmp2=xt*wr(:,off2:off2end);
              xntmp2=bsxfun(@plus,xntmp2,b(off2:off2end));
              xntmp2=cos(xntmp2);
              xnticxn(off:offend,off2:off2end)=xntmp'*xntmp2;
              xnticxn(off2:off2end,off:offend)=...
                  xnticxn(off:offend,off2:off2end)';
            end
        end
        clear xntmp xntmp2;
        xnticxn(1+f,1+f)=size(xt,1);
        xnticyt(1+f,:)=full(sum(yt,1));
        if (shrink < 1)
          % sum(shrink*yt+(1-shrink)*(1-yt),1)
          % = sum((2*shrink-1)*yt+(1-shrink)*ones(n,1)*ones(1,k),1)
          % = (2*shrink-1)*sum(yt,1)+(1-shrink)*n*ones(1,k)
          xnticyt(1+f,:)=(2*shrink-1)*xnticyt(1+f,:);
          xnticyt(1+f,:)=xnticyt(1+f,:)+(1-shrink)*n;
        end
        diagidx=sub2ind(size(xnticxn),1:size(xnticxn,1),1:size(xnticxn,1));
        xnticxn(diagidx)=xnticxn(diagidx)+(lambda*tr/(f+1));
        c=xnticxn;
        if (havemex)
          chofactor(c);
        else
          c=chol(c);
        end
        clear xnticxn;
    end
    if (havemex)
      ww=xnticyt+0; chosolve(c,ww);
    else
      ww=c\(c'\xnticyt);
    end
    monfunc(wr,b,ww,0.5);
    eta=eta*(1.0-alpha);
    if (alpha > 0)
      dwold=zeros(size(ww,1),size(ww,2),'single');
    end
    for i=1:logisticiter
      bs=min(ceil(n/10),ceil(bsfac/max(k,f)));
      perm=randperm(n);
      for off=1:bs:n
        offend=min(n,off+bs-1);
        xntmptic=zeros(1+f,offend-off+1,'single');
        xntmptic(1:f,:)=wr'*xt(perm(off:offend),:)';
        xntmptic(1:f,:)=bsxfun(@plus,xntmptic(1:f,:),b');
        xntmptic=cos(xntmptic); % xntmptic(f+1,:)=1;
        pptic=ww'*xntmptic;
        pptic=min(pptic,40); 
        if (havemex)
          fastexpprob(pptic);  
        else
          pptic=exp(pptic); pptic=pptic./(1+pptic);
        end                                       % probabilistic predictions
        g=-xntmptic*pptic'; clear xntmptic pptic; % gradient
        g=g+(bs/n)*xnticyt;
        if (havemex)
          chosolve(c,g);
        else
          g=c\(c'\g);
        end
        g=(eta*n/bs)*g;
        if (alpha > 0)
          dwold=alpha*dwold;
          dwold=dwold+g;
          ww=ww+dwold;
        else
          ww=ww+g;
        end
        clear g;
      end
      monfunc(wr,b,ww,0.0);
      if (decay < 1)
        if (alpha > 0)
          dwold=decay*dwold;
        end
        eta=eta*decay;
      end
    end
end

function [lambda,rfac,f,fbs,kernel,logisticiter,...
          monfunc,bsfac,decay,eta,alpha,shrink]=parseArgs(k,varargin)
  lambda=1.0;
  if (isfield(varargin{1},'lambda'))
      lambda=varargin{1}.lambda;
  end
  rfac=1.0;
  if (isfield(varargin{1},'rfac'))
      rfac=varargin{1}.rfac;
  end
  f=1000;
  if (isfield(varargin{1},'f'))
      f=varargin{1}.f;
  end
  fbs=f;
  if (isfield(varargin{1},'fbs'))
      fbs=varargin{1}.fbs;
  end
  kernel='g';
  if (isfield(varargin{1},'kernel'))
      kernel=varargin{1}.kernel;
  end
  logisticiter=0;
  if (isfield(varargin{1},'logisticiter'))
      logisticiter=varargin{1}.logisticiter;
  end
  monfunc=@(wr,b,ww,th) 1;
  if (isfield(varargin{1},'monfunc'))
      monfunc=varargin{1}.monfunc;
  end
  bsfac=1e+9;
  if (isfield(varargin{1},'bsfac'))
      bsfac=varargin{1}.bsfac;
  end  
  decay=1.0;
  if (isfield(varargin{1},'decay'))
      decay=min(1,varargin{1}.decay);
  end
  eta=1.0;
  if (isfield(varargin{1},'eta'))
      eta=varargin{1}.eta;
  end
  alpha=0.0;
  if (isfield(varargin{1},'alpha'))
      alpha=varargin{1}.alpha;
  end
  shrink=1.0;
  if (isfield(varargin{1},'shrink'))
      shrink=varargin{1}.shrink;
  end
end

function r=quasirandn(d,f,class)
  % --- latin hypercube on [0,1]^d
  r=rand(f,d);
  for ii=1:d
    [~,ind]=sort(r(:,ii));
    r(:,ii)=ind;
  end
  r=r-rand(size(r));
  r=r/f; 
  % ---

  r=erfinv(2*r'-1);
  if (strcmp(class,'single'))
    r=single(r);
  end
end

function r=octavesaferandn(d,f,class)
  r=randn(d,f);
  if (strcmp(class,'single'))
    r=single(r);
  end
end

function [r,b]=randfeats(x,rfac,f,kernel)
  chi2rnd=@(a,n,p) 2*arrayfun(@(z)randg(z),(a/2)*ones(n,p));

  [n,d]=size(x);
  subsz=min(n,1000);
  sample=x(randperm(n,subsz),:);
  norms=sum(sample.*sample,2);
  dist2=bsxfun(@plus,norms',bsxfun(@plus,-2*(sample*sample'),norms));
  sdd=sort(dist2(:));
  quantile=0.5;
  qd=sdd(round(quantile*length(sdd)));
  
  if (kernel(1) == 'q') % "quasirandom"
    kernel=kernel(2:end);
    randnfunc=@quasirandn;
  else
    randnfunc=@octavesaferandn;
  end

  switch kernel
      case 'g'
          scale=sqrt(2/qd);
          r=rfac*scale*randnfunc(d,f,'single');
      case 'c'
          scale=sqrt(2/qd);
          r=rfac*scale*tan(pi*(randnfunc(d,f,'single')-0.5));
      case 'm3'
          nu=3;
          scale=1/sqrt(qd);
          r=bsxfun(@times,sqrt(nu./chi2rnd(nu,1,f)),rfac*scale*randnfunc(d,f,'single'));
      case 'm5'
          nu=5;
          scale=1/sqrt(qd);
          r=bsxfun(@times,sqrt(nu./chi2rnd(nu,1,f)),rfac*scale*randnfunc(d,f,'single'));
      otherwise
          error('unknown kernel')
  end
  
  b=pi*(single(rand(1,f))-0.5);
end
