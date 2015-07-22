function res=traintestone(what,varargin)
  addpath('../matlab/');
  prefix='./';

  randn('seed',8675309);
  rand('seed',90210);
  
  start=tic;
  
  load(what);
  trainright=xt;
  trainleft=yt;
  testright=xs;
  testleft=ys;

  % train-test split

  [n,d]=size(trainright);
  [m,~]=size(testright);
  
%   v=sort(eig(trainleft'*trainleft),'descend');
%   vs=cumsum(v);
%   vs=vs/vs(end);
%   embedd=min(find(vs>0.9));
  embedd=50;

  % add constant feature
  trainright=horzcat(trainright,ones(n,1));
  testright=horzcat(testright,ones(m,1));
    
  ultraw=ones(1,size(trainleft,1));
  
  tic
  rs=rembed(trainright',ultraw,trainleft',embedd, ...
            struct('pre','diag','innerloop',100,'lambda',1,...
                   'tmax',2,'trainpredictor',true,'verbose',false));
  toc

  trainh=horzcat(rs.projectx(trainright),ones(n,1));
  testh=horzcat(rs.projectx(testright),ones(m,1));
  
  perm=randperm(n);
  nsubtrain=ceil(0.9*n);
  subtrainh=trainh(perm(1:nsubtrain),:);
  subtrainleft=trainleft(perm(1:nsubtrain),:);
  valtrainh=trainh(perm(nsubtrain+1:end),:);
  valtrainleft=trainleft(perm(nsubtrain+1:end),:);
  subppos=full(sum(subtrainleft,1))/size(subtrainleft,1);
  
  maxiter=100;
  allf=500+randi(6000,1,maxiter);
  allrfac=0.5+3.5*rand(1,maxiter);
  alllambda=exp(log(1e-5)+(log(1e-2)-log(1e-5))*rand(1,maxiter));
  alllogisticiter=1+randi(24,1,maxiter);
  alleta=0.25+4.0*rand(1,maxiter);
  allalpha=0.25+0.7*rand(1,maxiter);
  alldecay=0.9+0.15*rand(1,maxiter);
  allkern=randi(6,1,maxiter);
  kerntags={ 'g', 'm3', 'm5', 'qg', 'qm3', 'qm5' };
  maxshrink=1/sqrt(nsubtrain);
  minshrink=1/nsubtrain;
  allshrink=(1-maxshrink)+(maxshrink-minshrink)*rand(1,maxiter);

  for iter=1:maxiter
    f=allf(iter);
    rfac=allrfac(iter);
    lambda=alllambda(iter);
    logisticiter=alllogisticiter(iter);
    eta=alleta(iter);
    alpha=allalpha(iter);
    decay=alldecay(iter);
    kern=kerntags{allkern(iter)};
    shrink=allshrink(iter);
    
    clear wr b ww;
    try
      if (nargin > 1 && strcmp(varargin{1},'linear'))
       wr=0;
       b=0;
       ww=megamls(subtrainh,subtrainleft,...
                  struct('lambda',lambda,'f',f,'fbs',1000,...
                         'rfac',rfac,'kernel',kern,...
                         'logisticiter',4*logisticiter,'eta',40*eta,...
                         'alpha',0.5+0.5*alpha,'decay',decay));
      else
       [wr,b,ww]=calmultimls(subtrainh,subtrainleft,...
                             struct('lambda',64*lambda,'f',f,'fbs',500,...
                                    'rfac',rfac,'kernel',kern,...
                                    'logisticiter',logisticiter,'eta',eta,...
                                    'alpha',alpha,'decay',decay,...
                                    'shrink',shrink,'multiclass',true));
      end
    catch
      fprintf('*');
      continue
    end
    thres=(logisticiter==0)*0.5;
    fprintf('.');
    [~,~,~,macroF1,macroF1lb,macroF1ub]=multiF1Boot(wr,b,ww,subppos,valtrainh,valtrainleft,false);
    [teste,testelb,testeub,microF1,microF1lb,microF1ub,merror,merrorlb,merrorub]=multiHammingBoot(thres,wr,b,ww,valtrainh,valtrainleft,false);
        
    if (~exist('bestham','var'))
      bestham=struct('iter',iter,'loss',[teste,testelb,testeub]);
      fprintf('\niter = %u, bestham.loss = %g',iter,bestham.loss(1));
      bestmicro=struct('iter',iter,'loss',[microF1,microF1lb,microF1ub]);
      fprintf('\niter = %u, bestmicro.loss = %g',iter,bestmicro.loss(1));
      bestmacro=struct('iter',iter,'loss',[macroF1,macroF1lb,macroF1ub]);
      fprintf('\niter = %u, bestmacro.loss = %g',iter,bestmacro.loss(1));
      besterror=struct('iter',iter,'loss',[merror,merrorlb,merrorub]);
      fprintf('\niter = %u, besterror.loss = %g',iter,besterror.loss(1));
    else
      if (teste < bestham.loss(1))
        bestham=struct('iter',iter,'loss',[teste,testelb,testeub]);
        fprintf('\niter = %u, bestham.loss = %g',iter,bestham.loss(1));
      end
      if (merror < besterror.loss(1))
        besterror=struct('iter',iter,'loss',[merror,merrorlb,merrorub]);
        fprintf('\niter = %u, besterror.loss = %g',iter,besterror.loss(1));
      end
      if (microF1 > bestmicro.loss(1))
        bestmicro=struct('iter',iter,'loss',[microF1,microF1lb,microF1ub]);
        fprintf('\niter = %u, bestmicro.loss = %g',iter,bestmicro.loss(1));
      end
      if (macroF1 > bestmacro.loss(1))
        bestmacro=struct('iter',iter,'loss',[macroF1,macroF1lb,macroF1ub]);
        fprintf('\niter = %u, bestmacro.loss = %g',iter,bestmacro.loss(1));
      end         
    end
  end
  
  fprintf('\n');
  which=1;
  ppos=full(sum(trainleft,1))/size(trainleft,1);
  for iter=[bestham.iter bestmicro.iter bestmacro.iter besterror.iter]
    f=allf(iter);
    rfac=allrfac(iter);
    lambda=alllambda(iter);
    logisticiter=alllogisticiter(iter);
    eta=alleta(iter);
    alpha=allalpha(iter);
    decay=alldecay(iter);
    kern=kerntags{allkern(iter)};
    shrink=allshrink(iter);
  
    tic
    clear wr b ww;
    if (nargin > 1 && strcmp(varargin{1},'linear'))
       wr=0;
       b=0;
       ww=megamls(trainh,trainleft,...
                  struct('lambda',lambda,'f',f,'fbs',1000,...
                         'rfac',rfac,'kernel',kern,...
                         'logisticiter',4*logisticiter,'eta',40*eta,...
                         'alpha',0.5+0.5*alpha,'decay',decay));
    else
       [wr,b,ww]=calmultimls(trainh,trainleft,...
                             struct('lambda',64*lambda,'f',f,'fbs',500,...
                                    'rfac',rfac,'kernel',kern,...
                                    'logisticiter',logisticiter,'eta',eta,...
                                    'alpha',alpha,'decay',decay,...
                                    'shrink',shrink,'multiclass',true));
    end
    thres=(logisticiter==0)*0.5;
    fprintf('iter=%u f=%g rfac=%g lambda=%g logisticiter=%u eta=%g alpha=%g decay=%g kern=%s shrink=%g\n',...
            iter,f,rfac,lambda,logisticiter,eta,alpha,decay,kern,shrink);
    if (which == 1 || which == 2 || which == 4)
      [teste,testelb,testeub,microF1,microF1lb,microF1ub,merror,merrorlb,merrorub]=multiHammingBoot(thres,wr,b,ww,testh,testleft,true);
      if (which == 1)
        res.calmls_embed_test_hamming=[testelb,teste,testeub];
      elseif (which == 4)
        res.calmls_embed_test_error=[merrorlb,merror,merrorub];
      else
        res.calmls_embed_test_microF1=[microF1lb,microF1,microF1ub];
      end
    else
      [~,~,~,macroF1,macroF1lb,macroF1ub]=multiF1Boot(wr,b,ww,ppos,testh,testleft,true);
      res.calmls_embed_test_macroF1=[macroF1lb,macroF1,macroF1ub];
    end
    which=which+1;
  end
         
  toc(start)
end

function [yhati,yhatj,ymax]=blockmultiinfer(X,th,wr,b,ww)
  if (size(wr,1)==1 && size(wr,2)==1 && wr==0)
    Z=X*ww;
  else
    [fplus1,~]=size(ww);
    f=fplus1-1;
    Z=cos(bsxfun(@plus,X*wr,b))*ww(1:f,:);
    Z=bsxfun(@plus,Z,ww(fplus1,:));
  end
  [yhati,yhatj]=find(Z>th);
  [~,ymax]=max(Z,[],2);
end

function [loss,truepos,falsepos,falseneg,multierror]=multiHammingImpl(th,wr,b,ww,X,Y,imp)
  [~,f]=size(wr);
  [n,k]=size(Y);
  [~,s]=size(imp);
  
  truepos=zeros(1,s);
  falsepos=zeros(1,s);
  falseneg=zeros(1,s);
  multierror=zeros(1,s);
  bs=min(n,ceil(1e+9/max(k,f)));
  for off=1:bs:n
      offend=min(n,off+bs-1);
      mybs=offend-off+1;
      [yhati,yhatj,ymax]=blockmultiinfer(X(off:offend,:),th,wr,b,ww);
      [i,j,~]=find(Y(off:offend,:));
      
      for ss=1:s
        weights=imp(i+(off-1),ss);
        megay=sparse(i,j,weights,mybs,k); clear weights;
        weightshat=imp(yhati+(off-1),ss);
        megayhat=sparse(yhati,yhatj,weightshat,mybs,k); clear weightshat;
        bstruepos=sum(megay(sub2ind(size(megay),yhati,yhatj)));
        bsfalsepos=sum(sum(megayhat))-bstruepos;
        bsfalseneg=sum(sum(megay))-bstruepos;
        truepos(ss)=truepos(ss)+bstruepos;
        falsepos(ss)=falsepos(ss)+bsfalsepos;
        falseneg(ss)=falseneg(ss)+bsfalseneg;

        [~,y]=max(megay,[],2);
        multierror(ss)=multierror(ss)+sum(imp(off:offend,ss).*(y~=ymax));
        
        clear megay megayhat;
      end
  end
  
  total=k*sum(imp,1);
  loss=(falsepos+falseneg)./total;
  multierror=multierror./sum(imp,1);
end

function [microF1,macroF1]=F1metric(truepos,falsepos,falseneg)
  classprec=truepos./(truepos+falsepos+1e-12);
  classrec=truepos./(truepos+falseneg+1e-12);
  % http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
  macroF1=mean(2*(classprec.*classrec)./(classprec+classrec+1e-12),1);
  % http://www.kaggle.com/c/lshtc/details/evaluation
  % MaP=mean(classprec,1);
  % MaR=mean(classrec,1);
  % macroF1=2*MaP*MaR./(MaP+MaR+1e-12);
  
  prec=sum(truepos,1)./(sum(truepos,1)+sum(falsepos,1)+1e-12);
  rec=sum(truepos,1)./(sum(truepos,1)+sum(falseneg,1)+1e-12);
  microF1=2*(prec.*rec)./(prec+rec+1e-12);
end

function [loss,cilb,ciub,microloss,microlb,microub,merror,merrorlb,merrorub]=multiHammingBoot(th,wr,b,ww,testh,testy,doprint)
  [m,~]=size(testh);
  imp=poissrnd(1,m,16);
  [testloss,testtruepos,testfalsepos,testfalseneg,multierror]=...
      multiHammingImpl(th,wr,b,ww,testh,testy,imp);
  
  [~,ind]=sort(testloss);
  loss=mean(testloss(ind(8:9)));
  cilb=testloss(ind(2));
  ciub=testloss(ind(15));
  
  [~,ind]=sort(multierror);
  merror=mean(multierror(ind(8:9)));
  merrorlb=multierror(ind(2));
  merrorub=multierror(ind(15));
    
  [microF1,~]=F1metric(testtruepos,testfalsepos,testfalseneg);
  [~,ind]=sort(microF1);
  microloss=mean(microF1(ind(8:9)));
  microlb=microF1(ind(2));
  microub=microF1(ind(15));
 
  if (doprint)
    fprintf('per-ex inference: (hamming) [%g,%g,%g] (micro) [%g,%g,%g] (multiclass) [%g,%g,%g]\n',...
            cilb,loss,ciub,microlb,microloss,microub,merrorlb,merror,merrorub);
  end
end

function [microF1,macroF1]=F1Impl(wr,b,ww,trainh,trainy,ppos,imp)
  persistent havemex;

  if (isempty(havemex))
    havemex = (exist('fastsoftmax', 'file') == 3);
  end

  [~,f]=size(wr);
  [n,k]=size(trainy);
  kbs=min(k,ceil(1e+9/n));
  [~,s]=size(imp);
  
  truepos=zeros(k,s);
  falsepos=zeros(k,s);
  falseneg=zeros(k,s);
  for off=1:kbs:k
    offend=min(k,off+kbs-1);
    if (size(wr,1)==1 && size(wr,2)==1 && wr==0)
      Z=trainh*ww(:,off:offend);
    else
      Z=cos(bsxfun(@plus,trainh*wr,b))*ww(1:f,off:offend);
      Z=bsxfun(@plus,Z,ww(f+1,off:offend));
    end
    if (havemex)
      fastsoftmax(Z,max(Z));
    else
      Z=bsxfun(@minus,Z,max(Z));
      Z=exp(Z);
      Z=bsxfun(@rdivide,Z,sum(Z));
    end
    [allsp,allind]=sort(Z,'descend');
    r=bsxfun(@plus,linspace(1,n,n)'*ones(1,offend-off+1),n*ppos(off:offend));
    fscores=bsxfun(@rdivide,cumsum(allsp),r);
    [~,am]=max(fscores);

    for jj=off:offend
      yhati=allind(1:am(jj-off+1),jj-off+1);
      i=find(trainy(:,jj));

      bstruepos=sum(imp(intersect(i,yhati),:),1);
      bsfalsepos=sum(imp(setdiff(yhati,i),:),1);
      bsfalseneg=sum(imp(setdiff(i,yhati),:),1);
      truepos(jj,:)=truepos(jj,:)+bstruepos;
      falsepos(jj,:)=falsepos(jj,:)+bsfalsepos;
      falseneg(jj,:)=falseneg(jj,:)+bsfalseneg;
    end
  end  
  
  [microF1,macroF1]=F1metric(truepos,falsepos,falseneg);
end

function [microF1,microF1lb,microF1ub,macroF1,macroF1lb,macroF1ub]=multiF1Boot(wr,b,ww,ppos,testh,testy,doprint)
  [m,~]=size(testh);
  imp=poissrnd(1,m,16);
  [testmicroF1,testmacroF1]=F1Impl(wr,b,ww,testh,testy,ppos,imp);
  
  [~,ind]=sort(testmicroF1);
  microF1=mean(testmicroF1(ind(8:9)));
  microF1lb=testmicroF1(ind(2));
  microF1ub=testmicroF1(ind(15));
  
  [~,ind]=sort(testmacroF1);
  macroF1=mean(testmacroF1(ind(8:9)));
  macroF1lb=testmacroF1(ind(2));
  macroF1ub=testmacroF1(ind(15));

  if (doprint)
    fprintf('per-class inference: (micro) [%g,%g,%g] (macro) [%g,%g,%g]\n',...
            microF1lb,microF1,microF1ub,...
            macroF1lb,macroF1,macroF1ub);
  end
end
