function res=traintestlshtc()
  addpath('../matlab/');
  prefix='./';
  
  if exist('poissrnd') == 0
    error('*** you are missing poissrnd: try ''make poissrnd.m'' ***')
  end

  randn('seed',8675309);
  rand('seed',90210);
  
  start=tic;
  fprintf('loading data ... ');
  instruct=struct('NumHeaderLines',0, ...
                  'NumColumns',3, ...
                  'Format', '%f %f %f', ...
                  'InfoLevel', 0);
  leftraw=txt2mat(strcat(prefix,'left'),instruct);
  left=spconvert(leftraw);
  clear leftraw;
  rightraw=txt2mat(strcat(prefix,'right'),instruct);
  right=spconvert(rightraw);
  clear rightraw;
  toc(start)

  % train-test split

  perm=randperm(size(right,1));
  split=ceil(0.75*size(right,1));

  right=right';
  trainright=right(:,perm(1:split));
  testright=right(:,perm(split+1:end));
  %right=right';
  clear right;
  trainright=trainright';
  testright=testright';

  left=left';
  trainleft=left(:,perm(1:split));
  testleft=left(:,perm(split+1:end));
  %left=left';
  clear left;
  trainleft=trainleft';
  testleft=testleft';

  [n,d]=size(trainright);
  [m,~]=size(testright);

  % document length normalization
  wordcount=full(sum(trainright,2));
  trainright=spdiags(1./wordcount,0,n,n)*trainright;
  wordcount=full(sum(testright,2));
  testright=spdiags(1./wordcount,0,m,m)*testright;
  clear wordcount;

  % add constant feature
  trainright=horzcat(trainright,ones(n,1));
  testright=horzcat(testright,ones(m,1));
    
  ultraw=ones(1,size(trainleft,1));
  
  embedd=800;
  tic
  rs=rembed(trainright',ultraw,trainleft',embedd, ...
            struct('pre','diag','innerloop',50,'lambda',128,'kbs',200,...
                   'tmax',2,'trainpredictor',true,'verbose',true));
  toc
  
  trainh=horzcat(rs.projectx(trainright),ones(n,1));
  testh=horzcat(rs.projectx(testright),ones(m,1));
  ppos=full(sum(trainleft,1))/size(trainleft,1);
  trainstart=tic;
  clear wr b ww;
  [wr,b,ww]=calmultimls(trainh,trainleft,...
                        struct('lambda',5e-4,'f',4000,'fbs',1000,...
                               'direct',false,'rfac',0.1,'kernel','c',...
                               'logisticiter',20,'focused',true,...
                               'monfunc',@(wr,b,ww,th) multiPrecAt1(wr,b,ww,testh,testleft,true,trainstart)));
  toc(trainstart)
  [~,teste,~]=multiPrecAt1(wr,b,ww,testh,testleft,false,trainstart);
  res.calmultimls_embed_test_error=teste;

  toc(start)
end

function [yhat,Z]=blockinfer(X,wr,b,ww)
  [fplus1,~]=size(ww);
  f=fplus1-1;
  Z=cos(bsxfun(@plus,X*wr,b))*ww(1:f,:);
  Z=bsxfun(@plus,Z,ww(fplus1,:));
  [~,yhat]=max(Z,[],2);
end

function [testelb,teste,testeub]=multiPrecAt1(wr,b,ww,testh,testleft,doprint,ts)
  dt=toc(ts);
  
  [~,f]=size(wr);
  [m,k]=size(testleft);
  imp=poissrnd(1,m,16);
  [~,s]=size(imp);
  
  testgood=zeros(1,s);
  testtotal=sum(imp,1);
  bs=min(m,ceil(1e+9/max(k,f)));
  for off=1:bs:m
      offend=min(m,off+bs-1);
      mybs=offend-off+1;
      testyhat=blockinfer(testh(off:offend,:),wr,b,ww);
      [i,j,~]=find(testleft(off:offend,:));
      for ss=1:s
        megay=sparse(i,j,imp(i,ss),mybs,k);
        testgood(ss)=testgood(ss)+sum(megay(sub2ind(size(megay),(1:mybs)',testyhat)));
        clear megay;
      end
  end
  testeboot=testgood./testtotal;
  [~,ind]=sort(testeboot);
  testelb=testeboot(ind(2));
  teste=mean(testeboot(ind(8:9)));
  testeub=testeboot(ind(15));
  if (doprint)
    fprintf('[%g,%g,%g] %g\n',testelb,teste,testeub,dt);
  end
end

% traintestlshtclowrank
% ...
% Elapsed time is 21735.284404 seconds.
% [0.314066,0.321634,0.326393] 1108.2
% [0.439703,0.449689,0.459077] 53761.9
% [0.472215,0.481135,0.486277] 106301
% [0.480235,0.497382,0.510262] 158791
% [0.496519,0.51221,0.522006] 211237
% [0.499025,0.514792,0.532844] 263749
% [0.516133,0.526161,0.536645] 316325
% [0.518794,0.523671,0.536948] 368778
% [0.519409,0.531646,0.543682] 421331
% [0.525258,0.538932,0.54743] 475376
% [0.517964,0.532429,0.541099] 529511
