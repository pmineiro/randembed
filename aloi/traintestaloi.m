function traintestaloi()
  addpath('../matlab/');
  prefix='./';
  more off;
  
  if exist('poissrnd') == 0
    error('*** you are missing poissrnd: try ''make poissrnd.m'' ***')
  end

  if exist('aloi.mat') == 0
    warning('*** regenerating matlab data does not work under octave, only matlab ***')
    
    if exist('trainleft') == 0
      error('*** first you need to ''make rebuildmat'' before regenerating matlab data ***')
    end
    
    tic
    instruct=struct('NumHeaderLines',0, ...
                    'NumColumns',3, ...
                    'Format', '%f %f %f', ...
                    'InfoLevel', 0);
    trainleftraw=txt2mat(strcat(prefix,'trainleft'),instruct);
    trainleft=spconvert(trainleftraw);
    testleftraw=txt2mat(strcat(prefix,'testleft'),instruct);
    testleft=spconvert(testleftraw);
    trainrightraw=txt2mat(strcat(prefix,'trainright'),instruct);
    trainright=spconvert(trainrightraw);
    testrightraw=txt2mat(strcat(prefix,'testright'),instruct);
    testright=spconvert(testrightraw);
  
    xt=trainright;
    yt=trainleft;
    xs=testright;
    ys=testleft;
    save('aloi.mat','xt','yt','xs','ys','-v7.3');
    toc
  end

  fprintf('*** aloi *** \n');
  

  fprintf('***** linear ***** \n');
  traintestone('aloi.mat','linear')
  fprintf('***** calibrated ***** \n');
  traintestone('aloi.mat')
end

% >> traintestaloi
% *** aloi *** 
% Elapsed time is 4.543151 seconds.
% ***** linear ***** 
% Elapsed time is 5.805386 seconds.
% .
% iter = 1, bestham.loss = 0.476391
% iter = 1, bestmicro.loss = 0.00418068
% iter = 1, bestmacro.loss = 0.195215
% iter = 1, besterror.loss = 0.108951..
% iter = 3, bestham.loss = 0.464327
% iter = 3, bestmicro.loss = 0.00428723
% iter = 3, bestmacro.loss = 0.306526.
% iter = 4, bestham.loss = 0.462586
% iter = 4, bestmicro.loss = 0.00430403
% iter = 4, bestmacro.loss = 0.312194.
% iter = 5, bestham.loss = 0.445094
% iter = 5, bestmicro.loss = 0.00446642
% iter = 5, bestmacro.loss = 0.39809......
% iter = 11, besterror.loss = 0.103644..
% iter = 13, besterror.loss = 0.101338........
% iter = 21, bestmacro.loss = 0.399924....
% iter = 25, bestmacro.loss = 0.418894.............
% iter = 38, bestmacro.loss = 0.424668.....
% iter = 43, bestmacro.loss = 0.440836.........................................................
% iter=5 f=3871 rfac=2.38917 lambda=0.000661031 logisticiter=3 eta=0.484062 alpha=0.855809 decay=0.908807 kern=g shrink=0.998445
% per-ex inference: (hamming) [0.444702,0.44506,0.44542] (micro) [0.00446533,0.00446915,0.00447352] (multiclass) [0.168029,0.175089,0.181459]
% iter=5 f=3871 rfac=2.38917 lambda=0.000661031 logisticiter=3 eta=0.484062 alpha=0.855809 decay=0.908807 kern=g shrink=0.998445
% per-ex inference: (hamming) [0.444743,0.444963,0.445302] (micro) [0.00446755,0.00447141,0.00447399] (multiclass) [0.171448,0.175627,0.179098]
% iter=43 f=4954 rfac=1.41319 lambda=5.17992e-05 logisticiter=5 eta=0.345502 alpha=0.289332 decay=0.906478 kern=m5 shrink=0.996661
% per-class inference: (micro) [0.43548,0.441988,0.44556] (macro) [0.429614,0.434264,0.43975]
% iter=13 f=2498 rfac=1.51832 lambda=0.00416642 logisticiter=21 eta=3.07768 alpha=0.349913 decay=0.988964 kern=qg shrink=0.997137
% per-ex inference: (hamming) [0.47375,0.474058,0.474163] (micro) [0.00420024,0.00420117,0.00420389] (multiclass) [0.0949045,0.0997362,0.101455]
% Elapsed time is 4489.571921 seconds.
% 
% ans = 
% 
%     calmls_embed_test_hamming: [0.4447 0.4451 0.4454]
%     calmls_embed_test_microF1: [0.0045 0.0045 0.0045]
%     calmls_embed_test_macroF1: [0.4296 0.4343 0.4397]
%       calmls_embed_test_error: [0.0949 0.0997 0.1015]
% 
% ***** calibrated ***** 
% Elapsed time is 5.593018 seconds.
% .
% iter = 1, bestham.loss = 0.409602
% iter = 1, bestmicro.loss = 0.00485813
% iter = 1, bestmacro.loss = 0.469431
% iter = 1, besterror.loss = 0.0661569.
% iter = 2, besterror.loss = 0.064148
% iter = 2, bestmacro.loss = 0.577292.
% iter = 3, besterror.loss = 0.0453502..
% iter = 5, bestmacro.loss = 0.64917.
% iter = 6, besterror.loss = 0.0452683................
% iter = 22, bestham.loss = 0.399068
% iter = 22, bestmicro.loss = 0.00498516...
% iter = 25, bestmacro.loss = 0.658416....
% iter = 29, besterror.loss = 0.0419328.......................................
% iter = 68, besterror.loss = 0.0405786..............
% iter = 82, besterror.loss = 0.0396387..
% iter = 84, bestmacro.loss = 0.680539..
% iter = 86, besterror.loss = 0.0395229..............
% iter=22 f=3776 rfac=0.577075 lambda=0.0084128 logisticiter=24 eta=3.18195 alpha=0.834506 decay=0.90877 kern=m5 shrink=0.998125
% per-ex inference: (hamming) [0.397284,0.398427,0.399044] (micro) [0.00498698,0.00499445,0.00500896] (multiclass) [0.0679937,0.0712663,0.0740054]
% iter=22 f=3776 rfac=0.577075 lambda=0.0084128 logisticiter=24 eta=3.18195 alpha=0.834506 decay=0.90877 kern=m5 shrink=0.998125
% per-ex inference: (hamming) [0.399218,0.399643,0.400482] (micro) [0.00496917,0.00497923,0.00498391] (multiclass) [0.0661779,0.0714453,0.0752229]
% iter=84 f=725 rfac=2.92048 lambda=0.000446557 logisticiter=10 eta=0.275814 alpha=0.361849 decay=1.00841 kern=qm3 shrink=0.999759
% per-class inference: (micro) [0.678833,0.686851,0.688683] (macro) [0.681697,0.687623,0.690481]
% iter=86 f=6057 rfac=3.30041 lambda=3.11033e-05 logisticiter=18 eta=3.10791 alpha=0.260713 decay=0.92757 kern=qm3 shrink=0.999042
% per-ex inference: (hamming) [0.459922,0.460178,0.46042] (micro) [0.00432457,0.00432714,0.0043295] (multiclass) [0.0347688,0.0365135,0.038118]
% Elapsed time is 22881.108215 seconds.
% 
% ans = 
% 
%     calmls_embed_test_hamming: [0.3973 0.3984 0.3990]
%     calmls_embed_test_microF1: [0.0050 0.0050 0.0050]
%     calmls_embed_test_macroF1: [0.6817 0.6876 0.6905]
%       calmls_embed_test_error: [0.0348 0.0365 0.0381]
