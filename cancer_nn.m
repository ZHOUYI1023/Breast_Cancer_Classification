clear
clc
rng('default')  % For reproducibility
load cancer_dataset.mat
inputs = cancerInputs;
targets = cancerTargets;
hiddenLayerSize = [2 4 8 32];
trainFcn = 'trainscg';
performFcn = 'crossentropy';
nEpochs = [4 8 16 32 64];

loop = 30;
elapsed = zeros(1,length(hiddenLayerSize));

for i = 1:length(hiddenLayerSize)
trMeanErrorRate = zeros(1,length(nEpochs));
tsMeanErrorRate = zeros(1,length(nEpochs));
trStdErrorRate = zeros(1,length(nEpochs));
tsStdErrorRate = zeros(1,length(nEpochs));

for j = 1:length(nEpochs)
if nEpochs(j) == 16
    tic;
end
trErrorRate = zeros(1,loop);
tsErrorRate = zeros(1,loop);



net = patternnet(hiddenLayerSize(i), trainFcn,performFcn);
net.trainParam.epochs = nEpochs(j);
net.trainParam.showWindow = false;

net.divideParam.trainRatio = 0.5;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0.5;
for l=1:loop
net = init(net);
[net,tr] = train(net,inputs,targets);

target = targets(1,:);
trTarg = [target(tr.trainInd);1-target(tr.trainInd)];
tsTarg = [target(tr.testInd);1-target(tr.testInd)];

outputs = net(inputs);
output = outputs(1,:);
trOut = [output(tr.trainInd); 1-output(tr.trainInd)];
tsOut = [output(tr.testInd); 1-output(tr.testInd)];

[trErrorRate(l),~,~,~] = confusion(trTarg,trOut);
[tsErrorRate(l),~,~,~] = confusion(tsTarg,tsOut);
performance = perform(net,targets,outputs);
end
trMeanErrorRate(j) = mean(trErrorRate);
trStdErrorRate(j) = std(trErrorRate);
tsMeanErrorRate(j) = mean(tsErrorRate);
tsStdErrorRate(j) = std(tsErrorRate);

if(nEpochs(j) == 16)
    elapsed(i) = toc;
end

end

figure(1)
subplot(2,2,i)
hold on
plot(nEpochs,trMeanErrorRate,'x-')
plot(nEpochs,tsMeanErrorRate,'x-')
plot(nEpochs,trStdErrorRate,'x-')
plot(nEpochs,tsStdErrorRate,'x-')
legend('TrainingErrorRateMean','TestingErrorRateMean','TrainingErrorRateStd','TestingErrorRateStd')
xlabel('Epochs')
ylabel('Value')
t = 'Node = '+ string(hiddenLayerSize(i));
title(t)
ax = gca;
ax.FontSize = 12;
set(findall(gcf,'type','line'),'linewidth',2);
axis([0 70 0 0.1])
% 
figure(2)
X = categorical({'2 Nodes','4 Nodes','8 Nodes','32 Nodes'});
X = reordercats(X,{'2 Nodes','4 Nodes','8 Nodes','32 Nodes'});
Y = elapsed/50/16;
bar(X,Y)
ylabel('Elapsed Time Per Epoch/s')
ax = gca;
ax.FontSize = 12;
set(findall(gcf,'type','line'),'linewidth',2);

end

% view(net)
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotconfusion(targets,outputs)
% figure, ploterrhist(errors)