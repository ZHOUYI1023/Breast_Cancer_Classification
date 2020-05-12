clear
clc
rng(6)  % For reproducibility
load cancer_dataset.mat
inputs = cancerInputs;
targets = cancerTargets;
% shallow network
hiddenLayerSize = [2 8 32];
nEpochs = [4 8 16];
trainFcn = ["trainscg", "trainlm", "trainrp"];
performFcn = 'crossentropy';
nClassifier = [1 3 15 25];

loop = 30;

tic

% voting nodes = 4, epoch = 16
trMeanErrorRate = zeros(1,length(nClassifier));
tsMeanErrorRate = zeros(1,length(nClassifier));
trStdErrorRate = zeros(1,length(nClassifier));
tsStdErrorRate = zeros(1,length(nClassifier));
for i = 1:length(nClassifier)
trErrorRate = zeros(1,loop);
tsErrorRate = zeros(1,loop);

net = patternnet(4, "trainscg",performFcn);
net.trainParam.epochs = 16;

for l=1:loop
nets = {};
net.divideParam.trainRatio = 0.5;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0.5;
net.trainParam.showWindow = false;
for j = 1:nClassifier(i)
net = init(net);
[net,tr] = train(net,inputs,targets);
nets{j} = net;
end

% [~,tr] = train(net,inputs,targets);

target = targets(1,:);
trTarg = [target(tr.trainInd);1-target(tr.trainInd)];
tsTarg = [target(tr.testInd);1-target(tr.testInd)];

output = zeros(1,length(target));
for j = 1:nClassifier(i)
net = nets{j};
outputs = net(inputs);
output = output + round(outputs(1,:));
end

output = sign(output/nClassifier(i) - 0.5)/2 + 0.5;
% hard voting
trOut = [output(tr.trainInd); 1-output(tr.trainInd)];
tsOut = [output(tr.testInd); 1-output(tr.testInd)];

[trErrorRate(l),~,~,~] = confusion(trTarg,trOut);
[tsErrorRate(l),~,~,~] = confusion(tsTarg,tsOut);
end
trMeanErrorRate(i) = mean(trErrorRate);
trStdErrorRate(i) = std(trErrorRate);
tsMeanErrorRate(i) = mean(tsErrorRate);
tsStdErrorRate(i) = std(tsErrorRate);
end

figure(1)
hold on
plot(nClassifier,trMeanErrorRate,'x-')
plot(nClassifier,tsMeanErrorRate,'x-')
plot(nClassifier,trStdErrorRate,'x-')
plot(nClassifier,tsStdErrorRate,'x-')
legend('TrainingErrorRateMean','TestingErrorRateMean','TrainingErrorRateStd','TestingErrorRateStd')
xlabel('Number of Classifiers')
ylabel('Value')
%t = 'Number of Classifiers = '+ string(nClassifier(i));
title('4 Hidden Nodes, 16 Epochs')
axis([0 27 0 0.05])
ax = gca;
ax.FontSize = 12;
set(findall(gcf,'type','line'),'linewidth',2);

toc

% nClassifier = 15tic
trMeanErrorRate = zeros(length(hiddenLayerSize), length(nEpochs));
trStdErrorRate = zeros(length(hiddenLayerSize), length(nEpochs));
tsMeanErrorRate = zeros(length(hiddenLayerSize), length(nEpochs));
tsStdErrorRate = zeros(length(hiddenLayerSize), length(nEpochs));

for i = 1:length(hiddenLayerSize)
for j = 1:length(nEpochs)
trErrorRate = zeros(1,loop);
tsErrorRate = zeros(1,loop);

net = patternnet(hiddenLayerSize(i), "trainscg", performFcn);
net.trainParam.epochs =nEpochs(j);

for l=1:loop
nets = {};
net.divideParam.trainRatio = 0.5;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0.5;
net.trainParam.showWindow = false;
for k = 1:15
net = init(net);
[net,~] = train(net,inputs,targets);
nets{k} = net;
end

[~,tr] = train(net,inputs,targets);

target = targets(1,:);
trTarg = [target(tr.trainInd);1-target(tr.trainInd)];
tsTarg = [target(tr.testInd);1-target(tr.testInd)];

output = zeros(1,length(target));
for k = 1:15
net = nets{k};
outputs = net(inputs);
output = output + round(outputs(1,:));
end

output = sign(output/15 - 0.5)/2 + 0.5;
% hard voting
trOut = [output(tr.trainInd); 1-output(tr.trainInd)];
tsOut = [output(tr.testInd); 1-output(tr.testInd)];

[trErrorRate(l),~,~,~] = confusion(trTarg,trOut);
[tsErrorRate(l),~,~,~] = confusion(tsTarg,tsOut);
end
trMeanErrorRate(i,j) = mean(trErrorRate);
trStdErrorRate(i,j) = std(trErrorRate);
tsMeanErrorRate(i,j) = mean(tsErrorRate);
tsStdErrorRate(i,j) = std(tsErrorRate);
end
end
toc
xvalues = {'2', '8', '32'};
yvalues = {'32','16','8'};
figure(2)
ax = gca;
ax.FontSize = 12;
set(findall(gcf,'type','line'),'linewidth',2);
subplot(2,2,1)
h = heatmap(xvalues,yvalues,flipud(trMeanErrorRate));
h.XLabel = 'Hidden Nodes';
h.YLabel = 'Epochs';
h.Title = 'TrainingErrorRateMean';
subplot(2,2,2)
h = heatmap(xvalues,yvalues,flipud(tsMeanErrorRate));
h.XLabel = 'Hidden Nodes';
h.YLabel = 'Epochs';
h.Title = 'TestingErrorRateMean';
subplot(2,2,3)
h = heatmap(xvalues,yvalues,flipud(trStdErrorRate));
h.XLabel = 'Hidden Nodes';
h.YLabel = 'Epochs';
h.Title = 'TrainingErrorRateStd';
subplot(2,2,4)
h = heatmap(xvalues,yvalues,flipud(tsStdErrorRate));
h.XLabel = 'Hidden Nodes';
h.YLabel = 'Epochs';
h.Title = 'TestingErrorRateStd';




trMeanErrorRate = zeros(length(trainFcn), length(hiddenLayerSize)*length(nEpochs));
trStdErrorRate = zeros(length(trainFcn), length(hiddenLayerSize)*length(nEpochs));
tsMeanErrorRate = zeros(length(trainFcn), length(hiddenLayerSize)*length(nEpochs));
tsStdErrorRate = zeros(length(trainFcn), length(hiddenLayerSize)*length(nEpochs));
elapsedTime = zeros(length(trainFcn), length(hiddenLayerSize)*length(nEpochs));

for s = 1:length(trainFcn)
for i = 1:length(hiddenLayerSize)
for j = 1:length(nEpochs)
tic
trErrorRate = zeros(1,loop);
tsErrorRate = zeros(1,loop);

net = patternnet(hiddenLayerSize(i), trainFcn(s), performFcn);
net.trainParam.epochs =nEpochs(j);


for l=1:loop
nets = {};
net.divideParam.trainRatio = 0.5;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0.5;
net.trainParam.showWindow = false;
for k = 1:15
net = init(net);
[net,~] = train(net,inputs,targets);
nets{k} = net;
end

[~,tr] = train(net,inputs,targets);

target = targets(1,:);
trTarg = [target(tr.trainInd);1-target(tr.trainInd)];
tsTarg = [target(tr.testInd);1-target(tr.testInd)];

output = zeros(1,length(target));
for k = 1:15
net = nets{k};
outputs = net(inputs);
output = output + round(outputs(1,:));
end

output = sign(output/15 - 0.5)/2 + 0.5;
% hard voting
trOut = [output(tr.trainInd); 1-output(tr.trainInd)];
tsOut = [output(tr.testInd); 1-output(tr.testInd)];

[trErrorRate(l),~,~,~] = confusion(trTarg,trOut);
[tsErrorRate(l),~,~,~] = confusion(tsTarg,tsOut);
end
trMeanErrorRate(s, 3*i-3+j) = mean(trErrorRate);
trStdErrorRate(s, 3*i-3+j) = std(trErrorRate);
tsMeanErrorRate(s, 3*i-3+j) = mean(tsErrorRate);
tsStdErrorRate(s, 3*i-3+j) = std(tsErrorRate);
elapsedTime(s, 3*i-3+j) = toc;
end
end
end
figure(3)
for i = 1:9
subplot(3,3,i)
X = categorical({'trMeanErrorRate','tsMeanErrorRate','trStdErrorRate','tsStdErrorRate', 'elapsedTime'});
Y = [trMeanErrorRate(:,i),tsMeanErrorRate(:,i),trStdErrorRate(:,i), tsStdErrorRate(:,i), elapsedTime(:,i)/2000]';
bar(X,Y)
ylim([0, 0.06])
t = string(hiddenLayerSize(ceil(i/3))) + ' Nodes '+ string(nEpochs(i-ceil(i/3)*3+3)) + ' Epochs';
title(t)
end
