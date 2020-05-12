clear
rng('default')  % For reproducibility
R1 = mvnrnd([1 0],[1 0; 0 1],3300/2);
R2 = mvnrnd([2 0],[4 0; 0 4],3300/2);

inputs = [R1;R2]';
targets = [ones(size(R1,1),1), zeros(size(R1,1),1); zeros(size(R2,1),1), ones(size(R2,1),1)]';

trainFcn = 'trainscg';
performFcn = 'crossentropy';
hiddenLayerSize = [2 4 8 32];
nEpochs =  [4 8 16 32];
loop = 5;
% 
% 
% for i = 1:length(hiddenLayerSize)
% trMeanErrorRate = zeros(1,length(nEpochs));
% tsMeanErrorRate = zeros(1,length(nEpochs));
% trStdErrorRate = zeros(1,length(nEpochs));
% tsStdErrorRate = zeros(1,length(nEpochs));
% 
% for j = 1:length(nEpochs)
% 
% trErrorRate = zeros(1,loop);
% tsErrorRate = zeros(1,loop);
% 
% 
% 
% net = patternnet(hiddenLayerSize(i), trainFcn,performFcn);
% net.trainParam.epochs = nEpochs(j);
% 
% net.divideParam.trainRatio = 300/3300;
% net.divideParam.valRatio = 0;
% net.divideParam.testRatio = 3000/3300;
% for l=1:loop
% net = init(net);
% [net,tr] = train(net,inputs,targets);
% 
% 
% target = targets(1,:);
% trTarg = [target(tr.trainInd);1-target(tr.trainInd)];
% tsTarg = [target(tr.testInd);1-target(tr.testInd)];
% 
% outputs = net(inputs);
% output = outputs(1,:);
% trOut = [output(tr.trainInd); 1-output(tr.trainInd)];
% tsOut = [output(tr.testInd); 1-output(tr.testInd)];
% 
% [trErrorRate(l),~,~,~] = confusion(trTarg,trOut);
% [tsErrorRate(l),~,~,~] = confusion(tsTarg,tsOut);
% performance = perform(net,targets,outputs);
% end
% trMeanErrorRate(j) = mean(trErrorRate);
% trStdErrorRate(j) = std(trErrorRate);
% tsMeanErrorRate(j) = mean(tsErrorRate);
% tsStdErrorRate(j) = std(tsErrorRate);
% 
% end
% 
% figure(1)
% subplot(2,2,i)
% hold on
% plot(nEpochs,trMeanErrorRate,'x-')
% plot(nEpochs,tsMeanErrorRate,'x-')
% plot(nEpochs,trStdErrorRate,'x-')
% plot(nEpochs,tsStdErrorRate,'x-')
% legend('TrainingErrorRateMean','TestingErrorRateMean','TrainingErrorRateStd','TestingErrorRateStd')
% xlabel('Epochs')
% ylabel('Value')
% t = 'Node = '+ string(hiddenLayerSize(i));
% title(t)
% ax = gca;
% ax.FontSize = 12;
% set(findall(gcf,'type','line'),'linewidth',2);
% end


% distribution
figure(1)
plot(R1(:,1),R1(:,2),'+')
hold on
plot(R2(:,1),R2(:,2),'x')
axis equal
grid on

% bayes optimal
 u = linspace(-1.5, 3, 450);
 v = linspace(-2.5, 3.5, 600);
z_optimal = zeros(length(v), length(u));
for i = 1:length(u)
    for j = 1:length(v)
          if  ln_gaussian_2d([1;0], [1 0; 0 1], [u(i); v(j)])>ln_gaussian_2d([2;0], [4 0; 0 4], [u(i); v(j)])
            z_optimal(j,i) = 1;
          end
    end
end

figure(1)
hold on
contour(u,v,z_optimal,[0,1],'y', 'LineWidth', 2)

u = linspace(-3, 3,90);
v = linspace(-2.5, 3.5, 120);
z = zeros(length(v), length(u));

% nn
% net = patternnet(8, "trainscg",performFcn);
% net.trainParam.epochs =32;
% net.divideParam.trainRatio = 300/3300;
% net.divideParam.valRatio = 0;
% net.divideParam.testRatio = 3000/3300;
% net = init(net);
% [net,~] = train(net,inputs,targets);
% for i = 1:length(u)
%     for j = 1:length(v)
%           output = round(net([u(i); v(j)]));
%           z(j,i) = output(1);
%     end
% end

% nn_voting
net = patternnet(8, "trainscg",performFcn);
net.trainParam.epochs = 6;
nets = {};
net.divideParam.trainRatio = 300/3300;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 3000/3300;
for j = 1:8
net = init(net);
[net,~] = train(net,inputs,targets);
nets{j} = net;
end

for i = 1:length(u)
    for j = 1:length(v)
        output = 0;
        for k = 1:8
            net = nets{k};
            outputs = net([u(i); v(j)]);
            output = output + round(outputs(1,:));
        end
        z(j,i) = sign(output/8 - 0.5)/2 + 0.5;      
    end
end

figure(1)
hold on
contour(u,v,z,[0,1],'r', 'LineWidth', 2)
legend("Class 1", "Class 2", "Bayes Optimal","Nerual Net")
axis([-6,15,-8,7])