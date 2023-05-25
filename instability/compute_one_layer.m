%Code for Ploting Figure 4 right in section 5.4
clear all;
L2 = zeros(1000,1);
interval = 100;
L2_average = zeros(size(L2,1)- interval,1);
folder = '../result/parameter/1layer';

for epoch = 1:1000

    load([folder, '/dt0.01_L20/1_1/',num2str(epoch), '_conv_weight.mat']);
    conv_weight1 = conv_weight;
    load([folder, '/dt0.01_L20/1_2/',num2str(epoch), '_conv_weight.mat']);
    conv_weight2 = conv_weight;
    L2(epoch,1) = 2*mean(mean((abs(conv_weight1(:) - conv_weight2(:)))./(abs(conv_weight1(:)) + abs(conv_weight2(:)))));

    load([folder, '/dt0.03_L20/1_1/',num2str(epoch), '_conv_weight.mat']);
    conv_weight1 = conv_weight;
    load([folder, '/dt0.03_L20/1_2/',num2str(epoch), '_conv_weight.mat']);
    conv_weight2 = conv_weight;
    L2(epoch,2) = 2*mean(mean((abs(conv_weight1(:) - conv_weight2(:)))./(abs(conv_weight1(:)) + abs(conv_weight2(:)))));


    load([folder,'/dt0.05_L20/1_1/',num2str(epoch), '_conv_weight.mat']);
    conv_weight1 = conv_weight;
    load([folder,'/dt0.05_L20/1_2/',num2str(epoch), '_conv_weight.mat']);
    conv_weight2 = conv_weight;
    L2(epoch,3) = 2*mean(mean((abs(conv_weight1(:) - conv_weight2(:)))./(abs(conv_weight1(:)) + abs(conv_weight2(:)))));

        
    load([folder,'/dt0.07_L20/1_1/',num2str(epoch), '_conv_weight.mat']);
    conv_weight1 = conv_weight;
    load([folder,'/dt0.07_L20/1_2/',num2str(epoch), '_conv_weight.mat']);
    conv_weight2 = conv_weight;
    L2(epoch,4) = 2*mean(mean((abs(conv_weight1(:) - conv_weight2(:)))./(abs(conv_weight1(:)) + abs(conv_weight2(:)))));

        
    load([folder,'/dt0.09_L20/1_1/',num2str(epoch), '_conv_weight.mat']);
    conv_weight1 = conv_weight;
    load([folder,'/dt0.09_L20/1_2/',num2str(epoch), '_conv_weight.mat']);
    conv_weight2 = conv_weight;
    L2(epoch,5) = 2*mean(mean((abs(conv_weight1(:) - conv_weight2(:)))./(abs(conv_weight1(:)) + abs(conv_weight2(:)))));
        
  
end
figure;

for l = 1 : size(L2, 2)
    for i = 1 : size(L2, 1) - interval
        L2_average(i, l) = mean(L2(i:i+interval-1,l));
    end
end
        
        
plot(L2_average,'Linewidth', 1.4)
grid on
set(gca,'FontSize',18);
legend('lr = 0.01', 'lr = 0.03', 'lr = 0.05', 'lr = 0.07', 'lr = 0.09', 'Location', 'SouthEast', 'FontSize',13);
    
ylabel('Average Relative L1 Difference','fontsize',18)
xlabel('Scaled Iteration','fontsize',18)
    




