%Code for Ploting Figure 2 right in section 5.4
clear all;
L2 = zeros(100,1);
folder = '../result/parameter/nlayer/swish_0.1';
for epoch = 1:100
    load([folder, '/L0.0005_2_1/',num2str(epoch), '_conv_weight.mat']);
    conv_weight1 = conv_weight;
    load([folder, '/L0.0005_2_2/',num2str(epoch), '_conv_weight.mat']);
    conv_weight2 = conv_weight;
    L2(epoch,1) = 2*mean(mean((abs(conv_weight1(:) - conv_weight2(:)))./(abs(conv_weight1(:)) + abs(conv_weight2(:)))));

    load([folder, '/L0.0005_4_1/',num2str(epoch), '_conv_weight.mat']);
    conv_weight1 = conv_weight;
    load([folder, '/L0.0005_4_2/',num2str(epoch), '_conv_weight.mat']);
    conv_weight2 = conv_weight;
    L2(epoch,2) = 2*mean(mean((abs(conv_weight1(:) - conv_weight2(:)))./(abs(conv_weight1(:)) + abs(conv_weight2(:)))));


    load([folder,'/L0.0005_8_1/',num2str(epoch), '_conv_weight.mat']);
    conv_weight1 = conv_weight;
    load([folder,'/L0.0005_8_2/',num2str(epoch), '_conv_weight.mat']);
    conv_weight2 = conv_weight;
    L2(epoch,3) = 2*mean(mean((abs(conv_weight1(:) - conv_weight2(:)))./(abs(conv_weight1(:)) + abs(conv_weight2(:)))));

        
    load([folder,'/L0.0005_16_1/',num2str(epoch), '_conv_weight.mat']);
    conv_weight1 = conv_weight;
    load([folder,'/L0.0005_16_2/',num2str(epoch), '_conv_weight.mat']);
    conv_weight2 = conv_weight;
    L2(epoch,4) = 2*mean(mean((abs(conv_weight1(:) - conv_weight2(:)))./(abs(conv_weight1(:)) + abs(conv_weight2(:)))));

        
    load([folder,'/L0.0005_32_1/',num2str(epoch), '_conv_weight.mat']);
    conv_weight1 = conv_weight;
    load(['./', folder,'/L0.0005_32_2/',num2str(epoch), '_conv_weight.mat']);
    conv_weight2 = conv_weight;
    L2(epoch,5) = 2*mean(mean((abs(conv_weight1(:) - conv_weight2(:)))./(abs(conv_weight1(:)) + abs(conv_weight2(:)))));
        
    load([folder,'/L0.0005_320_1/',num2str(epoch), '_conv_weight.mat']);
    conv_weight1 = conv_weight;
    load([folder,'/L0.0005_320_2/',num2str(epoch), '_conv_weight.mat']);
    conv_weight2 = conv_weight;
    L2(epoch,6) = 2*mean(mean((abs(conv_weight1(:) - conv_weight2(:)))./(abs(conv_weight1(:)) + abs(conv_weight2(:)))));
        
    load([folder,'/L0.0005_3200_1/',num2str(epoch), '_conv_weight.mat']);
    conv_weight1 = conv_weight;
    load([folder,'/L0.0005_3200_2/',num2str(epoch), '_conv_weight.mat']);
    conv_weight2 = conv_weight;
    L2(epoch,7) = 2*mean(mean((abs(conv_weight1(:) - conv_weight2(:)))./(abs(conv_weight1(:)) + abs(conv_weight2(:)))));
        
    load([folder,'/L0.0005_32000_1/',num2str(epoch), '_conv_weight.mat']);
    conv_weight1 = conv_weight;
    load([folder,'/L0.0005_32000_2/',num2str(epoch), '_conv_weight.mat']);
    conv_weight2 = conv_weight;
    L2(epoch,8) = 2*mean(mean((abs(conv_weight1(:) - conv_weight2(:)))./(abs(conv_weight1(:)) + abs(conv_weight2(:)))));
  
end
figure;  
plot(L2,'Linewidth', 1.4)
grid on
set(gca,'FontSize',18);
legend('lr = 5e-2', 'lr = 2.5e-2', 'lr = 1.25e-2', 'lr = 6.25e-3', 'lr = 3.125e-3', 'lr = 3.125e-4', 'lr = 3.125e-5', 'lr = 3.125e-6', 'Location','NorthWest', 'FontSize',13);
ylabel('Average Relative L1 Difference','fontsize',18)
xlabel('Scaled Iteration','fontsize',18)





