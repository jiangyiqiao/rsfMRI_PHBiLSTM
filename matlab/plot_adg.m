adg = (0.1:0.1:0.9);
acc = [46.67,46.67,66.67,73.33,73.33,86.67,86.67,80,73.33]
% adg 对准确率影响

plot(adg,acc,'*-');
xlabel('R1 parameter');ylabel('Accuracy(ACC)');
grid on;