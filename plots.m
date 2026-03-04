clearvars;
close all;

kernel = [0, 1, 2, 3, 4, 5, 6];
time = [99.1749, 67.6059, 157.0669, 67.9323, 16.57148, 12.66285, 7.83214];
speedup = time(1) ./ time;

figure;
subplot(1, 2, 1);
p = plot(kernel, time, '-o', 'LineWidth', 1);
p.MarkerFaceColor = p.Color;
xlabel('Kernel version');
ylabel('Sum of op times (ms)');
title('Kernel Op Times');
grid on;

subplot(1, 2, 2);
p = plot(kernel, speedup, '-o', 'LineWidth', 1);
p.MarkerFaceColor = p.Color;
xlabel('Kernel version');
ylabel('Relative speedup');
title('Kernel Speedup');
grid on;
exportgraphics(gcf, 'speedup_plot.png', 'Resolution', 600);

