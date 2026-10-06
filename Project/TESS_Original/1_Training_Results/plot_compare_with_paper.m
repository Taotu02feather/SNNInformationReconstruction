clear; clc; close all;

compare_dir = 'paper_compare';
save_dir = 'paper_compare';

if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

%% =========================================================
% 1) Alpha ablation 对比
% 每个 dataset 一张图：x轴是 alpha_post, 两组柱：paper / mine
%% =========================================================
alpha_file = fullfile(compare_dir, 'alpha_post_vs_paper.csv');

if exist(alpha_file, 'file')
    T = readtable(alpha_file);

    datasets = unique(T.dataset, 'stable');
    alpha_order = [-1, 0, 1];

    for i = 1:length(datasets)
        ds = datasets{i};
        S = T(strcmp(T.dataset, ds), :);

        M = nan(length(alpha_order), 2); % col1=paper, col2=my

        for j = 1:length(alpha_order)
            a = alpha_order(j);
            idx = S.alpha_post == a;
            if any(idx)
                row = S(find(idx, 1), :);
                M(j, 1) = row.paper_acc;
                M(j, 2) = row.my_acc;
            end
        end

        fig = figure('Color', 'w');
        bar(M, 'grouped');
        grid on; box on;

        set(gca, ...
            'XTick', 1:3, ...
            'XTickLabel', {'$\alpha_{post}=-1$', '$\alpha_{post}=0$', '$\alpha_{post}=1$'}, ...
            'TickLabelInterpreter', 'latex', ...
            'FontSize', 12);

        ylabel('Accuracy (\%)', 'Interpreter', 'latex');
        title(sprintf('%s: paper vs mine ($\\alpha_{post}$ ablation)', ds), 'Interpreter', 'latex');
        legend({'Paper', 'Mine'}, 'Interpreter', 'latex', 'Location', 'best');

        exportgraphics(fig, fullfile(save_dir, sprintf('%s_alpha_post_vs_paper.png', ds)), 'Resolution', 300);
        savefig(fig, fullfile(save_dir, sprintf('%s_alpha_post_vs_paper.fig', ds)));
    end
end


%% =========================================================
% 2) BPTT / S-TLLR / TESS 对比
% 每个 dataset 一张图：x轴是 method, 两组柱：paper / mine
%% =========================================================
compare_file = fullfile(compare_dir, 'bptt_tess_vs_paper.csv');

if exist(compare_file, 'file')
    T = readtable(compare_file);

    datasets = unique(T.dataset, 'stable');
    method_order = {'BPTT', 'S-TLLR', 'TESS'};

    for i = 1:length(datasets)
        ds = datasets{i};
        S = T(strcmp(T.dataset, ds), :);

        M = nan(length(method_order), 2); % col1=paper, col2=my

        for j = 1:length(method_order)
            m = method_order{j};
            idx = strcmp(S.method, m);
            if any(idx)
                row = S(find(idx, 1), :);
                M(j, 1) = row.paper_acc;
                M(j, 2) = row.my_acc;
            end
        end

        fig = figure('Color', 'w');
        bar(M, 'grouped');
        grid on; box on;

        set(gca, ...
            'XTick', 1:length(method_order), ...
            'XTickLabel', method_order, ...
            'TickLabelInterpreter', 'latex', ...
            'FontSize', 12);

        ylabel('Accuracy (\%)', 'Interpreter', 'latex');
        title(sprintf('%s: paper vs mine (BPTT / S-TLLR / TESS)', ds), 'Interpreter', 'latex');
        legend({'Paper', 'Mine'}, 'Interpreter', 'latex', 'Location', 'best');

        exportgraphics(fig, fullfile(save_dir, sprintf('%s_bptt_tess_vs_paper.png', ds)), 'Resolution', 300);
        savefig(fig, fullfile(save_dir, sprintf('%s_bptt_tess_vs_paper.fig', ds)));
    end

else

    TestStr = 'Something is Wrong\n';
    fprintf(TestStr)

end


fprintf('[DONE] Figures saved to: %s\n', save_dir);