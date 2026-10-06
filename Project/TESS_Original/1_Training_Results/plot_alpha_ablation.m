clear; clc; close all;

curve_dir = 'alpha_post_curves';
summary_file = fullfile('alpha_post_summary', 'alpha_ablation_summary.csv');
save_dir = 'alpha_post_figures';

if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

files = dir(fullfile(curve_dir, '*_curve.csv'));

if isempty(files)
    error('No alpha curve csv files found in folder: %s', curve_dir);
end

dataset_names = {};
info = struct([]);

% =========================
% 读取并解析文件名
% 文件名格式应类似:
%   CIFAR10_alpha-1_curve.csv
%   CIFAR100_alpha0_curve.csv
%   DVSGesture_alpha1_curve.csv
% =========================
for i = 1:length(files)
    fname = files(i).name;
    token = regexp(fname, '(.+)_alpha(-?1|0|1)_curve\.csv', 'tokens');

    if isempty(token)
        fprintf('[SKIP] Unrecognized curve file: %s\n', fname);
        continue;
    end

    token = token{1};
    dataset = token{1};
    alpha_val = str2double(token{2});

    dataset_names{end+1} = dataset; %#ok<SAGROW>
    info(end+1).name = fname; %#ok<SAGROW>
    info(end).dataset = dataset;
    info(end).alpha = alpha_val;
end

dataset_names = unique(dataset_names, 'stable');

% 你可以改成 'train_acc1'
metric_name = 'test_acc1';

% =========================
% 每个数据集画一张曲线图
% =========================
for d = 1:length(dataset_names)
    dataset = dataset_names{d};

    fig = figure('Name', ['Alpha Ablation - ' dataset], 'Color', 'w');
    hold on; grid on; box on;

    legend_entries = {};
    target_alphas = [-1, 0, 1];

    for a = 1:length(target_alphas)
        alpha_val = target_alphas(a);

        idx = find(strcmp({info.dataset}, dataset) & [info.alpha] == alpha_val, 1);
        if isempty(idx)
            fprintf('[WARN] Missing %s alpha=%d curve.\n', dataset, alpha_val);
            continue;
        end

        T = readtable(fullfile(curve_dir, info(idx).name));

        if ~ismember('epoch', T.Properties.VariableNames)
            fprintf('[WARN] Missing epoch column in %s\n', info(idx).name);
            continue;
        end

        if ~ismember(metric_name, T.Properties.VariableNames)
            fprintf('[WARN] Missing %s column in %s\n', metric_name, info(idx).name);
            continue;
        end

        plot(T.epoch, T.(metric_name), 'LineWidth', 2);

        legend_entries{end+1} = sprintf('$\\alpha_{post} = %d$', alpha_val); %#ok<SAGROW>
    end

    xlabel('Epoch', 'Interpreter', 'latex');
    ylabel('Test Acc@1', 'Interpreter', 'latex');
    title(sprintf('%s: $\\alpha_{post}$ ablation', dataset), 'Interpreter', 'latex');
    legend(legend_entries, 'Location', 'best', 'Interpreter', 'latex');

    set(gca, 'FontSize', 12);
    set(gca, 'TickLabelInterpreter', 'latex');

    % 保存图
    png_path = fullfile(save_dir, sprintf('%s_alpha_ablation_curve.png', dataset));
    fig_path = fullfile(save_dir, sprintf('%s_alpha_ablation_curve.fig', dataset));

    exportgraphics(fig, png_path, 'Resolution', 300);
    savefig(fig, fig_path);
end

% =========================
% 汇总柱状图
% =========================
if exist(summary_file, 'file')
    S = readtable(summary_file);

    required_cols = {'dataset', 'alpha_post', 'best_acc'};
    for k = 1:length(required_cols)
        if ~ismember(required_cols{k}, S.Properties.VariableNames)
            error('Summary file missing required column: %s', required_cols{k});
        end
    end

    datasets = unique(S.dataset, 'stable');
    alphas = [-1, 0, 1];
    M = nan(length(datasets), length(alphas));

    for i = 1:length(datasets)
        for j = 1:length(alphas)
            idx = strcmp(S.dataset, datasets{i}) & S.alpha_post == alphas(j);
            if any(idx)
                M(i, j) = S.best_acc(find(idx, 1));
            end
        end
    end

    fig = figure('Name', 'Alpha Ablation Summary', 'Color', 'w');
    bar(M, 'grouped');
    grid on; box on;

    set(gca, ...
        'XTick', 1:length(datasets), ...
        'XTickLabel', datasets, ...
        'FontSize', 12, ...
        'TickLabelInterpreter', 'latex');

    xlabel('Dataset', 'Interpreter', 'latex');
    ylabel('Best Acc@1', 'Interpreter', 'latex');
    title('Ablation study on including non-causal terms', 'Interpreter', 'latex');

    legend({ ...
        '$\alpha_{post}=-1$', ...
        '$\alpha_{post}=0$', ...
        '$\alpha_{post}=1$' ...
        }, 'Interpreter', 'latex', 'Location', 'best');

    png_path = fullfile(save_dir, 'alpha_ablation_summary_bar.png');
    fig_path = fullfile(save_dir, 'alpha_ablation_summary_bar.fig');

    exportgraphics(fig, png_path, 'Resolution', 300);
    savefig(fig, fig_path);
else
    fprintf('[WARN] Summary file not found: %s\n', summary_file);
end

fprintf('[DONE] Alpha ablation figures saved to folder: %s\n', save_dir);