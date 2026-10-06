clear; clc; close all;

curve_dir = 'bptt_tess_curves';
summary_file = fullfile('bptt_tess_summary', 'bptt_tess_summary.csv');
save_dir = 'bptt_tess_figures';

if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

files = dir(fullfile(curve_dir, '*_curve.csv'));

if isempty(files)
    error('No BPTT/TESS curve csv files found in folder: %s', curve_dir);
end

dataset_names = {};
info = struct([]);

% =========================
% 读取并解析文件名
% 文件名格式应类似:
%   CIFAR10_BPTT_curve.csv
%   CIFAR10_TESS_curve.csv
%   DVSGesture_TESS_curve.csv
% =========================
for i = 1:length(files)
    fname = files(i).name;
    token = regexp(fname, '(.+)_(BPTT|TESS)_curve\.csv', 'tokens');

    if isempty(token)
        fprintf('[SKIP] Unrecognized curve file: %s\n', fname);
        continue;
    end

    token = token{1};
    dataset = token{1};
    method = token{2};

    dataset_names{end+1} = dataset; %#ok<SAGROW>
    info(end+1).name = fname; %#ok<SAGROW>
    info(end).dataset = dataset;
    info(end).method = method;
end

dataset_names = unique(dataset_names, 'stable');

% 你可以改成 'train_acc1'
metric_name = 'test_acc1';

% =========================
% 每个数据集画一张曲线图
% =========================
for d = 1:length(dataset_names)
    dataset = dataset_names{d};

    fig = figure('Name', ['BPTT vs TESS - ' dataset], 'Color', 'w');
    hold on; grid on; box on;

    legend_entries = {};
    methods = {'BPTT', 'TESS'};

    for m = 1:length(methods)
        method = methods{m};

        idx = find(strcmp({info.dataset}, dataset) & strcmp({info.method}, method), 1);
        if isempty(idx)
            fprintf('[WARN] Missing %s %s curve.\n', dataset, method);
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
        legend_entries{end+1} = method; %#ok<SAGROW>
    end

    xlabel('Epoch', 'Interpreter', 'latex');
    ylabel('Test Acc@1', 'Interpreter', 'latex');
    title(sprintf('%s: BPTT vs TESS', dataset), 'Interpreter', 'latex');
    legend(legend_entries, 'Location', 'best', 'Interpreter', 'latex');

    set(gca, 'FontSize', 12);
    set(gca, 'TickLabelInterpreter', 'latex');

    % 保存图
    png_path = fullfile(save_dir, sprintf('%s_bptt_tess_curve.png', dataset));
    fig_path = fullfile(save_dir, sprintf('%s_bptt_tess_curve.fig', dataset));

    exportgraphics(fig, png_path, 'Resolution', 300);
    savefig(fig, fig_path);
end

% =========================
% 汇总柱状图
% =========================
if exist(summary_file, 'file')
    S = readtable(summary_file);

    required_cols = {'dataset', 'method', 'best_acc'};
    for k = 1:length(required_cols)
        if ~ismember(required_cols{k}, S.Properties.VariableNames)
            error('Summary file missing required column: %s', required_cols{k});
        end
    end

    datasets = unique(S.dataset, 'stable');
    methods = {'BPTT', 'TESS'};
    M = nan(length(datasets), length(methods));

    for i = 1:length(datasets)
        for j = 1:length(methods)
            idx = strcmp(S.dataset, datasets{i}) & strcmp(S.method, methods{j});
            if any(idx)
                M(i, j) = S.best_acc(find(idx, 1));
            end
        end
    end

    fig = figure('Name', 'BPTT vs TESS Summary', 'Color', 'w');
    bar(M, 'grouped');
    grid on; box on;

    set(gca, ...
        'XTick', 1:length(datasets), ...
        'XTickLabel', datasets, ...
        'FontSize', 12, ...
        'TickLabelInterpreter', 'latex');

    xlabel('Dataset', 'Interpreter', 'latex');
    ylabel('Best Acc@1', 'Interpreter', 'latex');
    title('Comparison of BPTT and TESS across different image recognition tasks', 'Interpreter', 'latex');
    legend({'BPTT', 'TESS'}, 'Location', 'best', 'Interpreter', 'latex');

    png_path = fullfile(save_dir, 'bptt_tess_summary_bar.png');
    fig_path = fullfile(save_dir, 'bptt_tess_summary_bar.fig');

    exportgraphics(fig, png_path, 'Resolution', 300);
    savefig(fig, fig_path);
else
    fprintf('[WARN] Summary file not found: %s\n', summary_file);
end

fprintf('[DONE] BPTT vs TESS figures saved to folder: %s\n', save_dir);