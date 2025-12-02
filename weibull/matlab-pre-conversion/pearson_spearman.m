clear all

% framework
% should import dataset-> ...
% det pearson and spearman correlation

% pre
% import dataset
rp_data= readtable('rodpump_dataset.csv');

% DROP: roduid, UWI, NODEID, IDWELL, tbguid, ...
% lifetime_start, lifetime_end, lifetime_duration_days ...
% IDRECJOBPULL, FAILURETYPE, bha_configuration, ...
% wellbore_category, manual_scale, packer_vs_tac, ...
% rod_sinker_type, rod_has_guides, rod_make, rod_apigrade, ...
% DESANDDEGAS_TYP, gasanchor_od

rp_data(:, {'roduid', 'UWI', 'NODEID', 'IDWELL', 'tbguid', ...
        'lifetime_start', 'lifetime_end', 'lifetime_duration_days', ...
        'IDRECJOBPULL', 'FAILURETYPE', 'bha_configuration', ...
        'wellbore_category', 'manual_scale', 'packer_vs_tac', ...
        'rod_sinker_type', 'rod_has_guides', 'rod_make', 'rod_apigrade', ...
        'DESANDDEGAS_TYP', 'gasanchor_od' ... % also drop chemgroups
        'chemgroup1_any', 'chemgroup1_all', 'chemgroup2_any', 'chemgroup2_all', ...
        'chemgroup3_any', 'chemgroup3_all'
        })= [];

isNumerical= varfun(@isnumeric, rp_data, 'OutputFormat', 'uniform');
numericalColumns= rp_data(:, isNumerical);
numericalData= numericalColumns{:, :}; % table-> array
numericalVarNames= numericalColumns.Properties.VariableNames; % Added this missing line

% need to det pearson and spearman correlation
% ignoring certain columns
pearsonCorr= corr(numericalData, 'Type', 'Pearson','Rows','pairwise');
spearmanCorr= corr(numericalData, 'Type', 'Spearman','Rows','pairwise');

% generate heatmap(s)
figure(1)
h1= heatmap(numericalVarNames, numericalVarNames, pearsonCorr, ...
    'Title', 'Pearson Correlation Heatmap');
colormap(gca,'jet');
h1.FontSize= 8;

figure(2)
h2= heatmap(numericalVarNames, numericalVarNames, spearmanCorr, ...
    'Title', 'Spearman Correlation Heatmap');
colormap(gca,'jet');
h2.FontSize= 8;

% generate list of correlations for pearson

% to avoid duplicates
[n, ~]= size(pearsonCorr);
[upperRow, upperCol]= find(triu(ones(n), 1));
pearsonValues= pearsonCorr(sub2ind([n n], upperRow, upperCol));
pearsonPairs= cell(length(upperRow), 1);

for i=1:length(upperRow)
    pearsonPairs{i}= sprintf('%s - %s', numericalVarNames{upperRow(i)}, numericalVarNames{upperCol(i)});
end

% by abs correlation strength
[~, sortIdx]= sort(abs(pearsonValues), 'descend');
pearsonValues= pearsonValues(sortIdx);
pearsonPairs= pearsonPairs(sortIdx);

% print as text list
fprintf('\n=== PEARSON CORRELATIONS (sorted by strength) ===\n');
fprintf('Correlation | Variable Pair\n');
fprintf('------------|---------------\n');

for i=1:length(pearsonPairs)
    fprintf('%11.3f | %s\n', pearsonValues(i), pearsonPairs{i});
end

% generate list of correlations for spearman

% to avoid duplicates
spearmanValues= spearmanCorr(sub2ind([n n], upperRow, upperCol));
spearmanPairs= cell(length(upperRow), 1);

for i=1:length(upperRow)
    spearmanPairs{i}= sprintf('%s - %s', numericalVarNames{upperRow(i)}, numericalVarNames{upperCol(i)});
end

% by abs correlation strength
[~, sortIdx]= sort(abs(spearmanValues), 'descend');
spearmanValues= spearmanValues(sortIdx);
spearmanPairs= spearmanPairs(sortIdx);

% print as text list
fprintf('\n=== SPEARMAN CORRELATIONS (sorted by strength) ===\n');
fprintf('Correlation | Variable Pair\n');
fprintf('------------|---------------\n');

for i=1:length(spearmanPairs)
    fprintf('%11.3f | %s\n', spearmanValues(i), spearmanPairs{i});
end