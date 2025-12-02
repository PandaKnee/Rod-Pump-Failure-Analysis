function fit_to_weibull()
% fit_to_weibull - fits rod pump data to weibull distribution after cleaning
    
    % import dataset
    rp_data= readtable('rodpump_dataset.csv');
    
    fprintf('original data size: %d rows x %d columns\n', size(rp_data));
    
    % drop columns
    rp_data(:, {'roduid', 'UWI', 'NODEID', 'IDWELL', 'tbguid', ...
        'lifetime_start', 'lifetime_end', 'lifetime_duration_days', ...
        'IDRECJOBPULL', 'FAILURETYPE', 'bha_configuration', ...
        'wellbore_category', 'manual_scale', 'packer_vs_tac', ...
        'rod_sinker_type', 'rod_has_guides', 'rod_make', 'rod_apigrade', ...
        'DESANDDEGAS_TYP', 'gasanchor_od' ... % also drop chemgroups
        'chemgroup1_any', 'chemgroup1_all', 'chemgroup2_any', 'chemgroup2_all', ...
        'chemgroup3_any', 'chemgroup3_all'
        })= [];
    
    fprintf('after first drop: %d rows x %d columns\n', size(rp_data));
    
    % drop 2: additional specified columns
    additional_drops= {'Shallow_max_sideload', 'Avg_liquid_volume', ...
                       'Max_unguided_sideload', 'Avg_pressure_casing', ...
                       'dls_high_in_hole' ... % reasoning
                       'REPORTTO', 'FAILSTART', 'H2S_CONCENTRATION', 'PrimarySetpoint', ...
                       'SecondarySetpoint'
                       };
    
    % check which additional columns exist
    existing_additional= ismember(additional_drops, rp_data.Properties.VariableNames);
    rp_data(:, additional_drops(existing_additional))= [];
    
    fprintf('after additional drops: %d rows x %d columns\n', size(rp_data));
    fprintf('remaining variables:\n');
    disp(rp_data.Properties.VariableNames');
    
    % extract numerical data
    isNumerical= varfun(@isnumeric, rp_data, 'OutputFormat', 'uniform');
    numericalColumns= rp_data(:, isNumerical);
    numericalData= numericalColumns{:, :}; % table-> array
    numericalVarNames= numericalColumns.Properties.VariableNames;
    
    fprintf('numerical variables for weibull fitting (%d):\n', length(numericalVarNames));
    disp(numericalVarNames');
    
    % fit weibull distribution to each numerical variable
    fit_weibull_to_variables(numericalData, numericalVarNames);
    
end

function fit_weibull_to_variables(data, varNames)
% fit_weibull_to_variables - fits weibull distribution to each variable
    
    [nSamples, nVars]= size(data);
    
    % determine subplot layout
    nPlots= nVars;
    nCols= 3;
    nRows= ceil(nPlots / nCols);
    
    figure('Position', [100, 100, 1400, 900]);
    
    for i= 1:nVars
        currentData= data(:, i);
        
        % remove missing values and ensure positive values for weibull
        currentData= currentData(~isnan(currentData) & ~isinf(currentData) & currentData > 0);
        
        if isempty(currentData)
            fprintf('skipping variable %s: no valid positive data\n', varNames{i});
            continue;
        end
        
        if length(currentData) < 5
            fprintf('skipping variable %s: insufficient positive data points (%d)\n', varNames{i}, length(currentData));
            continue;
        end
        
        % fit weibull distribution
        try
            paramEsts= wblfit(currentData);
            a= paramEsts(1); % scale parameter
            b= paramEsts(2); % shape parameter
            
            % create subplot
            subplot(nRows, nCols, i);
            
            % plot histogram with fitted weibull pdf
            histogram(currentData, 30, 'Normalization', 'pdf', ...
                     'FaceColor', [0.7 0.7 0.9], 'EdgeColor', 'k');
            hold on;
            
            % generate pdf for weibull distribution
            x= linspace(min(currentData), max(currentData), 1000);
            y= wblpdf(x, a, b);
            plot(x, y, 'r-', 'LineWidth', 2);
            
            title(sprintf('%s\n(\\alpha=%.2f, \\beta=%.2f, n=%d)', ...
                          varNames{i}, a, b, length(currentData)), 'Interpreter', 'tex', ...
                          'FontSize', 8);
            xlabel('value');
            ylabel('probability density');
            legend('data', 'weibull fit', 'Location', 'best');
            grid on;
            
            % store results
            weibull_params(i).variable= varNames{i};
            weibull_params(i).scale= a;
            weibull_params(i).shape= b;
            weibull_params(i).n= length(currentData);
            weibull_params(i).original_n= sum(~isnan(data(:, i)) & ~isinf(data(:, i)));
            
        catch ME
            fprintf('error fitting weibull to %s: %s\n', varNames{i}, ME.message);
            
            % create empty subplot to maintain layout
            subplot(nRows, nCols, i);
            text(0.5, 0.5, sprintf('cannot fit\n%s', varNames{i}), ...
                 'HorizontalAlignment', 'center', 'Units', 'normalized');
            title(varNames{i}, 'FontSize', 8);
            set(gca, 'XTick', [], 'YTick', []);
            continue;
        end
    end
    
    % add overall title
    sgtitle('weibull distribution fits for rod pump variables (positive values only)', 'FontSize', 14, 'FontWeight', 'bold');
    
    % display parameter summary for successful fits
    successful_fits= [];
    for i= 1:length(weibull_params)
        if ~isempty(weibull_params(i).variable) && weibull_params(i).n >= 5
            successful_fits= [successful_fits, weibull_params(i)];
        end
    end
    
    if ~isempty(successful_fits)
        display_weibull_summary(successful_fits);
    else
        fprintf('\nno successful weibull fits - all variables contain zeros/negatives\n');
    end
end

function display_weibull_summary(params)
% display_weibull_summary - display summary of weibull parameters
    
    fprintf('\n=== weibull distribution fitting summary ===\n');
    fprintf('%-25s %-10s %-10s %-8s %-8s\n', 'variable', 'scale(α)', 'shape(β)', 'n_used', 'n_total');
    fprintf('%-25s %-10s %-10s %-8s %-8s\n', '--------', '---------', '--------', '------', '-------');
    
    for i= 1:length(params)
        fprintf('%-25s %-10.4f %-10.4f %-8d %-8d\n', ...
                params(i).variable, params(i).scale, ...
                params(i).shape, params(i).n, params(i).original_n);
    end
    
    % interpretation guide
    fprintf('\n=== weibull shape parameter interpretation ===\n');
    fprintf('β < 1: decreasing failure rate (early failures)\n');
    fprintf('β = 1: constant failure rate (exponential distribution)\n');
    fprintf('β > 1: increasing failure rate (wear-out failures)\n');
    
    % show variables that couldn't be fit
    fprintf('\n=== variables with zeros/negatives (cannot fit weibull) ===\n');
    for i= 1:length(params)
        if params(i).n < params(i).original_n
            fprintf('%s: used %d/%d observations (excluded zeros/negatives)\n', ...
                    params(i).variable, params(i).n, params(i).original_n);
        end
    end
end