% Reproduce the Python minimum-variance optimization with MATLAB.
% Requires Optimization Toolbox for quadprog.
root = fileparts(fileparts(mfilename('fullpath')));
covariance = readtable(fullfile(root, 'outputs', 'daily_covariance.csv'), 'VariableNamingRule', 'preserve');
tickers = covariance{:, 1};
sigma = table2array(covariance(:, 2:end)) * 252;
n = length(tickers);

H = 2 * sigma;
f = zeros(n, 1);
Aeq = ones(1, n);
beq = 1;
lb = zeros(n, 1);
ub = 0.30 * ones(n, 1);
options = optimoptions('quadprog', 'Display', 'off');
[weights, objective, exitflag] = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], options);

if exitflag <= 0
    error('MATLAB optimization did not converge.');
end

result = table(tickers, weights, 'VariableNames', {'ticker', 'weight'});
writetable(result, fullfile(root, 'outputs', 'matlab_minimum_variance_weights.csv'));
fprintf('Annualized portfolio variance: %.8f\n', objective / 2);
disp(result);
