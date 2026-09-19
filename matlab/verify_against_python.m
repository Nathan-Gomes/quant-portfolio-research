% VERIFY_AGAINST_PYTHON  Independent MATLAB check of the Python optimizer.
%
% The point of a cross-check is to disagree when something is wrong. The earlier
% version of this script recomputed the weights and wrote them to a file, which
% left a human to notice a discrepancy; this one compares them and fails loudly.
%
% It reads the covariance and the weights the Python pipeline published, solves
% the same two problems with the independent implementations in this folder, and
% reports the largest disagreement against a stated tolerance.
%
% Runs in MATLAB and in GNU Octave, with no toolbox required.
%
%   octave --no-gui matlab/verify_against_python.m
%   matlab -batch "run('matlab/verify_against_python.m')"

root = fileparts(fileparts(mfilename('fullpath')));
if isempty(root)
    root = fileparts(pwd);        % Octave running the file directly
end
addpath(fullfile(root, 'matlab'));

maximum_weight = 0.30;            % must match config/research.yaml
tolerance = 1e-4;                 % weights; a flat optimum moves more than the variance does
variance_tolerance = 1e-8;

% The covariance the published weights were solved from — not the full-sample
% matrix, which describes a different problem and would make any disagreement
% here meaningless.
covariance_file = fullfile(root, 'outputs', 'optimizer_input_covariance.csv');
weights_file = fullfile(root, 'outputs', 'minimum_variance_weights.csv');
if ~exist(covariance_file, 'file') || ~exist(weights_file, 'file')
    error('Run the Python pipeline first: python run_pipeline.py');
end

[tickers, covariance_values] = read_labelled_csv(covariance_file);
sigma = covariance_values * 252;

[~, python_values] = read_labelled_csv(weights_file);
python_weights = python_values(:, end);

matlab_weights = minimum_variance_portfolio(sigma, maximum_weight);

weight_gap = max(abs(matlab_weights - python_weights));
python_variance = python_weights' * sigma * python_weights;
matlab_variance = matlab_weights' * sigma * matlab_weights;
variance_gap = abs(python_variance - matlab_variance);

fprintf('\nIndependent MATLAB cross-check\n');
fprintf('------------------------------------------------------------\n');
fprintf('%-10s %14s %14s %12s\n', 'ticker', 'python', 'matlab', 'difference');
for i = 1:numel(tickers)
    name = tickers{i};
    fprintf('%-10s %14.6f %14.6f %12.2e\n', name, python_weights(i), ...
            matlab_weights(i), abs(python_weights(i) - matlab_weights(i)));
end
fprintf('------------------------------------------------------------\n');
fprintf('largest weight difference   %.3e   (tolerance %.0e)\n', weight_gap, tolerance);
fprintf('portfolio variance, python  %.10f\n', python_variance);
fprintf('portfolio variance, matlab  %.10f\n', matlab_variance);
fprintf('variance difference         %.3e   (tolerance %.0e)\n', variance_gap, variance_tolerance);

% Constraints, checked on the MATLAB answer rather than assumed from the Python one.
assert(abs(sum(matlab_weights) - 1) < 1e-10, 'MATLAB weights are not fully invested.');
assert(all(matlab_weights > -1e-12), 'MATLAB weights include a short position.');
assert(max(matlab_weights) <= maximum_weight + 1e-9, 'MATLAB weights breach the cap.');

passed = (weight_gap < tolerance) && (variance_gap < variance_tolerance);
if passed
    fprintf('\nRESULT: PASS — the two implementations agree.\n\n');
else
    fprintf('\nRESULT: FAIL — the implementations disagree beyond tolerance.\n\n');
end

write_csv(fullfile(root, 'outputs', 'matlab_cross_check.csv'), ...
    {'problem', 'max_weight_difference', 'variance_difference', 'passed'}, ...
    {'minimum_variance'}, [weight_gap, variance_gap, double(passed)]);

write_csv(fullfile(root, 'outputs', 'matlab_minimum_variance_weights.csv'), ...
    {'ticker', 'python_weight', 'matlab_weight'}, tickers, ...
    [python_weights, matlab_weights]);

if ~passed
    error('MATLAB cross-check failed.');
end
