
try
    addpath([pwd '/build']);

    %% test with simplified inputs
    x = [3.0, 5.0];

    tic
    % y = net_forward(x);
    y = net_forward_mex(x);
    toc

    % print x and y
    fprintf('x -> [ %s ]\n', num2str(x, '%+.4f '));
    fprintf('y -> [ %s ]\n', num2str(y, '%+.4f '));


    % test if matlab linear algebra still works
    A = rand(1000);
    B = A * A';  % symmetric positive-definite
    chol(B);     % Cholesky decomposition (calls LAPACK via MKL)
    A = rand(5000);
    B = rand(5000);
    C = A * B;  % should call dgemm from MKL


catch ME
    fprintf('Error: %s\n', ME.message);
end