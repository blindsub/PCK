function [ P] = compute_dist( data,centers,num_clusters)
% Calculate the distance (square) between data and centers
n = size(data, 1);
x = sum(data.*data, 2)';
X = x(ones(num_clusters, 1), :);
y = sum(centers.*centers, 2);
Y = y(:, ones(n, 1));
P = X + Y - 2*centers*data';
P=P'