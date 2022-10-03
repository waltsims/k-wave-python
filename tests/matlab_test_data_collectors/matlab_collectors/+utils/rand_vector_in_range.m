function [vec] = rand_vector_in_range(min_bound, max_bound, num_points)
    vec = (max_bound-min_bound).*rand(num_points, 1) + min_bound;
end
