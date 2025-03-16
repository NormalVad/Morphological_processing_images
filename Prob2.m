% Ayush Goyal
% 7184517074
% ayushgoy@usc.edu
% March 16, 2025

% Problem 2 - Homographic Transformation and Image Stitching
IMG_HEIGHT = 480;
IMG_WIDTH = 640;
CHANNELS = 3;

% Feature matching parameters
feat_thresh = 0.5;
ratio_thresh = 0.75; 
dist_thresh = 2.0;


first_path = 'P1P2/Street_Left.raw';
center_path = 'P1P2/Street_Middle.raw';
third_path = 'P1P2/Street_Right.raw';
total_bytes = IMG_WIDTH * IMG_HEIGHT * CHANNELS;


fid = fopen(first_path, 'rb');
first_raw = fread(fid, total_bytes, 'uint8=>uint8');
fclose(fid);
first_raw = reshape(first_raw, [CHANNELS, IMG_WIDTH, IMG_HEIGHT]);
first_img = permute(first_raw, [3, 2, 1]);

fid = fopen(center_path, 'rb');
center_raw = fread(fid, total_bytes, 'uint8=>uint8');
fclose(fid);
center_raw = reshape(center_raw, [CHANNELS, IMG_WIDTH, IMG_HEIGHT]);
center_img = permute(center_raw, [3, 2, 1]);

fid = fopen(third_path, 'rb');
third_raw = fread(fid, total_bytes, 'uint8=>uint8');
fclose(fid);
third_raw = reshape(third_raw, [CHANNELS, IMG_WIDTH, IMG_HEIGHT]);
third_img = permute(third_raw, [3, 2, 1]);


figure(1); imshow(uint8(first_img)); title('First Image');
figure(2); imshow(uint8(center_img)); title('Center Image');
figure(3); imshow(uint8(third_img)); title('Third Image');

% Convert images to grayscale
first_gray = rgb2gray(uint8(first_img));
center_gray = rgb2gray(uint8(center_img));
third_gray = rgb2gray(uint8(third_img));

% Feature detection and extraction using SURF
first_kp = detectSURFFeatures(first_gray, 'MetricThreshold', 800, 'NumOctaves', 4, 'NumScaleLevels', 6);
first_feat = extractFeatures(first_gray, first_kp, 'Method', 'SURF');

center_kp = detectSURFFeatures(center_gray, 'MetricThreshold', 800, 'NumOctaves', 4, 'NumScaleLevels', 6);
center_feat = extractFeatures(center_gray, center_kp, 'Method', 'SURF');

third_kp = detectSURFFeatures(third_gray, 'MetricThreshold', 800, 'NumOctaves', 4, 'NumScaleLevels', 6);
third_feat = extractFeatures(third_gray, third_kp, 'Method', 'SURF');


fc_matches = matchFeatures(first_feat, center_feat, ...
                                 'Method', 'NearestNeighborRatio', ...
                                 'MatchThreshold', feat_thresh, ...
                                 'MaxRatio', ratio_thresh);

first_pts = SURFPoints();
center_pts1 = SURFPoints();
for i = 1:size(fc_matches, 1)
    src_idx = fc_matches(i, 1);
    dst_idx = fc_matches(i, 2);
    first_pts(i) = first_kp(src_idx);
    center_pts1(i) = center_kp(dst_idx);
end
fprintf('Number of matched points (First-Center): %d\n', size(fc_matches, 1));

ct_matches = matchFeatures(center_feat, third_feat, ...
                                  'Method', 'NearestNeighborRatio', ...
                                  'MatchThreshold', feat_thresh, ...
                                  'MaxRatio', ratio_thresh);

center_pts2 = SURFPoints();
third_pts = SURFPoints();
for i = 1:size(ct_matches, 1)
    src_idx = ct_matches(i, 1);
    dst_idx = ct_matches(i, 2);
    center_pts2(i) = center_kp(src_idx);
    third_pts(i) = third_kp(dst_idx);
end
fprintf('Number of matched points (Center-Third): %d\n', size(ct_matches, 1));

figure(4);
showMatchedFeatures(first_gray, center_gray, first_pts, center_pts1, 'montage');
title('Control Points Between First and Center Images');

figure(5);
showMatchedFeatures(center_gray, third_gray, center_pts2, third_pts, 'montage');
title('Control Points Between Center and Third Images');

% Homography Estimation
fc_idx = select_points(first_pts.Location, center_pts1.Location, first_pts.Metric, center_pts1.Metric);
first_p = zeros(4, 2);
center_p1 = zeros(4, 2);

for i = 1:4
    idx = fc_idx(i);
    first_p(i, 1) = first_pts.Location(idx, 1);
    first_p(i, 2) = first_pts.Location(idx, 2);
    center_p1(i, 1) = center_pts1.Location(idx, 1);
    center_p1(i, 2) = center_pts1.Location(idx, 2);
end

ct_idx = select_points(third_pts.Location, center_pts2.Location, third_pts.Metric, center_pts2.Metric);
center_p2 = zeros(4, 2);
third_p = zeros(4, 2);

for i = 1:4
    idx = ct_idx(i);
    center_p2(i, 1) = center_pts2.Location(idx, 1);
    center_p2(i, 2) = center_pts2.Location(idx, 2);
    third_p(i, 1) = third_pts.Location(idx, 1);
    third_p(i, 2) = third_pts.Location(idx, 2);
end

% Compute homography matrices
[H_first_to_center] = compute_homography(first_p, center_p1);
[H_third_to_center] = compute_homography(third_p, center_p2);

% Control Points
figure(6); 
imshow(first_gray); hold on;
plot(first_p(:,1), first_p(:,2), 'ro', 'MarkerSize', 10);
title('First Image Control Points');

figure(7); imshow(center_gray); hold on;
plot(center_p1(:,1), center_p1(:,2), 'ro', 'MarkerSize', 10);
plot(center_p2(:,1), center_p2(:,2), 'go', 'MarkerSize', 10);
title('Center Image Control Points (Red=vs First, Green=vs Third)');

figure(8);
imshow(third_gray); hold on;
plot(third_p(:,1), third_p(:,2), 'go', 'MarkerSize', 10);
title('Third Image Control Points');

% Panorama Creation
[h1, w1, ~] = size(first_img);
[h2, w2, ~] = size(center_img);
[h3, w3, ~] = size(third_img);

first_corners = [1, 1; w1, 1; w1, h1; 1, h1];
center_corners = [1, 1; w2, 1; w2, h2; 1, h2];
third_corners = [1, 1; w3, 1; w3, h3; 1, h3];

first_corners_transformed = transform_points(first_corners, H_first_to_center);
third_corners_transformed = transform_points(third_corners, H_third_to_center);

all_corners = [first_corners_transformed; center_corners; third_corners_transformed];
min_x_val = all_corners(1, 1);
max_x_val = all_corners(1, 1);
min_y_val = all_corners(1, 2);
max_y_val = all_corners(1, 2);

for i = 1:size(all_corners, 1)
    if all_corners(i, 1) < min_x_val
        min_x_val = all_corners(i, 1);
    end
    if all_corners(i, 1) > max_x_val
        max_x_val = all_corners(i, 1);
    end
    if all_corners(i, 2) < min_y_val
        min_y_val = all_corners(i, 2);
    end
    if all_corners(i, 2) > max_y_val
        max_y_val = all_corners(i, 2);
    end
end

min_x = floor(min_x_val);
max_x = ceil(max_x_val);
min_y = floor(min_y_val);
max_y = ceil(max_y_val);

min_x = min_x - 10;
max_x = max_x + 10;
min_y = min_y - 10;
max_y = max_y + 10;

pano_width = max_x - min_x + 1;
pano_height = max_y - min_y + 1;
disp(['Panorama size: ' num2str(pano_width) ' x ' num2str(pano_height)]);
panorama = zeros(pano_height, pano_width, 3, 'uint8');
weight_map = zeros(pano_height, pano_width);

H_center_to_first = inv(H_first_to_center);
H_center_to_third = inv(H_third_to_center);

% Place center image
for y = 1:h2
    for x = 1:w2
        pano_x = round(x - min_x);
        pano_y = round(y - min_y);
        
        if pano_x >= 1 && pano_x <= pano_width && pano_y >= 1 && pano_y <= pano_height
            for c = 1:3
                panorama(pano_y, pano_x, c) = center_img(y, x, c);
            end
            weight_map(pano_y, pano_x) = 1;
        end
    end
end

% Warp and blend images
% First Image
for y = 1:pano_height
    for x = 1:pano_width
        center_x = x + min_x - 1;
        center_y = y + min_y - 1;
        
        [first_x, first_y] = transform_point(center_x, center_y, H_center_to_first);
        
        if first_x >= 1 && first_x <= w1 && first_y >= 1 && first_y <= h1
            x1 = floor(first_x);
            y1 = floor(first_y);
            x2 = ceil(first_x);
            y2 = ceil(first_y);
            
            if x2 > w1, x2 = w1; end
            if y2 > h1, y2 = h1; end
            
            wx = first_x - x1;
            wy = first_y - y1;
            
            pixel_vals = zeros(1, 3, 'uint8');
            for c = 1:3
                if x1 == x2 || y1 == y2
                    pixel_vals(c) = first_img(y1, x1, c);
                else
                    top_left = double(first_img(y1, x1, c));
                    top_right = double(first_img(y1, x2, c));
                    bottom_left = double(first_img(y2, x1, c));
                    bottom_right = double(first_img(y2, x2, c));
                    
                    interp_val = (1-wx)*(1-wy)*top_left + ...
                               wx*(1-wy)*top_right + ...
                               (1-wx)*wy*bottom_left + ...
                               wx*wy*bottom_right;
                    
                    pixel_vals(c) = uint8(round(interp_val));
                end
            end
            
            if weight_map(y, x) > 0
                for c = 1:3
                    current_val = double(panorama(y, x, c));
                    new_val = double(pixel_vals(c));
                    blended_val = (current_val * weight_map(y, x) + new_val) / (weight_map(y, x) + 1);
                    panorama(y, x, c) = uint8(round(blended_val));
                end
            else
                for c = 1:3
                    panorama(y, x, c) = pixel_vals(c);
                end
            end
            weight_map(y, x) = weight_map(y, x) + 1;
        end
    end
end

% Third Image
for y = 1:pano_height
    for x = 1:pano_width
        center_x = x + min_x - 1;
        center_y = y + min_y - 1;
        
        [third_x, third_y] = transform_point(center_x, center_y, H_center_to_third);
        
        if third_x >= 1 && third_x <= w3 && third_y >= 1 && third_y <= h3
            x1 = floor(third_x);
            y1 = floor(third_y);
            x2 = ceil(third_x);
            y2 = ceil(third_y);
            
            if x2 > w3, x2 = w3; end
            if y2 > h3, y2 = h3; end
            
            wx = third_x - x1;
            wy = third_y - y1;
            
            pixel_vals = zeros(1, 3, 'uint8');
            for c = 1:3
                if x1 == x2 || y1 == y2
                    pixel_vals(c) = third_img(y1, x1, c);
                else
                    top_left = double(third_img(y1, x1, c));
                    top_right = double(third_img(y1, x2, c));
                    bottom_left = double(third_img(y2, x1, c));
                    bottom_right = double(third_img(y2, x2, c));
                    
                    interp_val = (1-wx)*(1-wy)*top_left + ...
                               wx*(1-wy)*top_right + ...
                               (1-wx)*wy*bottom_left + ...
                               wx*wy*bottom_right;
                    
                    pixel_vals(c) = uint8(round(interp_val));
                end
            end
            
            if weight_map(y, x) > 0
                for c = 1:3
                    current_val = double(panorama(y, x, c));
                    new_val = double(pixel_vals(c));
                    blended_val = (current_val * weight_map(y, x) + new_val) / (weight_map(y, x) + 1);
                    panorama(y, x, c) = uint8(round(blended_val));
                end
            else
                for c = 1:3
                    panorama(y, x, c) = pixel_vals(c);
                end
            end
            weight_map(y, x) = weight_map(y, x) + 1;
        end
    end
end


figure(9);
imshow(panorama);
title('Panorama');

% Transform Points
function transformed_points = transform_points(points, H)
    n = size(points, 1);
    transformed_points = zeros(n, 2);    
    for i = 1:n
        [transformed_points(i,1), transformed_points(i,2)] = transform_point(points(i,1), points(i,2), H);
    end
end

function [x_new, y_new] = transform_point(x, y, H)
    den = H(3,1)*x + H(3,2)*y + H(3,3);
    if abs(den) < 1e-10
        den = 1e-10 * sign(den);
    end
    x_new = (H(1,1)*x + H(1,2)*y + H(1,3)) / den;
    y_new = (H(2,1)*x + H(2,2)*y + H(2,3)) / den;
end

% Select Points
function idx = select_points(src_points, dst_points, src_metrics, dst_metrics)
    num_pts = size(src_points, 1);
    min_src_metric = src_metrics(1);
    max_src_metric = src_metrics(1);
    for i = 2:length(src_metrics)
        if src_metrics(i) < min_src_metric
            min_src_metric = src_metrics(i);
        end
        if src_metrics(i) > max_src_metric
            max_src_metric = src_metrics(i);
        end
    end
    min_dst_metric = dst_metrics(1);
    max_dst_metric = dst_metrics(1);
    for i = 2:length(dst_metrics)
        if dst_metrics(i) < min_dst_metric
            min_dst_metric = dst_metrics(i);
        end
        if dst_metrics(i) > max_dst_metric
            max_dst_metric = dst_metrics(i);
        end
    end
    src_metrics_norm = zeros(size(src_metrics));
    for i = 1:length(src_metrics)
        src_metrics_norm(i) = (src_metrics(i) - min_src_metric) / (max_src_metric - min_src_metric + eps);
    end
    
    dst_metrics_norm = zeros(size(dst_metrics));
    for i = 1:length(dst_metrics)
        dst_metrics_norm(i) = (dst_metrics(i) - min_dst_metric) / (max_dst_metric - min_dst_metric + eps);
    end
    combined_metrics = zeros(size(src_metrics_norm));
    for i = 1:length(src_metrics_norm)
        combined_metrics(i) = (src_metrics_norm(i) + dst_metrics_norm(i)) / 2;
    end
    ctr_x = 0;
    ctr_y = 0;
    for i = 1:num_pts
        ctr_x = ctr_x + src_points(i, 1);
        ctr_y = ctr_y + src_points(i, 2);
    end
    ctr_x = ctr_x / num_pts;
    ctr_y = ctr_y / num_pts;
    
    quadrant = zeros(num_pts, 1);
    for i = 1:num_pts
        x = src_points(i, 1);
        y = src_points(i, 2);
        
        if x < ctr_x && y < ctr_y
            quadrant(i) = 1;
        elseif x >= ctr_x && y < ctr_y
            quadrant(i) = 2;
        elseif x < ctr_x && y >= ctr_y
            quadrant(i) = 3;
        else
            quadrant(i) = 4;
        end
    end
    distances = zeros(num_pts, 1);
    for i = 1:num_pts
        dx = src_points(i, 1) - ctr_x;
        dy = src_points(i, 2) - ctr_y;
        distances(i) = sqrt(dx*dx + dy*dy);
    end
    max_dist = distances(1);
    for i = 2:num_pts
        if distances(i) > max_dist
            max_dist = distances(i);
        end
    end
    max_dist = max_dist + eps;
    dist_norm = zeros(size(distances));
    for i = 1:num_pts
        dist_norm(i) = distances(i) / max_dist;
    end
    
    alpha = 0.6;
    beta = 0.4;
    scores = zeros(size(combined_metrics));
    for i = 1:num_pts
        scores(i) = alpha * combined_metrics(i) + beta * dist_norm(i);
    end
    
    if size(dst_points, 1) == size(src_points, 1)
        corr_dists = zeros(num_pts, 1);
        for i = 1:num_pts
            dx = dst_points(i, 1) - src_points(i, 1);
            dy = dst_points(i, 2) - src_points(i, 2);
            corr_dists(i) = sqrt(dx*dx + dy*dy);
        end
        mean_dist = 0;
        for i = 1:num_pts
            mean_dist = mean_dist + corr_dists(i);
        end
        mean_dist = mean_dist / num_pts;
        sum_squared_diff = 0;
        for i = 1:num_pts
            diff = corr_dists(i) - mean_dist;
            sum_squared_diff = sum_squared_diff + diff * diff;
        end
        std_dist = sqrt(sum_squared_diff / num_pts);
        consistency_score = zeros(size(corr_dists));
        for i = 1:num_pts
            temp = 1 - abs(corr_dists(i) - mean_dist) / (3 * std_dist + eps);
            if temp < 0
                consistency_score(i) = 0;
            elseif temp > 1
                consistency_score(i) = 1;
            else
                consistency_score(i) = temp;
            end
        end 
        gamma = 0.3;
        alpha = alpha * (1 - gamma);
        beta = beta * (1 - gamma);
        for i = 1:num_pts
            scores(i) = alpha * combined_metrics(i) + beta * dist_norm(i) + gamma * consistency_score(i);
        end
    end
    selected_idx = zeros(4, 1);
    for q = 1:4
        q_points = find(quadrant == q);
        if ~isempty(q_points)
            best_score = -1;
            best_idx = -1;
            for i = 1:length(q_points)
                if scores(q_points(i)) > best_score
                    best_score = scores(q_points(i));
                    best_idx = i;
                end
            end
            selected_idx(q) = q_points(best_idx);
        end
    end
    empty_quads = selected_idx == 0;
    if any(empty_quads)
        remaining_pts = [];
        for i = 1:num_pts
            is_remaining = true;
            for j = 1:4
                if ~empty_quads(j) && selected_idx(j) == i
                    is_remaining = false;
                    break;
                end
            end
            if is_remaining
                remaining_pts = [remaining_pts, i];
            end
        end
        sorted_idx = zeros(size(remaining_pts));
        sorted_scores = zeros(size(remaining_pts));
        for i = 1:length(remaining_pts)
            sorted_scores(i) = scores(remaining_pts(i));
            sorted_idx(i) = i;
        end
        for i = 1:length(sorted_idx)-1
            for j = 1:length(sorted_idx)-i
                if sorted_scores(j) < sorted_scores(j+1)
                    temp_score = sorted_scores(j);
                    sorted_scores(j) = sorted_scores(j+1);
                    sorted_scores(j+1) = temp_score;         
                    temp_idx = sorted_idx(j);
                    sorted_idx(j) = sorted_idx(j+1);
                    sorted_idx(j+1) = temp_idx;
                end
            end
        end
        j = 1;
        for i = 1:4
            if empty_quads(i) && j <= length(sorted_idx)
                selected_idx(i) = remaining_pts(sorted_idx(j));
                j = j + 1;
            elseif empty_quads(i)
                available = [];
                for k = 1:num_pts
                    is_available = true;
                    for m = 1:4
                        if selected_idx(m) == k
                            is_available = false;
                            break;
                        end
                    end
                    if is_available
                        available = [available, k];
                    end
                end
                if ~isempty(available)
                    selected_idx(i) = available(1);
                end
            end
        end
    end
    
    if length(unique(selected_idx)) < 4
        unique_idx = [];
        for i = 1:length(selected_idx)
            is_unique = true;
            for j = 1:length(unique_idx)
                if selected_idx(i) == unique_idx(j)
                    is_unique = false;
                    break;
                end
            end
            if is_unique
                unique_idx = [unique_idx, selected_idx(i)];
            end
        end
        counts = zeros(size(unique_idx));
        for i = 1:length(unique_idx)
            for j = 1:length(selected_idx)
                if selected_idx(j) == unique_idx(i)
                    counts(i) = counts(i) + 1;
                end
            end
        end
        duplicates = [];
        for i = 1:length(unique_idx)
            if counts(i) > 1
                duplicates = [duplicates, unique_idx(i)];
            end
        end
        for dup = duplicates
            dup_positions = find(selected_idx == dup);
            for i = 2:length(dup_positions)
                pos = dup_positions(i);
                unused = [];
                for j = 1:num_pts
                    is_unused = true;
                    for k = 1:length(selected_idx)
                        if selected_idx(k) == j
                            is_unused = false;
                            break;
                        end
                    end
                    if is_unused
                        unused = [unused, j];
                    end
                end
                
                if ~isempty(unused)
                    best_score = -1;
                    best_idx = -1;
                    for j = 1:length(unused)
                        if scores(unused(j)) > best_score
                            best_score = scores(unused(j));
                            best_idx = j;
                        end
                    end
                    selected_idx(pos) = unused(best_idx);
                end
            end
        end
    end
    idx = selected_idx;
end

function [H] = compute_homography(src_pts, dst_pts)
    [src_pts_norm, T_src] = normalize_points(src_pts);
    [dst_pts_norm, T_dst] = normalize_points(dst_pts);
    A = zeros(8, 9);
    for i = 1:4
        x = src_pts_norm(i, 1);
        y = src_pts_norm(i, 2);
        u = dst_pts_norm(i, 1);
        v = dst_pts_norm(i, 2);
        A(2*i-1, :) = [0, 0, 0, -x, -y, -1, v*x, v*y, v];
        A(2*i, :) = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u];
    end
    
    [~, ~, V] = svd(A);
    h = V(:, 9);
    H_norm = reshape(h, 3, 3)';
    H = inv(T_dst) * H_norm * T_src;
    H = H / H(3,3);
end

function [norm_pts, T] = normalize_points(pts)
    centroid_x = 0;
    centroid_y = 0;
    for i = 1:size(pts, 1)
        centroid_x = centroid_x + pts(i, 1);
        centroid_y = centroid_y + pts(i, 2);
    end
    centroid_x = centroid_x / size(pts, 1);
    centroid_y = centroid_y / size(pts, 1);
    cent_pts = zeros(size(pts));
    for i = 1:size(pts, 1)
        cent_pts(i, 1) = pts(i, 1) - centroid_x;
        cent_pts(i, 2) = pts(i, 2) - centroid_y;
    end
    sum_squares = 0;
    for i = 1:size(cent_pts, 1)
        sum_squares = sum_squares + cent_pts(i, 1)^2 + cent_pts(i, 2)^2;
    end
    scale = sqrt(sum_squares / (2 * size(pts, 1))) + eps;
    scale_factor = sqrt(2) / scale;
    T = [scale_factor, 0, -scale_factor*centroid_x;
         0, scale_factor, -scale_factor*centroid_y;
         0, 0, 1];
    
    norm_pts = zeros(size(pts));
    for i = 1:size(pts, 1)
        p = zeros(3, 1);
        p(1) = pts(i, 1);
        p(2) = pts(i, 2);
        p(3) = 1;
        p_norm = zeros(3, 1);
        for j = 1:3
            for k = 1:3
                p_norm(j) = p_norm(j) + T(j, k) * p(k);
            end
        end
        norm_pts(i, 1) = p_norm(1) / p_norm(3);
        norm_pts(i, 2) = p_norm(2) / p_norm(3);
    end
end
