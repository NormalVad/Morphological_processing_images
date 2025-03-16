% Ayush Goyal
% 7184517074
% ayushgoy@usc.edu
% March 16, 2025

% Problem 1 - Geometric Image Modification
function image_warping()
    W = 800;
    H = 800;
    C = 3;
    numBytes = W * H * C;

    % Part 1a - Panda
    fid = fopen('P1P2/Panda.raw', 'rb');
    I_panda = fread(fid, numBytes, 'uint8=>uint8');
    fclose(fid);

    I_panda = reshape(I_panda, [C, W, H]);
    I_panda = permute(I_panda, [3, 2, 1]);

    warped_panda = warp_transform(I_panda);

    fid = fopen('Panda_warped.raw', 'wb');
    warped_data = permute(warped_panda, [3,2,1]);
    warped_data = reshape(warped_data, [], 1);
    fwrite(fid, warped_data, 'uint8');
    fclose(fid);

     % Part 1b - Recover Panda
    recovered_panda = reverse_warp_transform(warped_panda);
    
    fid = fopen('Panda_recovered.raw', 'wb');
    recovered_data = permute(recovered_panda, [3,2,1]);
    recovered_data = reshape(recovered_data, [], 1);
    fwrite(fid, recovered_data, 'uint8');
    fclose(fid);

    figure(1);
    subplot(1,3,1); imshow(uint8(I_panda)); title('Original Panda');
    subplot(1,3,2); imshow(uint8(warped_panda)); title('Warped Panda');
    subplot(1,3,3); imshow(uint8(recovered_panda)); title('Recovered Panda');
    
    % Part 1a - Cat
    fid = fopen('P1P2/Cat.raw', 'rb');
    I_cat = fread(fid, numBytes, 'uint8=>uint8');
    fclose(fid);
    
    I_cat = reshape(I_cat, [C, W, H]);
    I_cat = permute(I_cat, [3, 2, 1]);
    
    warped_cat = warp_transform(I_cat);
    
    fid = fopen('Cat_warped.raw', 'wb');
    warped_data = permute(warped_cat, [3,2,1]);
    warped_data = reshape(warped_data, [], 1);
    fwrite(fid, warped_data, 'uint8');
    fclose(fid);
    
    % Part 1b - Recover Cat
    recovered_cat = reverse_warp_transform(warped_cat);
    
    fid = fopen('Cat_recovered.raw', 'wb');
    recovered_data = permute(recovered_cat, [3,2,1]);
    recovered_data = reshape(recovered_data, [], 1);
    fwrite(fid, recovered_data, 'uint8');
    fclose(fid);

    figure(2);
    subplot(1,3,1); imshow(uint8(I_cat)); title('Original Cat');
    subplot(1,3,2); imshow(uint8(warped_cat)); title('Warped Cat');
    subplot(1,3,3); imshow(uint8(recovered_cat)); title('Recovered Cat');

end

function warped = warp_transform(I)
    [H, W, C] = size(I);
    % Initialize output image (black)
    warped = zeros(H, W, C, 'uint8');
    % red arc thickness
    arc_thickness = 128;
    left_boundary = zeros(H, 1);
    right_boundary = zeros(H, 1);
    bottom_boundary = zeros(W, 1);
    % left boundary - sine function at H/2 it will have thickness = 128
    for i = 1:H
        left_boundary(i) = arc_thickness * sin(pi * (i-1)/(H-1));
    end
    % bottom boundary - sine function at W/2 it will have thickness = 128
    for i = 1:W
        bottom_boundary(i) = H - arc_thickness * sin(pi * (i-1)/(W-1));
    end
    % green arc
    circle_center_x = 1;
    circle_center_y = H;
    circle_radius = W; % image width
    for i = 1:H
        dy = i - circle_center_y;
        if abs(dy) < circle_radius
            right_boundary(i) = circle_center_x + sqrt(circle_radius^2 - dy^2);
        else
            right_boundary(i) = W;
        end
        right_boundary(i) = min(right_boundary(i), W);
    end
    
    for y = 1:H
        for x = 1:W
            % Left boundary - red concave arc
            left_x = left_boundary(y);
            % Bottom boundary - red concave arc
            bottom_y = bottom_boundary(x);
            % Right boundary based on green arc
            right_x = right_boundary(y);
            
            % area outside warped region is black
            dist_from_center = sqrt((x - circle_center_x)^2 + (y - circle_center_y)^2);
            if x < left_x || y > bottom_y || dist_from_center > circle_radius
                continue;
            end
            
            % Diagonal top-left to bottom-right - unchanged
            if abs(x - y) < 1
                for c = 1:C
                    warped(y, x, c) = I(y, x, c);
                end
                continue;
            end
            
            % Diagonal from bottom-left to center - unchanged
            if y > H/2 && abs(x + y - (H+1)) < 1
                for c = 1:C
                    warped(y, x, c) = I(y, x, c);
                end
                continue;
            end
            
            if right_x > left_x
                norm_x = (x - left_x) / (right_x - left_x);
            else
                norm_x = 0.5;
            end
            
            top_y = 0;
            if bottom_y > top_y
                norm_y = (y - top_y) / (bottom_y - top_y);
            else
                norm_y = 0.5;
            end
            
            src_x = norm_x * W;
            src_y = norm_y * H;
            
            dist_to_main_diag = abs(x - y);
            dist_to_other_diag = abs(x + y - (H+1));
            
            % Gradual blending
            diag_blend_distance = min(H,W) * 0.04;
            main_diag_blend = 0;
            if (dist_to_main_diag < diag_blend_distance)
                main_diag_blend = 1 - dist_to_main_diag / diag_blend_distance;
            end

            other_diag_blend = 0;
            if (dist_to_other_diag < diag_blend_distance)
                other_diag_blend = 1 - dist_to_other_diag / diag_blend_distance;
            end

            % Blend with the diagonal positions
            if dist_to_main_diag < diag_blend_distance
                t = 1 - dist_to_main_diag / diag_blend_distance;
                main_diag_blend = smoothstep(t);
            else
                main_diag_blend = 0;
            end

            if dist_to_other_diag < diag_blend_distance
                t = 1 - dist_to_other_diag / diag_blend_distance;
                other_diag_blend = smoothstep(t);
            else
                other_diag_blend = 0;
            end
            
            % Bilinear interpolation
            for c = 1:C
                warped(y, x, c) = sample_bilinear(I, src_x, src_y, W, H, c);
            end
        end
    end
end

function recovered_img = reverse_warp_transform(warped)
    [H, W, C] = size(warped);
    recovered_img = zeros(H, W, C, 'uint8');

    arc_thickness = 128;
    circle_center_x = 1;
    circle_center_y = H;
    circle_radius = W;

    left_boundary = zeros(H, 1);
    right_boundary = zeros(H, 1);
    bottom_boundary = zeros(W, 1);

    for i = 1:H
        left_boundary(i) = arc_thickness * sin(pi * (i-1)/(H-1));
    end
    for i = 1:W
        bottom_boundary(i) = H - arc_thickness * sin(pi * (i-1)/(W-1));
    end
    for i = 1:H
        dy = i - circle_center_y;
        if abs(dy) < circle_radius
            right_boundary(i) = circle_center_x + sqrt(circle_radius^2 - dy^2);
        else
            right_boundary(i) = W;
        end
        right_boundary(i) = min(right_boundary(i), W);
    end
    for y = 1:H
        for x = 1:W
            if abs(x - y) < 1 || (y > H/2 && abs(x + y - (H+1)) < 1)
                for c = 1:C
                    recovered_img(y, x, c) = warped(y, x, c);
                end
                continue;
            end
            
            norm_x = x / W;
            norm_y = y / H;
            
            left_x = left_boundary(y);
            bottom_y = bottom_boundary(x);
            right_x = right_boundary(y);
            
            warped_x = left_x + norm_x * (right_x - left_x);
            warped_y = norm_y * bottom_y;
            
            dist_from_center = sqrt((warped_x - circle_center_x)^2 + (warped_y - circle_center_y)^2);
            if warped_x < left_x || warped_y > bottom_y || dist_from_center > circle_radius
                continue; 
            end
            
            dist_to_main_diag = abs(warped_x - warped_y);
            dist_to_other_diag = abs(warped_x + warped_y - (H+1));
            
            diag_blend_distance = min(H, W) * 0.04;
            main_diag_blend = 0;
            if dist_to_main_diag < diag_blend_distance
                t = 1 - dist_to_main_diag / diag_blend_distance;
                main_diag_blend = smoothstep(t);
            end
            other_diag_blend = 0;
            if dist_to_other_diag < diag_blend_distance
                t = 1 - dist_to_other_diag / diag_blend_distance;
                other_diag_blend = smoothstep(t);
            end
            
            for c = 1:C
                recovered_img(y, x, c) = sample_bilinear(warped, warped_x, warped_y, W, H, c);
            end
        end
    end
end

function result = smoothstep(t)
    if t <= 0
        result = 0;
    elseif t >= 1
        result = 1;
    else
        % Cubic smoothstep: 3t² - 2t³
        result = t * t * (3 - 2 * t);
    end
end

% Bilinear interpolation
function pixel = sample_bilinear(I, src_x, src_y, W, H, channel)
    if src_x < 1
        src_x = 1;
    elseif src_x > W
        src_x = W;
    end
    
    if src_y < 1
        src_y = 1;
    elseif src_y > H
        src_y = H;
    end
    
    % Four neighboring pixels
    x0 = floor(src_x);
    y0 = floor(src_y);
    
    x1 = x0 + 1;
    if x1 > W
        x1 = W;
    end
    
    y1 = y0 + 1;
    if y1 > H
        y1 = H;
    end
    
    % Interpolation weights
    wx = src_x - x0;
    wy = src_y - y0;
    
    % pixel values of the four neighboring pixels
    p00 = double(I(y0, x0, channel));
    p10 = double(I(y0, x1, channel));
    p01 = double(I(y1, x0, channel));
    p11 = double(I(y1, x1, channel));
    
    value = (1-wx)*(1-wy)*p00 + wx*(1-wy)*p10 + (1-wx)*wy*p01 + wx*wy*p11;
    pixel = uint8(value);
end 