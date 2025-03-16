% Ayush Goyal
% 7184517074
% ayushgoy@usc.edu
% March 16, 2025

% Problem 3 - Morphological Image Processing
function Prob3()
    % Part A: Thinning
    procImg('P3/Spring.raw', 'Spring', 512, 512);
    procImg('P3/Flower.raw', 'Flower', 512, 512);
    procImg('P3/Circle.raw', 'Circle', 512, 512);
    procImg('P3/Tree.raw', 'Tree', 512, 512);
    
    % Part B: Defect detection
    procChest('P3/Chest_cavity.raw', 410, 305);
end

function procImg(filename, imgName, width, height)
    raw_img = readRaw(filename, width, height);
    bin_img = binImg(raw_img, width, height);
    
    [thin_img, iters, inter_results] = thin(bin_img, width, height, 20);
    
    figure('Name', ['Results for ' imgName]);
    subplot(1, 2, 1); imshow(bin_img); title('Binary Image');
    subplot(1, 2, 2); imshow(thin_img); title(['Thinned (' num2str(iters) ' iterations)']);
    
    figure('Name', ['Thinning for ' imgName]);
    key_iters = [1, 5, 10, 15, 20];
    num_disp = min(length(key_iters), length(inter_results));
    
    for i = 1:num_disp
        idx = key_iters(i);
        if idx <= length(inter_results)
            subplot(2, 3, i);
            imshow(inter_results{idx});
            title(['Iter ' num2str(idx)]);
        end
    end
end

function procChest(filename, width, height)
    raw_img = readRaw(filename, width, height);
    bin_img = binImg(raw_img, width, height);
    
    [lab_img, num_labs, lab_sizes] = labelCC(bin_img, width, height);
    [def_labs, def_sizes, def_freq] = findDefects(lab_sizes, 50);
    
    corr_img = fixDefects(bin_img, lab_img, def_labs, width, height);
    
    figure('Name', 'Chest Cavity Analysis');
    subplot(1, 2, 1); imshow(bin_img); title('Binary Chest Image');
    subplot(1, 2, 2); imshow(corr_img); title(['Corrected Image (' num2str(length(def_labs)) ' defects)']);
    
    disp(['Total defects: ' num2str(length(def_labs))]);
    
    uniq_count = 0;
    for i = 1:length(def_freq)
        if def_freq(i) > 0
            uniq_count = uniq_count + 1;
        end
    end
    disp(['Different defect sizes: ' num2str(uniq_count)]);
    
    for i = 1:length(def_freq)
        if def_freq(i) > 0
            disp(['Size ' num2str(i) ' pixels: ' num2str(def_freq(i)) ' defects']);
        end
    end
end

function img = readRaw(filename, width, height)
    fid = fopen(filename, 'r');
    raw_data = fread(fid, width*height, 'uint8');
    fclose(fid);
    
    img = reshape(raw_data, width, height);
    img = permute(img, [2, 1]);
end

function bin_img = binImg(img, width, height)
    Fmax = 0;
    for i = 1:height
        for j = 1:width
            if img(i, j) > Fmax
                Fmax = img(i, j);
            end
        end
    end
    thresh = 0.5 * Fmax;
    bin_img = zeros(height, width);
    for i = 1:height
        for j = 1:width
            if img(i, j) > thresh
                bin_img(i, j) = 1;
            end
        end
    end
end

function [result, iters, inter_results] = thin(img, width, height, max_iters)
    result = img;
    inter_results = cell(1, max_iters);
    iters = 0;
    changed = true;
    while changed && iters < max_iters
        iters = iters + 1;
        changed = false;
        temp = result;

        for i = 2:height-1
            for j = 2:width-1
                if temp(i, j) == 1            
                    p2 = temp(i-1, j);
                    p3 = temp(i-1, j+1);
                    p4 = temp(i, j+1);
                    p5 = temp(i+1, j+1);
                    p6 = temp(i+1, j);
                    p7 = temp(i+1, j-1);
                    p8 = temp(i, j-1);
                    p9 = temp(i-1, j-1);
                    
                    B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                    if B >= 2 && B <= 6
                        A = 0;
                        if (p2 == 0 && p3 == 1) || (p3 == 0 && p4 == 1) || ...
                           (p4 == 0 && p5 == 1) || (p5 == 0 && p6 == 1) || ...
                           (p6 == 0 && p7 == 1) || (p7 == 0 && p8 == 1) || ...
                           (p8 == 0 && p9 == 1) || (p9 == 0 && p2 == 1)
                            A = A + 1;
                        end
                        
                        if A == 1
                            if (p2 * p4 * p6 == 0) && (p4 * p6 * p8 == 0)
                                result(i, j) = 0;
                                changed = true;
                            end
                        end
                    end
                end
            end
        end

        temp = result;
        for i = 2:height-1
            for j = 2:width-1
                if temp(i, j) == 1
                    p2 = temp(i-1, j);
                    p3 = temp(i-1, j+1);
                    p4 = temp(i, j+1);
                    p5 = temp(i+1, j+1);
                    p6 = temp(i+1, j);
                    p7 = temp(i+1, j-1);
                    p8 = temp(i, j-1);
                    p9 = temp(i-1, j-1);
                    
                    B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                    if B >= 2 && B <= 6
                        A = 0;
                        if (p2 == 0 && p3 == 1) || (p3 == 0 && p4 == 1) || ...
                           (p4 == 0 && p5 == 1) || (p5 == 0 && p6 == 1) || ...
                           (p6 == 0 && p7 == 1) || (p7 == 0 && p8 == 1) || ...
                           (p8 == 0 && p9 == 1) || (p9 == 0 && p2 == 1)
                            A = A + 1;
                        end
                        
                        if A == 1
                            if (p2 * p4 * p8 == 0) && (p2 * p6 * p8 == 0)
                                result(i, j) = 0;
                                changed = true;
                            end
                        end
                    end
                end
            end
        end
        
        if iters == 1 || iters == 5 || iters == 10 || iters == 15 || iters == 20
            inter_results{iters} = result;
        end
    end
    
    if iters < max_iters
        for i = iters+1:max_iters
            inter_results{i} = result;
        end
        iters = max_iters;
    end
end

% Label Connected Components
function [lab_img, num_labs, lab_sizes] = labelCC(bin_img, width, height)
    lab_img = zeros(height, width);
    next_lab = 1;
    eq_table = zeros(width * height, 1);
    for i = 1:width * height
        eq_table(i) = i;
    end
    for i = 1:height
        for j = 1:width
            if bin_img(i, j) == 0
                neighs = [];
                n_count = 0;         
                for ni = max(1, i-1):min(height, i+1)
                    for nj = max(1, j-1):min(width, j+1)
                        if (ni ~= i || nj ~= j) && bin_img(ni, nj) == 0 && lab_img(ni, nj) > 0
                            n_count = n_count + 1;
                            neighs(n_count) = lab_img(ni, nj);
                        end
                    end
                end
                if n_count == 0
                    lab_img(i, j) = next_lab;
                    next_lab = next_lab + 1;
                else
                    min_lab = neighs(1);
                    for k = 2:n_count
                        if neighs(k) < min_lab
                            min_lab = neighs(k);
                        end
                    end 
                    lab_img(i, j) = min_lab;
                    for k = 1:n_count
                        unionFind(eq_table, min_lab, neighs(k));
                    end
                end
            end
        end
    end
    
    for i = 1:length(eq_table)
        eq_table(i) = findRoot(eq_table, i);
    end
    uniq_labs = [];
    uniq_count = 0;
    for i = 1:length(eq_table)
        if eq_table(i) > 0
            is_uniq = true;
            for j = 1:uniq_count
                if eq_table(i) == uniq_labs(j)
                    is_uniq = false;
                    break;
                end
            end
            
            if is_uniq
                uniq_count = uniq_count + 1;
                uniq_labs(uniq_count) = eq_table(i);
            end
        end
    end
    
    remap = zeros(next_lab, 1);
    for i = 1:uniq_count
        remap(uniq_labs(i)) = i;
    end
    num_labs = uniq_count;
    for i = 1:height
        for j = 1:width
            if lab_img(i, j) > 0
                orig_lab = lab_img(i, j);
                root_lab = eq_table(orig_lab);
                lab_img(i, j) = remap(root_lab);
            end
        end
    end
    
    lab_sizes = zeros(num_labs, 1);
    for i = 1:height
        for j = 1:width
            lab = lab_img(i, j);
            if lab > 0
                lab_sizes(lab) = lab_sizes(lab) + 1;
            end
        end
    end
end

% Union Find
% Root of the label
function root = findRoot(eq_table, label)
    if eq_table(label) ~= label
        eq_table(label) = findRoot(eq_table, eq_table(label));
    end
    root = eq_table(label);
end

function unionFind(eq_table, lab1, lab2)
    root1 = findRoot(eq_table, lab1);
    root2 = findRoot(eq_table, lab2);
    
    if root1 < root2
        eq_table(root2) = root1;
    else
        eq_table(root1) = root2;
    end
end

function [def_labs, def_sizes, def_freq] = findDefects(lab_sizes, max_size)
    def_labs = [];
    def_sizes = [];
    def_count = 0;
    
    for i = 1:length(lab_sizes)
        if lab_sizes(i) > 0 && lab_sizes(i) < max_size
            def_count = def_count + 1;
            def_labs(def_count) = i;
            def_sizes(def_count) = lab_sizes(i);
        end
    end
    
    if def_count > 0
        max_sz = def_sizes(1);
        for i = 2:def_count
            if def_sizes(i) > max_sz
                max_sz = def_sizes(i);
            end
        end
        
        def_freq = zeros(max_sz, 1);
        
        for i = 1:def_count
            sz = def_sizes(i);
            def_freq(sz) = def_freq(sz) + 1;
        end
    else
        def_freq = [];
    end
end

function corr_img = fixDefects(bin_img, lab_img, def_labs, width, height)
    corr_img = bin_img;
    
    for i = 1:height
        for j = 1:width
            lab = lab_img(i, j);
            if lab > 0
                is_def = false;
                for k = 1:length(def_labs)
                    if lab == def_labs(k)
                        is_def = true;
                        break;
                    end
                end
                
                if is_def
                    corr_img(i, j) = 1;
                end
            end
        end
    end
end

