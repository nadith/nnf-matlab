function [ vector ] = rand_unq( count, max )
%rand_unq: Generate random vector with unique values
if (count > max)
    error(['Err: element count is higher than maximum value provided']);
end
  
vector = -1 * ones(1, count);
for i=1:count
    while (true)
        value = round(rand()* (max-1)) + 1;
        if (isempty(find(vector == value)))
            break;
        end
    end
    vector(i) = value;
end