function [prod, coeff] = pauliProduct(pauliL,pauliR)
    prod = zeros(1,length(pauliL));
    coeff = 1;
    for i = 1:length(pauliL)
        if pauliL(i) == 0
            prod(i) = pauliR(i);
        elseif pauliR(i) == 0
            prod(i) = pauliL(i);
        elseif pauliL(i) ~= pauliR(i)
            prod(i) = setdiff([1,2,3],[pauliL(i),pauliR(i)]);
            if mod(pauliR(i) - pauliL(i),3) == 1
                coeff = coeff*1i;
            elseif mod(pauliR(i) - pauliL(i),3) == 2
                coeff = coeff*(-1i);
            end
        end
    end
end
