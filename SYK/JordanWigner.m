function [whichPauli,coeff] = JordanWigner(whichMajorana,N)
    % e.g. whichMajorana = [5,3,2,1] means \chi_5 \chi_3 \chi_2 \chi_1
    % whichPauli = [3,3,0,1] means Z_4 Z_3 I_2 X_1
    % N is the number of fermions.
    % note: convention is that least significant bit is 1.

    whichPaulis = zeros(length(whichMajorana),N/2);
    for i = 1:length(whichMajorana)
        type = mod(whichMajorana(i)+1,2)+1;
        qubit = ceil(whichMajorana(i)/2);
        whichPaulis(i,qubit) = type;
        for j = 1:(qubit-1)
            whichPaulis(i,j) = 3;
        end
    end
    
    if length(whichMajorana) == 1
        whichPauli = whichPaulis;
        coeff = 1;
    else
        [whichPauli,coeff] = pauliProduct(whichPaulis(2,:),whichPaulis(1,:));
        for i = 3:length(whichMajorana)
            [whichPauli,new_coeff] = pauliProduct(whichPaulis(i,:),whichPauli);
            coeff = coeff*new_coeff;
        end
    end
end

