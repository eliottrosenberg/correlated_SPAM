function [whichPauli,coeff] = BravyiKitaev(whichMajorana,N)
    % e.g. whichMajorana = [5,3,2,1] means \chi_5 \chi_3 \chi_2 \chi_1
    % whichPauli = [3,3,0,1] means Z_4 Z_3 I_2 X_1
    % N is the number of fermions.
    % note: convention is that least significant bit is 1.

    whichPaulis = zeros(length(whichMajorana),N/2);
    for i = 1:length(whichMajorana)
        type = mod(whichMajorana(i)+1,2)+1;
        qubit = ceil(whichMajorana(i)/2);
        x_indices = partial_order(qubit-1,N/2)+1;
        if type == 1
            z_indices = L_set(qubit-1,N/2)+1;
        elseif type == 2
            z_indices = L_set(qubit,N/2)+1;
        end
        y_indices = intersect(x_indices,z_indices);
        whichPaulis(i,x_indices) = 1;
        whichPaulis(i,z_indices) = 3;
        whichPaulis(i,y_indices) = 2;
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




function j = partial_order(i,n)
    % returns all j >= i, using the partial order above Eq. 19 in the
    % Bravyi-Kitaev paper. n is the number of qubits
    
    j = zeros(n,1);
    i_binary = de2bi(i,n);
    
    for l0 = 1:n
        j_l0 = i_binary;
        j_l0(1:(l0-1)) = 1;
        j(l0) = bi2de(j_l0);
    end

    j = unique(j);
    j(j>(n-1)) = [];
end


function k = L_set(i,n)
    % returns the elements in the set L from Eq. 21 of the Bravyi-Kitaev
    % paper
    
    i_binary = de2bi(i,n);
    num_ones = sum(i_binary);
    l0_all = find(i_binary);
    k = zeros(num_ones,1);
    
    for which_l0 = 1:num_ones
        l0 = l0_all(which_l0);
        k_l0 = i_binary;
        k_l0(l0) = 0;
        k_l0(1:(l0-1)) = 1;
        k(which_l0) = bi2de(k_l0);
    end
    k = unique(k);
    k(k>(n-1)) = [];
end
