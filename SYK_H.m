function H = SYK_H(N,J,q)
    addpath('G:\My Drive\QuantiSED Meetings\Matlab code\Upload from CLASSE Cluster\Qubit Operations')
    H = sparse(2^(N/2),1);
    majoranas = nchoosek(1:N,q);
    numTerms = nchoosek(N,q);
    for i = 1:numTerms
        [paulis,coeff] = JordanWigner(majoranas(i,:),N);
        Pi = pauli_list_to_matrix(paulis);
        H = H +  coeff*J(i)*Pi;
        disp(['% complete = ',num2str(i/numTerms)])
    end

end