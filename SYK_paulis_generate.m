clearvars
global N
N = 20;
majoranas = nchoosek(1:N,4);
numTerms = nchoosek(N,4);
paulis = zeros(numTerms,N/2);
coeffs = zeros(numTerms,1);
for i = 1:numTerms
    [paulis(i,:),coeffs(i)] = JordanWigner(majoranas(i,:),N);
end

% now store parameters for each of the above Pauli operators
l = 3;
startIndex = 4151;
numTrials = 50;
for i = 1:numTerms
    if ~isequal(paulis(i,:),[0,3,0,3]) %skip this one since we already have it.
        generate_thetas(l,paulis(i,:)',numTrials,startIndex)
        startIndex = startIndex+numTrials;
    end
    disp(['% complete = ',num2str(i/numTerms)])
end