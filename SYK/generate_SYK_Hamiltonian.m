clearvars


N = 65*2;
q = 4;
majoranas = nchoosek(1:N,q);
numTerms = nchoosek(N,q);

numToMeasure = 25;
whichRandTerms = randperm(numTerms,numToMeasure);

paulis = zeros(numToMeasure,N/2);
coeffs = zeros(numToMeasure,1);
for i = 1:numToMeasure
    [paulis(i,:),coeffs(i)] = JordanWigner(majoranas(whichRandTerms(i),:),N);
end

numMeasured = sum( paulis > 0, 2);



[numMeasured_sorted,remaining] = sort(numMeasured,'descend');
paulis_measured = [];
terms_measured = {};

while ~isempty(remaining)
    disp(['number remaining = ',num2str(length(remaining))])
    pauli_to_measure = paulis(remaining(1),:);
    terms_measured_i = remaining(1);
    remaining(1) = [];
    measured_without_assignment = already_measured(remaining,pauli_to_measure,paulis);
    terms_measured_i = [terms_measured_i; remaining(measured_without_assignment)];
    remaining(measured_without_assignment) = [];
    
    pauli_to_measure = find_assignment(pauli_to_measure,remaining,paulis);
    measured_after_assignment = already_measured(remaining,pauli_to_measure,paulis);
    terms_measured_i = [terms_measured_i; remaining(measured_after_assignment)];
    remaining(measured_after_assignment) = [];
    
    paulis_measured = [paulis_measured; pauli_to_measure];
    terms_measured{end+1} = terms_measured_i;
    
    
end


writematrix(paulis,'paulis.csv')
writematrix(paulis_measured,'paulis_measured.csv')
fid = fopen('terms_measured.csv','w');
for tm = 1:length(terms_measured)
    fprintf(fid,'%d,',terms_measured{tm}-1); % the -1 is because this will be imported into python
    fprintf(fid,'%d\n',[]);
end
fclose(fid);

function [measured] = already_measured(remaining,pauli_measured,paulis)

    pauli = paulis(remaining,:);
    %measured = prod( pauli == 0 | pauli == pauli_measured | pauli_measured == 0);
    measured = logical(prod( pauli == 0 | pauli == pauli_measured ,2));
    %require_assignment = measured & ~measured_without_assignment;
end


function assignments = all_assignments(remaining,pauli_measured,paulis)
    % assumes that already_measured have been removed from remaining.
    pauli = paulis(remaining,:);
    compatible = logical(prod( pauli == 0 | pauli == pauli_measured | pauli_measured == 0,2));
    qubits_to_assign = pauli_measured == 0;
    num_qubits_to_assign = sum(qubits_to_assign);
    disp(['Using largest assignment. Number of qubits to assign = ',num2str(num_qubits_to_assign)])
    
    assignments = pauli(compatible,qubits_to_assign);
    
    % now run the original algorithm on assignments
    
    
    

end


function pauli = assign_qubits(pauli,assignment)
    
    qubits_to_assign = find(pauli == 0);
    num_qubits_to_assign = length(qubits_to_assign);
    assignment_str = dec2base(assignment,3,num_qubits_to_assign);
    for i = 1:num_qubits_to_assign
        q = qubits_to_assign(i);
        pauli(q) = str2double(assignment_str(i))+1;
    end

end


function pauli_assigned = find_optimal_assignment(pauli,remaining,paulis)

    qubits_to_assign = pauli == 0;
    num_qubits_to_assign = sum(qubits_to_assign);
    disp(['Using full search. Number of qubits to assign = ',num2str(num_qubits_to_assign)])
    if num_qubits_to_assign == 0
        pauli_assigned = pauli;
        return
    else
        num_measured_per_assignment = zeros(3^num_qubits_to_assign,1);
        for assignment = 0:(3^(num_qubits_to_assign)-1)
            pauli_i = assign_qubits(pauli,assignment);
            num_measured_per_assignment(assignment+1) = sum(already_measured(remaining,pauli_i,paulis));
        end
        
        % now pick the assignment that has the largest num_measured_per_assignment
        
        max_num_measured_per_assignment = max(num_measured_per_assignment);
        if max_num_measured_per_assignment == 0
            pauli_assigned = pauli;
            return
        elseif max_num_measured_per_assignment > 0
            % for now, just pick the first optimal assignment. Later, may
            % want to do better than this.
            assignment = find(num_measured_per_assignment == max_num_measured_per_assignment,1) - 1;
            pauli_assigned = assign_qubits(pauli,assignment);
        end
    end

end


function pauli_assigned = find_good_assignment(pauli,remaining,paulis)
    % this one isn't exponentially difficult and can be used in a loop
    % until it converges
    qubits_to_assign = pauli == 0;
    num_qubits_to_assign = sum(qubits_to_assign);
    if num_qubits_to_assign == 0
        pauli_assigned = pauli;
        return
    else
        possible_assignments = all_assignments(remaining,pauli,paulis);
        if numel(possible_assignments) ~= 0
            num_assigned = sum( possible_assignments > 0, 2);
            [max_assigned, index] = max(num_assigned); % find the assignment that assigns the most qubits
            assignment = possible_assignments(index,:);
            pauli(qubits_to_assign) = assignment;
            pauli_assigned = pauli;
            return
        else
            pauli_assigned = pauli;
            return
        end
    end
end

function pauli_assigned = find_assignment(pauli,remaining,paulis)
    cutoff = 9; % do full search if fewer than this many qubits are to be assigned
    qubits_to_assign = pauli == 0;
    num_qubits_to_assign = sum(qubits_to_assign);
    if num_qubits_to_assign == 0
        pauli_assigned = pauli;
        return
    elseif num_qubits_to_assign <= cutoff
        pauli_assigned = find_optimal_assignment(pauli,remaining,paulis);
        return
    else
        num_qubits_to_assign_previous = 0;
        while num_qubits_to_assign ~= num_qubits_to_assign_previous
            pauli = find_good_assignment(pauli,remaining,paulis);
            % now check how many qubits remain to assign:
            num_qubits_to_assign_previous = num_qubits_to_assign;
            qubits_to_assign = pauli == 0;
            num_qubits_to_assign = sum(qubits_to_assign);
            if num_qubits_to_assign == 0
                pauli_assigned = pauli;
                return
            elseif num_qubits_to_assign <= cutoff
                pauli_assigned = find_optimal_assignment(pauli,remaining,paulis);
                return
            else
                measured = already_measured(remaining,pauli,paulis);
                remaining(measured) = [];
            end
        end
        pauli_assigned = pauli;
    end
end
