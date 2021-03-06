classdef test1
    properties
        Name
        Description
        x
        y
    end
 
    methods
        function layer = test1(Name)           
            % (Optional) Create a myClassificationLayer
            % Set layer name
            if nargin == 1
                layer.Name = Name;
            end

            % Set layer description
            layer.Description = 'Triplte_loss_layer';
            % Layer constructor function goes here
        end

        function loss = forwardLoss(layer, a)
            layer.backwardLoss()
            disp('forwardloss')
        end
        
        function dLdX = backwardLoss(layer, a)
%             layer.forwardLoss()
            disp('backwardloss')
        end
    end
end