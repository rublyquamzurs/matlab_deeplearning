classdef TriplteLossLayer < nnet.internal.cnn.layer.RegressionLayer
    properties
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % Name (char array)   A name for the layer
        Name
        
        % ResponseNames (cellstr)   The names of the responses
        ResponseNames
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'regressionoutput'
    end
    
    properties (SetAccess = private)
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined = true
    end
 
    methods
        function this = TriplteLossLayer(name)
            this.Name = name;
            this.ResponseNames = {};
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            % forwardPropagateSize  Output the size of the layer based on
            % the input size
            outputSize = inputSize;
        end
        
        function this = inferSize(this, ~)
            
            % no-op since this layer has nothing that can be inferred
        end
        
        function tf = isValidInputSize(~, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size.
            tf = numel(inputSize)==3;
        end
        
        function this = initializeLearnableParameters(this, ~)
            
            % no-op since there are no learnable parameters
        end
        
        function this = prepareForTraining(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter.empty();
        end
        
        function this = prepareForPrediction(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        end
        
        function this = setupForHostPrediction(this)
        end
        
        function this = setupForGPUPrediction(this)
        end
        
        function this = setupForHostTraining(this)
        end
        
        function this = setupForGPUTraining(this)
        end

        function loss = forwardLoss(this, Y, T)
            % Return the loss between the predictions Y and the 
            % training targets T
            %
            % Inputs:
            %         layer - Output layer
            %         Y     每 Predictions made by network
            %         T     每 Training targets
            %
            % Output:
            %         loss  - Loss between Y and T
            
            % Sort triplte to anchor  ,Positive and Negative
            Anchor = Y(:,:,:,1:3:end) ;
            Pos = Y(:,:,:,2:3:end) ;
            Neg = Y(:,:,:,3:3:end) ;
            
            % Normlize to obtain Norm of 1
            Anchor_Norm = bsxfun(@rdivide ,Anchor,sqrt(sum(Anchor.^2))) ;
            Pos_Norm = bsxfun(@rdivide ,Pos,sqrt(sum(Pos.^2))) ;
            Neg_Norm = bsxfun(@rdivide ,Neg,sqrt(sum(Neg.^2))) ;            
            % Calculate Positive and negative distance
            PosDiff = sqrt(sum(squeeze(Anchor_Norm-Pos_Norm).^2));
            NegDiff = sqrt(sum(squeeze(Anchor_Norm-Neg_Norm).^2));
            % Layer forward loss function goes here
            loss = 0.2 -   median( NegDiff )+( median( PosDiff )  ) ;
           % disp(['pos dist: ' num2str(median( PosDiff  )) ' ,  Neg dist: ' num2str(median(NegDiff)) ' ,  Diff: ' num2str(median(NegDiff)-median(PosDiff))])
            loss = max(0,loss);
        end
        
        function dLdX = backwardLoss(this, Y, T)
            % Backward propagate the derivative of the loss function
            %
            % Inputs:
            %         layer - Output layer
            %         Y     每 Predictions made by network
            %         T     每 Training targets
            %
            % Output:
            %         dLdX  - Derivative of the loss with respect to the input X        
            N = size(Y,4);
            % Sort triplte to anchor  ,Positive and Negative
            Anchor = Y(:,:,:,1:3:end) ;
            Pos = Y(:,:,:,2:3:end) ;
            Neg = Y(:,:,:,3:3:end) ;
            
            
            % Check if positive distanse bigger then negative to set
            % grdient decsent direction
            DiffTriplte1 =  sign( Neg - Pos );
            DiffTriplte2 =  sign( Pos - Anchor );
            DiffTriplte3 =  sign( Neg - Anchor );
            
            DiffTriplte = gpuArray(zeros(size(DiffTriplte1).*([1 1 1 3])));
            % Duplicate three times direction
            DiffTriplte(:,:,:,1:size(DiffTriplte1,4)) = DiffTriplte1 ;
            DiffTriplte(:,:,:,(1:size(DiffTriplte1,4)) + size(DiffTriplte1,4)) = DiffTriplte2 ;
            DiffTriplte(:,:,:,(1:size(DiffTriplte1,4)) + 2*size(DiffTriplte1,4)) = DiffTriplte3 ;
            
            % Set Gradient
            dLdX = DiffTriplte/(N^2);
            % Layer backward loss function goes here
        end
    end
end