import java.util.Arrays;
import java.util.Random;

/**
 * Implementation of a single-layer perceptron
 */

public class MultilayerPerceptron {

    //TODO: Add bias nodes

    //The number of input nodes, output nodes, and hidden layers
    private int input, hiddenLayers, output;

    //The numbers of nodes in each hidden layer. The size of this array should be hidden_layers
    private int[] hiddenNodes;

    //Weight matrix (num of layers - 1, regular weight matrices)
    //Format is: Origin node layer
    //           Destination node index
    //           Origin node index
    private double[][][] weights;

    //Momentum deltas for each weight.
    private double[][][] weightDeltas;

    //Input and output vectors
    private double[] netin, netout;
    private double[][] nethidden;

    //Input and output pre-activation
    //This is becoming a memory nightmare hahaha
    private double[] netin_unactiv, netout_unactiv;
    private double[][] nethidden_unactiv;

    //Inner and output vector deltas
//    private double[] netout_deltas;
//    private double[][] nethidden_deltas;
    private double[][] deltas;

    //Learning rate
    private double learningRate;

    /**
     * Creates perceptron
     *
     * @param i num of input nodes
     * @param h num of hidden layers, minimum 1
     * @param o num of output nodes
     * @param h_nodes num of hidden nodes in each layer, length of array should be h
     */
    public MultilayerPerceptron(int i, int h, int o, int[] h_nodes) {
        this.input = i;
        this.hiddenLayers = h; //This is assumed to be at least 1, otherwise just use the non-multilayer perceptron
        this.output = o;

        this.hiddenNodes = h_nodes;

        //Create the weight matrix; in this case, it has size i * o
        this.weights = new double[h + 1][][]; //We initialize the first array size to h + 1, the number of weights we
                                              //should end up with.

        //Create the first layer of weights between the input layer and the first layer of hidden nodes
        this.weights[0] = new double[h_nodes[0]][i];

        //Create all the layers of weights between the hidden nodes
        for(int j = 1; j < h; j++) {
            this.weights[j] = new double[h_nodes[j]][h_nodes[j-1]];
        }

        //Create the layer of weights between the last hidden layer and the output layer
        this.weights[h] = new double[o][h_nodes[h-1]]; //We specifically use o * h because that simplifies weighted sum processing

        //Create weightDeltas, maybe split this up later for readability?
        this.weightDeltas = new double[h + 1][][];

        this.weightDeltas[0] = new double[h_nodes[0]][i];

        for(int j = 1; j < h; j++) {
            this.weightDeltas[j] = new double[h_nodes[j]][h_nodes[j-1]];
        }

        this.weightDeltas[h] = new double[o][h_nodes[h-1]];

        this.netin = new double[i];
        this.nethidden = new double[h][]; //nethidden contains the hidden nodes, first dimension is the layer, second
                                          //dimension is the node within said layer
        for(int j = 0; j < h; j++) {
            this.nethidden[j] = new double[h_nodes[j]];
        }
        this.netout = new double[o];

        //Initialize unactivated variables
        this.nethidden_unactiv = new double[h][]; //nethidden contains the hidden nodes, first dimension is the layer, second
        //dimension is the node within said layer
        for(int j = 0; j < h; j++) {
            this.nethidden_unactiv[j] = new double[h_nodes[j]];
        }
        this.netout_unactiv = new double[o];

//        //Initialize deltas
//        this.nethidden_deltas = new double[h][]; //nethidden contains the hidden nodes, first dimension is the layer, second
//        //dimension is the node within said layer
//        for(int j = 0; j < h; j++) {
//            this.nethidden_deltas[j] = new double[h_nodes[j]];
//        }
//        this.netout_deltas = new double[o];

        this.deltas = new double[h + 1][]; //We initialize the first array size to h + 1, the number of delta columns we
        //should end up with.

        //Create all the layers of deltas for the hidden nodes
        for(int j = 0; j < h; j++) {
            this.deltas[j] = new double[h_nodes[j]];
        }

        //Create the layer of deltas for the output layer
        this.deltas[h] = new double[o]; //We specifically use o * h because that simplifies weighted sum processing



//        learningRate = 0.5;
        learningRate = 1;
    }


    //NOTE: not used at the moment. needs to be updated.
    public MultilayerPerceptron(int i, int h, int o, int[] h_nodes, double[][][] weights) {
        this.input = i;
        this.hiddenLayers = h; //This is assumed to be at least 1, otherwise just use the non-multilayer perceptron
        this.output = o;

        this.hiddenNodes = h_nodes;

        //Create the weight matrix; in this case, it has size i * o
//        this.weights = new double[3][];
//        this.weights[0] = new double[i];
//        this.weights[1] = new double[h];
//        this.weights[2] = new double[o];
        this.weights = weights;

        //Create weightDeltas, maybe split this up later for readability?
        this.weightDeltas = new double[h + 1][][];

        this.weightDeltas[0] = new double[h_nodes[0]][i];

        for(int j = 1; j < h; j++) {
            this.weightDeltas[j] = new double[h_nodes[j]][h_nodes[j-1]];
        }

        this.weightDeltas[h] = new double[o][h_nodes[h-1]];

        this.netin = new double[i];
        this.nethidden = new double[h][]; //nethidden contains the hidden nodes, first dimension is the layer, second
        //dimension is the node within said layer
        for(int j = 0; j < h; j++) {
            this.nethidden[j] = new double[h_nodes[j]];
        }
        this.netout = new double[o];

//        learningRate = 0.5;
        learningRate = 2;
    }

    public int getInput() {
        return input;
    }

    public int getHiddenLayers() {
        return hiddenLayers;
    }

    public int[] getHiddenNodes() {
        return hiddenNodes;
    }

    public int getOutput() {
        return output;
    }

    public double[][][] getWeights() {
        return weights;
    }

    public void putWeights(double[][][] weights) {
        this.weights = weights;
    }

    public double[][][] getWeightDeltas() {
        return weightDeltas;
    }

    public void putWeightDeltas(double[][][] weights) {
        this.weights = weights;
    }

    public void changeLearningRate(double rate) {
        this.learningRate = rate;
    }

    /**
     * Initializes the weight matrix of the perceptron with random numbers between -1 and 1
     */
    public void initializePerceptron() {
        Random rand = new Random();

        //Double loop to go through each weight
        for(int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for(int k = 0; k < weights[i][j].length; k++) {
                    weights[i][j][k] = ((rand.nextDouble() * 2) - 1); //Set the weight to a random value between -1 and 1
                    weightDeltas[i][j][k] = 0.0;
                }
            }
        }
    }

    /**
     * Activation (logical step) function
     *
     * @param input weighted sum
     * @return activation, between 0 and 1
     */
    public double threshold(double input) {
//        if(input > 0) {
//            return 1.0;
//        } else {
//            return 0.0;
//        }

        //Continuous implementation - in this case, the logistic function. Here, k = 10
        return (1/(1 + Math.exp((-2) * 10 * input)));
    }

    /**
     * Derivation of the activation function
     *
     * @param input weighted sum
     * @return derivation of the activation function
     */
    public double thresholdDeriv(double input) {
        double threshold = threshold(input);
        return threshold * (1-threshold);
    }

    /**
     * Presents one pattern to the perceptron and calculates the output
     *
     * @param pattern
     * @return
     */
    public double[] presentPattern(double[] pattern) {
        netin = pattern;

        //Iterate through each hidden layer and calculate the weighted sum for each
        for(int j = 0; j < hiddenLayers; j++) {
            if(j == 0) {
                //Iterate through each hidden node in the first layer and calculate the weighted sum for each
                for (int i = 0; i < hiddenNodes[j]; i++) {
                    nethidden_unactiv[j][i] = weightedSum(netin, weights[0][i]);
                    nethidden[j][i] = threshold(weightedSum(netin, weights[0][i]));
                }
            } else {
                //Iterate through each hidden node in the current layer and calculate the weighted sum for each
                for (int i = 0; i < hiddenNodes[j]; i++) {
//                    nethidden_unactiv[j][i] = weightedSum(nethidden[j-1], weights[0][i]);
                    nethidden_unactiv[j][i] = weightedSum(nethidden[j-1], weights[j][i]);
                    //weights is the wrong size here!!!!
                    nethidden[j][i] = threshold(weightedSum(nethidden[j-1], weights[j][i]));
                }
            }
        }

        //Iterate through each output node and calculate the weighted sum for each
        for(int i = 0; i < output; i++) {
//            netout_unactiv[i] = weightedSum(nethidden[hiddenLayers-1], weights[1][i]);
//            netout[i] = threshold(weightedSum(nethidden[hiddenLayers-1], weights[1][i]));
            netout_unactiv[i] = weightedSum(nethidden[hiddenLayers-1], weights[hiddenLayers][i]);
            netout[i] = threshold(weightedSum(nethidden[hiddenLayers-1], weights[hiddenLayers][i]));
        }

        return netout;
    }

    /**
     * Calculates weighted sum to one node. Takes an input layer and the weights from that layer to the node
     *
     * @param inputLayer input layer of nodes
     * @param weights must be the array of weights from the input layer to the output node
     * @return
     */
    private double weightedSum(double[] inputLayer, double[] weights) {
        //Note: we could make a variation of this function which can find the weights itself... not sure if that would
        //      be needed though

        double wSum = 0;

        for(int i = 0; i < inputLayer.length; i++) {
            wSum += inputLayer[i] * weights[i];
        }

        return wSum;
    }

    /**
     * Display the state of the perceptron
     */
    public void showState() {
        System.out.println("Current perceptron state");
        System.out.print("  Input nodes: ");
        for(double a : netin) {
            System.out.print(a + ", ");
        }
        System.out.println();
//        System.out.print("  Weights: ");
//        for(double[] b : weights) {
//            for(double c : b) {
//                System.out.print(c + ", ");
//            }
//            System.out.println();
//        }
        System.out.print("  Output nodes: ");
        for(double d : netout) {
            System.out.print(d + ", ");
        }
        System.out.println();
    }

    /**
     * Calculates perceptron error and propagates the error backwards. Local minimums could be a problem.
     *
     * @param target
     */
    public double[][][] backprop(double[] target) {
        double momentum = 0.25; //TODO: Move the momentum variable out of this function

        //For each output node, find the delta
        for (int n = 0; n < output; n++) {

            //Find the delta of the current output node
            deltas[hiddenLayers][n] = 2 * (target[n] - netout[n]) /*?? or target[n]-netout[n]*/ * thresholdDeriv(netout_unactiv[n]);


            //Correct the weights leading to the output layer
            //Iterate through the last
            for (int m = 0; m < weights[hiddenLayers][n].length; m++) {
                //Referencing pg. 11 from Leonardo Noriega, with the addition of momentum in the form of weightDeltas

                //Momentum, stored in weightDeltas
                weightDeltas[hiddenLayers][n][m] = (momentum * weightDeltas[hiddenLayers][n][m]) + deltas[hiddenLayers][n] * nethidden[hiddenLayers - 1][m] * learningRate;


//                weights[hiddenLayers][n][m] = weights[hiddenLayers][n][m] - (weightDeltas[hiddenLayers][n][m]);
            }
        }

        //Update weights for middle hidden nodes
        //Select the next hidden layer to work on
        for (int a = hiddenLayers - 1; a > 0; a--) { //Work backwards

            for (int n = 0; n < nethidden[a].length; n++) {

                //Find the sum of the deltas from the layer after the one we want to parse
                double deltasum = 0;

                //Iterate through each to find the sum of deltas which we use for this node.
                for (int i = 0; i < deltas[a + 1].length; i++) {
                    deltasum = deltasum + weights[a + 1][i][n] * deltas[a + 1][i];
                }

                //Find the delta of the current hidden node
                deltas[a][n] = 2 * deltasum * thresholdDeriv(nethidden_unactiv[a][n]);

                //Correct the output weights
                for (int m = 0; m < weights[a][n].length; m++) {

                    //Momentum, stored in weightDeltas
                    weightDeltas[a][n][m] = (momentum * weightDeltas[a][n][m]) + deltas[a][n] * nethidden[a - 1][m] * learningRate;

//                    weights[a][n][m] = weights[a][n][m] - (weightDeltas[a][n][m]);
                }
            }
        }

        for (int n = 0; n < nethidden[0].length; n++) {

            //Find the delta of the current output node
            double deltasum = 0;

            //Iterate through each to find the sum of deltas which we use for this node.
            for (int i = 0; i < deltas[1].length; i++) {
                deltasum = deltasum + weights[1][i][n] * deltas[1][i];
            }

            deltas[0][n] = 2 * deltasum * thresholdDeriv(nethidden_unactiv[0][n]);

            //Correct the output weights
            for (int m = 0; m < weights[0][n].length; m++) {

                //Momentum, stored in weightDeltas
                //?????
                weightDeltas[0][n][m] = (momentum * weightDeltas[0][n][m]) + deltas[0][n] * netin[m] * learningRate;

//                    weights[a][n][m] = weights[a][n][m] - (weightDeltas[a][n][m]);
            }
        }

        return weightDeltas;
    }

    public void backprop_weights(double[][][] gradientVector) {
        for(int k = 0; k < gradientVector.length; k++) {
            for(int l = 0; l < gradientVector[k].length; l++) {
                for(int m = 0; m < gradientVector[k][l].length; m++) {
                    weights[k][l][m] = weights[k][l][m] - gradientVector[k][l][m];
                }
            }
        }
    }


//        public void backprop(double[] target) {
//            double momentum = 0.3; //TODO: Move the momentum variable out of this function
//
//            //For each output node, find the delta
//            for(int n = 0; n < output; n++) {
//
//                //Find the delta of the current output node
//                deltas[hiddenLayers][n] = 2 * (target[n]-netout[n]) /*?? or target[n]-netout[n]*/ * thresholdDeriv(netout_unactiv[n]);
//
//
//                //Correct the weights leading to the output layer
//                //Iterate through the last
//                for(int m = 0; m < weights[hiddenLayers][n].length; m++) {
//                    //Referencing pg. 11 from Leonardo Noriega, with the addition of momentum in the form of weightDeltas
//
//                    //Momentum, stored in weightDeltas
//                    weightDeltas[hiddenLayers][n][m] = (momentum * weightDeltas[hiddenLayers][n][m]) + deltas[hiddenLayers][n] * nethidden[hiddenLayers-1][m] * learningRate;
//
//
//                    weights[hiddenLayers][n][m] = weights[hiddenLayers][n][m] - (weightDeltas[hiddenLayers][n][m]);
//                }
//            }
//
//            //Update weights for middle hidden nodes
//            //Select the next hidden layer to work on
//            for(int a = hiddenLayers-1; a > 0; a--) { //Work backwards
//
//                for (int n = 0; n < nethidden[a].length; n++) {
//
//                    //Find the sum of the deltas from the layer after the one we want to parse
//                    double deltasum = 0;
//
//                    //Iterate through each to find the sum of deltas which we use for this node.
//                    for(int i = 0; i < deltas[a+1].length; i++) {
//                        deltasum = deltasum + weights[a+1][i][n] * deltas[a+1][i];
//                    }
//
//                    //Find the delta of the current hidden node
//                    deltas[a][n] = 2 * deltasum * thresholdDeriv(nethidden_unactiv[a][n]);
//
//                    //Correct the output weights
//                    for (int m = 0; m < weights[a][n].length; m++) {
//
//                        //Momentum, stored in weightDeltas
//                        weightDeltas[a][n][m] = (momentum * weightDeltas[a][n][m]) + deltas[a][n] * nethidden[a-1][m] * learningRate;
//
//                        weights[a][n][m] = weights[a][n][m] - (weightDeltas[a][n][m]);
//                    }
//                }
//            }
//
////        //Update weights for the first layer of hidden nodes connected to inputs
////        for (int n = 0; n < hiddenNodes[0]; n++) {
////            //Correct the output weights
////            for (int m = 0; m < weights[0][n].length; m++) {
////                //Referencing pg. 11 from Leonardo Noriega
////                weights[0][n][m] = weights[0][n][m] + (learningRate * (nethidden[0][n] * (1 - nethidden[0][n]) *
////                        delta) * netin[m]) + (weightDeltas[0][n][m] * momentum); //This is kind of scuffed...
////
////                //Momentum...
////                weightDeltas[0][n][m] = (learningRate * (nethidden[0][n] * (1 - nethidden[0][n]) *
////                        delta) * netin[m]) + (weightDeltas[0][n][m] * momentum);
////            }
////        }
//    }

    public double error(double[] target) {
        double summation = 0.0;
        for(int i = 0; i < output; i++) {
            summation += Math.pow((target[i] - netout[i]), 2);
        }

        return ((Math.sqrt(summation)) / output);
    }

    /**
     * Prints the current fitness.
     *
     * @param testingInputs
     * @param testingOutputs
     * @return
     */
    public void fitness(double[][] testingInputs, double[][] testingOutputs) {
        double amountCorrect = 0;
        double totalError = 0;
        for(int i = 0; i < testingInputs.length; i++) {
            this.presentPattern(testingInputs[i]);

            totalError += error(testingOutputs[i]);

            //TODO: Make the length non-hardcoded
//            double[] roundedNetout = {Math.round(this.netout[0]), Math.round(this.netout[1]), Math.round(this.netout[2])};

            //Finds the node that the multilayer perceptron is most confident is the actual output. If two are
            //equivalent, it uses the first one. Could lead to potential bugs, so this is a potential point of failure.
            //TODO: Probably separate this into it's own function at some point soon...
            double[] roundedNetout = new double[netout.length];
            int largest_netout_index = 0;
            for(int j = 0; j < netout.length; j++) {
                if(this.netout[j] > this.netout[largest_netout_index]) {
                    largest_netout_index =  j;
                }
                roundedNetout[j] = 0.0;
            }
            roundedNetout[largest_netout_index] = 1.0;

            //Check if the rounded netout is equal to the testing output. If so, it's correct! Increment.
            if(Arrays.equals(roundedNetout, testingOutputs[i])) {
                amountCorrect++;
            }
        }

        System.out.println("Current error: " + (totalError / testingInputs.length));

        //Prints fitness of model out of 1, 1 being perfect and 0 being completely incorrect
        System.out.println("Current fitness: " + amountCorrect + "/" + testingInputs.length + " = " + (amountCorrect/testingInputs.length));
    }

    /**
     * Returns the current fitness minus the current error.
     *
     * @param testingInputs
     * @param testingOutputs
     * @return
     */
    public double fitnessVal(double[][] testingInputs, double[][] testingOutputs) {
        double amountCorrect = 0;
        double totalError = 0;
        for(int i = 0; i < testingInputs.length; i++) {
            this.presentPattern(testingInputs[i]);

            totalError += error(testingOutputs[i]);

            //TODO: Make the length non-hardcoded
//            double[] roundedNetout = {Math.round(this.netout[0]), Math.round(this.netout[1]), Math.round(this.netout[2])};

            //Finds the node that the multilayer perceptron is most confident is the actual output. If two are
            //equivalent, it uses the first one. Could lead to potential bugs, so this is a potential point of failure.
            //TODO: Probably separate this into it's own function at some point soon...
            double[] roundedNetout = new double[netout.length];
            int largest_netout_index = 0;
            for(int j = 0; j < netout.length; j++) {
                if(this.netout[j] > this.netout[largest_netout_index]) {
                    largest_netout_index =  j;
                }
                roundedNetout[j] = 0.0;
            }
            roundedNetout[largest_netout_index] = 1.0;

            //Check if the rounded netout is equal to the testing output. If so, it's correct! Increment.
            if(Arrays.equals(roundedNetout, testingOutputs[i])) {
                amountCorrect++;
            }
        }

        return /*(amountCorrect/testingInputs.length) -*/ (totalError / testingInputs.length);
    }

    public int[][] getErrorMatrix(double[][] testingInputs, double[][] testingOutputs) {
        int[][] errMatrix = new int[output][output]; //Reference, classified

        for(int i = 0; i < testingInputs.length; i++) {
            this.presentPattern(testingInputs[i]);

            //Detect what class the mlp thought the input belonged to
            int highestValue = 0;
            for(int j = 1; j < netout.length; j++) {
                if(Math.max(netout[highestValue], netout[j]) == netout[j]) {//TODO: just use a > idk
                    highestValue = j;
                }
            }

            //TODO: fix, this is pretty inefficient, but I can't bother to find a better way to do it
            int actualClass = 0;
            for(int j = 1; j < testingOutputs[i].length; j++) {
                if(Math.max(testingOutputs[i][actualClass], testingOutputs[i][j]) == testingOutputs[i][j]) {
                    actualClass = j;
                }
            }

            //Add one to the correct spot for this input
            errMatrix[actualClass][highestValue] += 1;
        }

        return errMatrix;
    }

    public void printErrorMatrix(double[][] testingInputs, double[][] testingOutputs) {
        int[][] errMatrix = getErrorMatrix(testingInputs, testingOutputs);

        //This gets really cursed for large numbers of outputs, but it'll be fine for now
        System.out.print("ErrMatrix   ");
        for(int i = 0; i < output; i++) {
            System.out.print((i+1) + "   ");
        }
        System.out.println("<- Reference data");

        for(int i = 0; i < output; i++) {
            System.out.print((i+1) + "           ");
            for(int j = 0; j < output; j++) {
                System.out.print(errMatrix[j][i] + "   ");
            }

            System.out.println();
        }

        System.out.println("^ Classified");
        System.out.println("| Data");
    }
}
