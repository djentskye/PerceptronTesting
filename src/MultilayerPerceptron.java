import java.util.Arrays;
import java.util.Random;

/**
 * Implementation of a single-layer perceptron
 */

public class MultilayerPerceptron {

    //TODO: Add bias nodes

    //The number of input, hidden, output nodes
    private int input, hidden, output;

    //Weight matrix (num of layers - 1, regular weight matrices)
    private double[][][] weights;
    private double[][][] weightDeltas;

    //Input and output vectors
    private double[] netin, nethidden, netout;

    //Learning rate
    private double learningRate;

    /**
     * Creates perceptron
     *
     * @param i num of input nodes
     * @param h num of hidden nodes
     * @param o num of output nodes
     */
    public MultilayerPerceptron(int i, int h, int o) {
        this.input = i;
        this.hidden = h; //This is initially being designed for one layer of hidden nodes
        this.output = o;

        //Create the weight matrix; in this case, it has size i * o
//        this.weights = new double[3][];
//        this.weights[0] = new double[i];
//        this.weights[1] = new double[h];
//        this.weights[2] = new double[o];
        this.weights = new double[2][][]; //We initialize the first array size to 2, as there are 2 array matrices
        this.weights[0] = new double[h][i]; //???
        this.weights[1] = new double[o][h]; //We specifically use o * h because that simplifies weighted sum processing?

        this.weightDeltas = new double[2][][];
        this.weightDeltas[0] = new double[h][i];
        this.weightDeltas[1] = new double[o][h];

        this.netin = new double[i];
        this.nethidden = new double[h];
        this.netout = new double[o];

        learningRate = 0.5;
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

        //Continuous implementation - in this case, the logistic function. Here, k = 2
        return (1/(1 + Math.exp((-2) * 2 * input)));
    }

    /**
     * Presents one pattern to the perceptron and calculates the output
     *
     * @param pattern
     * @return
     */
    public double[] presentPattern(double[] pattern) {
        netin = pattern;

        //Iterate through each hidden node and calculate the weighted sum for each
        for(int i = 0; i < this.hidden; i++) {
            nethidden[i] = threshold(weightedSum(netin, weights[0][i]));
        }

        //Iterate through each output node and calculate the weighted sum for each
        for(int i = 0; i < this.output; i++) {
            netout[i] = threshold(weightedSum(nethidden, weights[1][i]));
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
     * Calculates perceptron error and propagates the error backwards. Momentum is not currently implemented, so local
     * minimums could be a problem.
     *
     * @param target
     */
    public void backprop(double[] target) {
        double delO = 0.0;
        double momentum = 0.1;

        //For each output node, modify the weights
        for(int n = 0; n < output; n++) {
            //Correct the output weights
            for(int m = 0; m < weights[1][n].length; m++) {
                //Referencing pg. 11 from Leonardo Noriega, with the addition of momentum in the form of weightDeltas
                weights[1][n][m] = weights[1][n][m] + (learningRate * (netout[n] * (1 - netout[n]) *
                        (target[n] - netout[n])) * nethidden[m]) + (weightDeltas[1][n][m] * momentum);

                //Momentum, stored in weightDeltas
                weightDeltas[1][n][m] = (learningRate * (netout[n] * (1 - netout[n]) *
                        (target[n] - netout[n])) * nethidden[m]) + (weightDeltas[1][n][m] * momentum);

                //BACK propagation...
                delO += (weights[1][n][m] * (netout[n] * (1 - netout[n]) * (target[n] - netout[n])));
            }
        }

            //Calculate the node's signal error
            //Update the weights for each node
        for(int n = 0; n < hidden; n++) {
            //Correct the output weights
            for(int m = 0; m < weights[0][n].length; m++) {
                //Referencing pg. 11 from Leonardo Noriega
                weights[0][n][m] = weights[0][n][m] + (learningRate * (nethidden[n] * (1 - nethidden[n]) *
                        delO) * netin[m]) + (weightDeltas[0][n][m] * momentum); //This is kind of scuffed...

                weightDeltas[0][n][m] = (learningRate * (nethidden[n] * (1 - nethidden[n]) *
                        delO) * netin[m]) + (weightDeltas[0][n][m] * momentum);
            }
        }
    }

    public double error(double[] target) {
        double summation = 0.0;
        for(int i = 0; i < output; i++) {
            summation += Math.pow((target[i] - netout[i]), 2);
        }

        return ((Math.sqrt(summation)) / output);
    }

    public void fitness(double[][] testingInputs, double[][] testingOutputs) {
        double amountCorrect = 0;
        double totalError = 0;
        for(int i = 0; i < testingInputs.length; i++) {
            this.presentPattern(testingInputs[i]);

            totalError += error(testingOutputs[i]);

            double[] roundedNetout = {Math.round(this.netout[0]), Math.round(this.netout[1]), Math.round(this.netout[2])};

            if(Arrays.equals(roundedNetout, testingOutputs[i])) {
                amountCorrect++;
            }
        }

        System.out.println("Current error: " + (totalError / testingInputs.length) + "%");

        //Prints fitness of model out of 1, 1 being perfect and 0 being completely incorrect
        System.out.println("Current fitness: " + amountCorrect + "/" + testingInputs.length + " = " + (amountCorrect/testingInputs.length));
    }
}
