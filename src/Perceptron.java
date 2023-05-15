import java.util.Random;

/**
 * Implementation of a single-layer perceptron
 */

public class Perceptron {

    //The number of input, hidden, output nodes
    private int input, hidden, output;

    //Weight matrix
    private double[][] weights;

    //Input and output vectors
    private double[] netin, netout;

    //Learning rate
    private double learningRate;

    /**
     * Creates perceptron
     *
     * @param i num of input nodes
     * @param h num of hidden nodes
     * @param o num of output nodes
     */
    public Perceptron(int i, int h, int o) {
        this.input = i;
        this.hidden = h; //Since this is a single-layer perceptron, the hidden layer should be 0
        this.output = o;

        //Create the weight matrix; in this case, it has size i * o
//        this.weights = new double[3][];
//        this.weights[0] = new double[i];
//        this.weights[1] = new double[h];
//        this.weights[2] = new double[o];
        this.weights = new double[o][i]; //We specifically use o * i because that simplifies weighted sum processing

        this.netin = new double[i];
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
                weights[i][j] = ((rand.nextDouble() * 2) - 1); //Set the weight to a random value between -1 and 1
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
        if(input > 0) {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    /**
     * Presents one pattern to the perceptron and calculates the output
     *
     * @param pattern
     * @return
     */
    public double[] presentPattern(double[] pattern) {
        netin = pattern;

        //Iterate through each output node and calculate the weighted sum for each
        for(int i = 0; i < this.output; i++) {
            netout[i] = threshold(weightedSum(netin, weights[i]));
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
        System.out.print("  Weights: ");
        for(double[] b : weights) {
            for(double c : b) {
                System.out.print(c + ", ");
            }
            System.out.println();
        }
        System.out.print("  Output nodes: ");
        for(double d : netout) {
            System.out.print(d + ", ");
        }
        System.out.println();
    }

    /**
     * Calculates perceptron error and propagates the error backwards
     *
     * @param target
     */
    public void backprop(double[] target) {
        //For each output node, calculate the node error and then modify the weights
        for(int n = 0; n < output; n++) {
            //Calculate error
            double outputNodeError = target[n] - netout[n];

            //Add the node's error to each weight that connects to node n, multiplying by the learning rate so as to
            //  reduce the size of steps.
            for(int m = 0; m < weights[n].length; m++) {
                weights[n][m] += outputNodeError * learningRate;
            }
        }
    }
}
