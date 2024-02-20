import java.util.Random;

public class MainMultiLayer {

    public static double[][] irisInputs = new double[150][4];
    public static double[][] irisOutputs = new double[150][3];

    public static double[][] irisAmyInputs = new double[3][4];
    public static double[][] irisAmyOutputs = new double[3][3];

    public static void runEpoch(MultilayerPerceptron p, double learningRate, int epochSize) {

        p.changeLearningRate(learningRate);

        Random rand = new Random();

        double[][][] gradientVector;

        gradientVector = new double[p.getHiddenLayers()+1][][]; //We initialize the first array size to h + 1, the number of delta columns we
        //should end up with.

        //Create all the layers of deltas for the hidden nodes
        int[] hiddenNodes = p.getHiddenNodes();

        gradientVector[0] = new double[hiddenNodes[0]][p.getInput()];

        for(int j = 1; j < p.getHiddenLayers(); j++) {
            gradientVector[j] = new double[hiddenNodes[j]][hiddenNodes[j-1]];
        }

        //Create the layer of deltas for the output layer
        gradientVector[p.getHiddenLayers()] = new double[p.getOutput()][hiddenNodes[hiddenNodes.length-1]];


        //TODO: Gradient vector goes here! ^^
        //Wow this is a bad solution lol
        for(int i = 0; i < epochSize; i++) {
            int j = rand.nextInt(150);
            p.presentPattern(irisInputs[j]);
            double[][][] tempGradientVector = p.backprop(irisOutputs[j]);
            for(int k = 0; k < tempGradientVector.length; k++) {
                for(int l = 0; l < tempGradientVector[k].length; l++) {
                    for(int m = 0; m < tempGradientVector[k][l].length; m++) {
                        gradientVector[k][l][m] += tempGradientVector[k][l][m];
                    }
                }
            }
        }

        //Make the corrections from the cumulative gradients
        p.backprop_weights(gradientVector);
    }

    public static void runIrisDataset(MultilayerPerceptron p) {
        p.showState();
        p.fitness(irisInputs, irisOutputs);
        p.printErrorMatrix(irisInputs, irisOutputs);

        p.showState();
        runEpoch(p, 0.5, 10);
        p.fitness(irisInputs, irisOutputs);
        p.printErrorMatrix(irisInputs, irisOutputs);

        for(int i = 0; i < 500; i++) {
            runEpoch(p, 0.5, 10);
        }
        p.showState();

        p.fitness(irisInputs, irisOutputs);
        p.printErrorMatrix(irisInputs, irisOutputs);

//        FileIO.saveMultilayerPerceptron(p, "testing_perceptron.txt");
    }

    public static void main(String[] args) {
        ReadCSV.readIrisDataset("iris.data", irisInputs, irisOutputs);

//        MultilayerPerceptron p = new MultilayerPerceptron(4, 2, 3, new int[]{4, 3});
        MultilayerPerceptron p = new MultilayerPerceptron(4, 4, 3, new int[]{4, 4, 5, 3});
//        GraphicsWindow graphicsWindow = new GraphicsWindow("Multilayer Perceptron Output", 800, 1200, p);
        p.initializePerceptron();
        runIrisDataset(p);

        //Load a graphical window to represent the perceptron
//        GraphicsWindow graphicsWindow = new GraphicsWindow("Multilayer Perceptron Output", 800, 1200, p);

//        MultilayerPerceptron p = FileIO.loadMultilayerPerceptron("testing_perceptron_updated.txt");
//        runIrisDataset(p);

//        MultilayerPerceptron p = FileIO.loadMultilayerPerceptron("testing_perceptron_updated01.txt");
//        runIrisDataset(p);

        System.out.println("hi");
    }
}
