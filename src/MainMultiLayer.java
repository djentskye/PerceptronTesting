import java.util.Random;

public class MainMultiLayer {

    public static double[][] irisInputs = new double[150][4];
    public static double[][] irisOutputs = new double[150][3];

    public static double[][] irisAmyInputs = new double[3][4];
    public static double[][] irisAmyOutputs = new double[3][3];

    public static void runEpoch(MultilayerPerceptron p, double learningRate) {
        p.changeLearningRate(learningRate);

        Random rand = new Random();

        //TODO: Gradient vector goes here! ^^
        for(int i = 0; i < 10; i++) {
            int j = rand.nextInt(150);
            p.presentPattern(irisInputs[j]);
            p.backprop(irisOutputs[j]);
        }
    }

    public static void runIrisDataset(MultilayerPerceptron p) {
        p.showState();
        p.fitness(irisInputs, irisOutputs);
        p.printErrorMatrix(irisInputs, irisOutputs);

        p.showState();
        runEpoch(p, 0.3);
        p.fitness(irisInputs, irisOutputs);
        p.printErrorMatrix(irisInputs, irisOutputs);

        for(int i = 0; i < 500; i++) {
            runEpoch(p, 0.3);
        }
        p.showState();

        p.fitness(irisInputs, irisOutputs);
        p.printErrorMatrix(irisInputs, irisOutputs);

//        FileIO.saveMultilayerPerceptron(p, "testing_perceptron_updated.txt");
    }

    public static void main(String[] args) {
        ReadCSV.readIrisDataset("iris.data", irisInputs, irisOutputs);

        MultilayerPerceptron p = new MultilayerPerceptron(4, 2, 3, new int[]{4, 3}); //TODO: Breaks with {3, 4}
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
