import java.util.Random;

public class MainMultiLayer {

    public static double[][] irisInputs = new double[150][4];
    public static double[][] irisOutputs = new double[150][3];

    public static void runEpoch(MultilayerPerceptron p, double learningRate) {
        p.changeLearningRate(learningRate);

        Random rand = new Random();

        for(int i = 0; i < irisInputs.length; i++) {
            int j = rand.nextInt(150);
            p.presentPattern(irisInputs[j]);
            p.backprop(irisOutputs[j]);
        }
    }

    public static void main(String[] args) {
        ReadFile.readIrisDataset("iris.data", irisInputs, irisOutputs);

        MultilayerPerceptron p = new MultilayerPerceptron(4, 4, 3);
        p.initializePerceptron();
        p.showState();
        p.fitness(irisInputs, irisOutputs);
//        System.out.println("FIRST RUN:");

        p.showState();
        runEpoch(p, 20);
        p.fitness(irisInputs, irisOutputs);

        for(int i = 0; i < 25; i++) {
            runEpoch(p, 10/(i+1));
            p.showState();
            p.fitness(irisInputs, irisOutputs);
        }
        p.showState();

        p.fitness(irisInputs, irisOutputs);
    }
}
