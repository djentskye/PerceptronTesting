public class MainSingleLayer {

    public static double[][] andGateTwoInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    public static double[][] andGateTwoOutputs = {{0}, {0}, {0}, {1}};
    public static double[][] andGateThreeInputs = {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {1, 1, 1},
                                                   {0, 1, 0}, {1, 1, 0}, {1, 0, 0}, {1, 0, 1}};
    public static double[][] andGateThreeOutputs = {{0}, {0}, {0}, {1}, {0}, {0}, {0}, {0}};

    public static double[][] irisInputs = new double[150][4];
    public static double[][] irisOutputs = new double[150][3];

    public static void main(String[] args) {
        ReadFile.readIrisDataset("iris.data", irisInputs, irisOutputs);

        Perceptron p = new Perceptron(4, 0, 3);
        p.initializePerceptron();
        p.showState();
        p.fitness(irisInputs, irisOutputs);
        System.out.println("FIRST RUN:");
        for(int i = 0; i < irisInputs.length; i++) {
            p.presentPattern(irisInputs[i]);
            p.backprop(irisOutputs[i]);
        }
        p.showState();
        System.out.println("Changing learning rate to 0.2...");
        p.changeLearningRate(0.2);
        System.out.println("SECOND RUN:");
        for(int i = 0; i < irisInputs.length; i++) {
            p.presentPattern(irisInputs[i]);
            p.backprop(irisOutputs[i]);
        }
        p.showState();
        p.fitness(irisInputs, irisOutputs);
    }

    public static void runAndGateTwo() {
        Perceptron p = new Perceptron(2, 0, 1);
        p.initializePerceptron();
        p.showState();
        for(int i = 0; i < andGateTwoInputs.length; i++) {
            System.out.println(p.presentPattern(andGateTwoInputs[i])[0]);
            p.backprop(andGateTwoOutputs[i]);
        }
        p.changeLearningRate(0.3);
        p.showState();
    }

    public static void runAndGateThree() {
        Perceptron p = new Perceptron(3, 0, 1);
        p.initializePerceptron();
        p.showState();
        for(int i = 0; i < andGateThreeInputs.length; i++) {
            System.out.println(p.presentPattern(andGateThreeInputs[i])[0]);
            p.backprop(andGateThreeOutputs[i]);
        }
        p.changeLearningRate(0.3);
        p.showState();
        for(int i = 0; i < andGateThreeInputs.length; i++) {
            System.out.println(p.presentPattern(andGateThreeInputs[i])[0]);
        }
    }
}
