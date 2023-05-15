public class Main {

    public static double[][] andGateTwoInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    public static double[][] andGateTwoOutputs = {{0}, {0}, {0}, {1}};
    public static double[][] andGateThreeInputs = {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {1, 1, 1},
                                                   {0, 1, 0}, {1, 1, 0}, {1, 0, 0}, {1, 0, 1}};
    public static double[][] andGateThreeOutputs = {{0}, {0}, {0}, {1}, {0}, {0}, {0}, {0}};

    public static void main(String[] args) {
//        Perceptron p = new Perceptron(2, 0, 1);
//        p.initializePerceptron();
//        p.showState();
//        for(int i = 0; i < andGateTwoInputs.length; i++) {
//            System.out.println(p.presentPattern(andGateTwoInputs[i])[0]);
//            p.backprop(andGateTwoOutputs[i]);
//        }
//        p.changeLearningRate(0.3);
//        p.showState();


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
