import java.util.HashMap;
import java.util.Random;

public class MainMultiLayerGenerational {

    public static double[][] irisInputs = new double[150][4];
    public static double[][] irisOutputs = new double[150][3];

    public static void backprop40(MultilayerPerceptron p, double learningRate) {
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
        for(int i = 0; i < 40; i++) {
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

//    public static void runEpoch(MultilayerPerceptron p, double learningRate) {
//        int epochSize = 10;
//
//        //Make a bunch of modified versions of p
//
//        Random rand = new Random();
//
//
//        //TODO: Gradient vector goes here! ^^
//        for(int i = 0; i < epochSize; i++) {
//            int j = rand.nextInt(150);
//            p.presentPattern(irisInputs[j]);
//
//        }
//
//        //Make the corrections from the cumulative gradients
//        p.backprop_weights(gradientVector);
//    }

    /**
     * Modify the given model random by a specific amount.
     *
     * @param p
     * @param modificationRate
     * @return
     */
    public static MultilayerPerceptron modifyModelRandom(MultilayerPerceptron p, double modificationRate) {
        MultilayerPerceptron n = new MultilayerPerceptron(p.getInput(), p.getHiddenLayers(), p.getOutput(), p.getHiddenNodes());

        double[][][] weights = p.getWeights();
        double[][][] newWeights = n.getWeights(); //Clunky, but probably functional

        double[][][] weightDeltas = p.getWeightDeltas();
        double[][][] newWeightDeltas = n.getWeightDeltas();

        Random rand = new Random();

        //Double loop to go through each weight
        for(int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for(int k = 0; k < weights[i][j].length; k++) {
                    //TODO: check
                    newWeights[i][j][k] = weights[i][j][k] + ((rand.nextDouble() * 2*modificationRate) - modificationRate); //Set the weight
//                    newWeights[i][j][k] = weights[i][j][k] * ((rand.nextDouble() * 2*modificationRate) - modificationRate);
                }
            }
        }
        n.putWeights(newWeights);
        n.putWeightDeltas(weightDeltas);

        return n;
    }

    public static void main(String[] args) {
        int generations = 10000;

        //Read dataset
        ReadCSV.readIrisDataset("iris.data", irisInputs, irisOutputs);

        //Define 3 next-gen MLP's
        //Create 10 MLP's
        //Initialize MLPs
        HashMap<Integer, MultilayerPerceptron> mlps = new HashMap<>();
        for(int i = 0; i < 10; i++) {
            mlps.put(i, new MultilayerPerceptron(4, 2, 3, new int[]{4, 4}));

            //Initialize MLPs
            mlps.get(i).initializePerceptron();
        }

        //For loop, defined by # of generations desired
        for(int i = 0; i < generations; i++) {

            //For loop over each MLP
            for(int j = 0; j < 10; j++) {
                //Backpropogate each x100, low learning rate
                for(int k = 0; k < 10; k++) {
                    MainMultiLayer.runEpoch(mlps.get(j), 0.3, 10);
//                    MainMultiLayer.runEpoch(mlps.get(j), 2, 10);
                }
            }

            int[] top3 = {0, 1, 2};

            //Find the 3 MLP's with the best fitness/lowest error, select those for continuation
            for(int j = 3; j < 10; j++) {
                for(int k = 0; k < 3; k++) {
                    if(mlps.get(top3[k]).fitnessVal(irisInputs, irisOutputs) > mlps.get(j).fitnessVal(irisInputs, irisOutputs)) {
                        top3[k] = j;
                        break;
                    }
                }
            }

            //Replace the first 3 elements in the MLP map with the top 3 MLP's, wipe the rest
            MultilayerPerceptron temp1 = mlps.get(top3[0]);
            MultilayerPerceptron temp2 = mlps.get(top3[1]);
            MultilayerPerceptron temp3 = mlps.get(top3[2]);
//            MultilayerPerceptron temp4 = mlps.get(top3[3]);
            //mlps.clear();

            mlps.replace(0, temp1);
            mlps.replace(1, temp2);
            mlps.replace(2, temp3);
//            mlps.replace(3, temp4);

            //Print the performance of the top 3 upon the first iteration
            if(i == 0) {
                temp1.fitness(irisInputs, irisOutputs);
                temp2.fitness(irisInputs, irisOutputs);
                temp3.fitness(irisInputs, irisOutputs);
//                temp4.fitness(irisInputs, irisOutputs);
            }

            //Check in with a quick print statement every handful of generations (1k)
            if(i % 1000 == 0) {
                System.out.println("Gen " + i + " first lowest error: " + temp1.fitnessVal(irisInputs, irisOutputs));
            }

            //Re-write the other 7 MLP's with the 3 next-gen ones
            //Modify the 7 MLP's by a random amount (hopefully we can implement a better thing than random after this!)
            mlps.replace(3, modifyModelRandom(temp1, 0.005));
            mlps.replace(4, modifyModelRandom(temp2, 0.005));
            mlps.replace(5, modifyModelRandom(temp3, 0.005));
            mlps.replace(6, modifyModelRandom(temp1, 0.05));
            mlps.replace(7, modifyModelRandom(temp2, 0.05));
            mlps.replace(8, modifyModelRandom(temp3, 0.05));
            mlps.replace(9, modifyModelRandom(temp1, 0.5));
        }

        //Print the performance of the top 3
        mlps.get(0).fitness(irisInputs, irisOutputs);
        mlps.get(1).fitness(irisInputs, irisOutputs);
        mlps.get(2).fitness(irisInputs, irisOutputs);
    }
}

