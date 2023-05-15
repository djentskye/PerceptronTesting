import java.io.*;
import java.util.Scanner;

public class FileIO {

    public static void saveMultilayerPerceptron(MultilayerPerceptron p, String filepath) {
        try {
            FileWriter fileWriter = new FileWriter(filepath);
            PrintWriter printWriter = new PrintWriter(fileWriter);

            //Write the number of nodes
            printWriter.print("input=");
            printWriter.print(p.getInput());
            printWriter.print("\n");
            printWriter.print("hidden=");
            printWriter.print(p.getHidden());
            printWriter.print("\n");
            printWriter.print("output=");
            printWriter.print(p.getOutput());
            printWriter.print("\n");

            //Write the weights
            printWriter.print("weights=");
            double[][][] weights = p.getWeights();
            for(int i = 0; i < weights.length; i++) {
                for(int j = 0; j < weights[i].length; j++) {
                    for(int k = 0; k < weights[i][j].length; k++) {
                        printWriter.print(weights[i][j][k]);
                        printWriter.print(",");
                    }
                    printWriter.print(";");
                }
                printWriter.print("\n");
            }

            printWriter.print("\n");

//            printWriter.print("weightdeltas=");
//            printWriter.print(p.getWeightDeltas());
//            printWriter.print("\n");

            printWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static MultilayerPerceptron loadMultilayerPerceptron(String filepath) {
        MultilayerPerceptron p = null;

        int input = 0, hidden = 0, output = 0;

        double[][][] weights;

        try {
            Scanner scanner = new Scanner(new File(filepath));
            scanner.useDelimiter(",|;|=|\\n");

            while(scanner.hasNext()) {
                String s = scanner.next(); //Parse the next section (i.e. input, hidden, weights, etc)

                if(s.equals("input")) {
                    input = Integer.parseInt(scanner.next());
                } else if(s.equals("hidden")) {
                    hidden = Integer.parseInt(scanner.next());
                } else if(s.equals("output")) {
                    output = Integer.parseInt(scanner.next());
                } else if(s.equals("weights")) {
                    weights = new double[2][][]; //Number of layers of weights has to be hardcoded right now ):
                    weights[0] = new double[hidden][input];
                    weights[1] = new double[output][hidden];

                    for(int i = 0; i < weights.length; i++) {
                        for(int j = 0; j < weights[i].length; j++) {
                            for(int k = 0; k < weights[i][j].length; k++) {
                                weights[i][j][k] = Double.parseDouble(scanner.next());
                            }
                            scanner.next();
                        }
                        scanner.next();
                    }
                    p = new MultilayerPerceptron(input, hidden, output, weights);
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return p;
    }
}
