import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class ReadFile {

    /**
     * Reads a CSV of the iris dataset, separated with commas and lines separated with newlines
     *
     * @param filepath
     */
    public static void readIrisDataset(String filepath, double[][] irisDatasetIn, double[][] irisDatasetOut) {
        try {
            Scanner scanner = new Scanner(new File(filepath));
            scanner.useDelimiter(",|\\n");
            int currentLine = 0;
            while(scanner.hasNext()) {
                irisDatasetIn[currentLine][0] = Double.parseDouble(scanner.next());
                irisDatasetIn[currentLine][1] = Double.parseDouble(scanner.next());
                irisDatasetIn[currentLine][2] = Double.parseDouble(scanner.next());
                irisDatasetIn[currentLine][3] = Double.parseDouble(scanner.next());

                String name = scanner.next();
                if(name.equals("Iris-setosa")) {
                    irisDatasetOut[currentLine] = new double[]{1, 0, 0};
                }
                if(name.equals("Iris-versicolor")) {
                    irisDatasetOut[currentLine] = new double[]{0, 1, 0};
                }
                if(name.equals("Iris-virginica")) {
                    irisDatasetOut[currentLine] = new double[]{0, 0, 1};
                }
                currentLine++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
