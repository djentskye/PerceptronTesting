import javax.swing.*;
import java.awt.*;

public class GraphicsMultilayerPerceptron extends JPanel {

    MultilayerPerceptron mlp;

    public GraphicsMultilayerPerceptron(MultilayerPerceptron multilayerPerceptron) {
        this.mlp = multilayerPerceptron;
    }

    public int[][][] findNodeCoords() {
        int inputs = mlp.getInput();
        int hiddenLayers = mlp.getHiddenLayers();
        int[] hiddenNodes = mlp.getHiddenNodes();
        int outputs = mlp.getOutput();

        //layer, node number, coordinate x/y
        int[][][] funcOut = new int[1 + hiddenLayers + 1][][];

        int funcOutLength = funcOut.length;

        //Initialize the array sizes
        for(int i = 0; i < funcOutLength; i++) {
            //Input nodes
            if(i == 0) {
                funcOut[i] = new int[inputs][2];
            } else if(i == funcOutLength-1) {
                funcOut[i] = new int[outputs][2];
            } else {
                funcOut[i] = new int[hiddenNodes[i - 1]][2];
            }
        }

        //Calculate the coordinates for each node
        //Inputs
        for(int i = 0; i < inputs; i++) {
            funcOut[0][i][0] = 50;
            funcOut[0][i][1] = 50 + i*50;
        }

        //Hidden
        for(int i = 0; i < hiddenLayers; i++) {
            for(int j = 0; j < hiddenNodes[i]; j++) {
                funcOut[i+1][j][0] = 160 + 110*i /*50 + 60*/;
                funcOut[i+1][j][1] = 50 + j * 50;
            }
        }

        //Outputs
        for(int i = 0; i < outputs; i++) {
            funcOut[funcOutLength-1][i][0] = 160 + 110*hiddenLayers;
            funcOut[funcOutLength-1][i][1] = 50 + i*50;
        }

        return funcOut;
    }

    public void drawNodes(Graphics graphics, int[][][] coords) {
        graphics.setColor(Color.LIGHT_GRAY);
        //Draw each input node in the multilayer perceptron
        int inputs = mlp.getInput();
        for(int i = 0; i < inputs; i++) {
//            graphics.drawRect(50, 50 + i*50, 10, 10);
            graphics.drawRect(coords[0][i][0], coords[0][i][1], 10, 10);
        }

        //Draw each hidden node
        int hiddenLayers = mlp.getHiddenLayers();
        int[] hiddenNodes = mlp.getHiddenNodes();
        for(int i = 0; i < hiddenLayers; i++) {
            for(int j = 0; j < hiddenNodes[i]; j++) {
//                graphics.drawRect(160 + 110*i /*50 + 60*/, 50 + j * 50, 10, 10);
                graphics.drawRect(coords[i+1][j][0], coords[i+1][j][1], 10, 10);
            }
        }

        //Draw each output node
        int outputs = mlp.getOutput();
        for(int i = 0; i < outputs; i++) {
//            graphics.drawRect(160 + 110*hiddenLayers, 50 + i*50, 10, 10);
            graphics.drawRect(coords[coords.length-1][i][0], coords[coords.length-1][i][1], 10, 10);
        }
    }

    public void paintComponent(Graphics graphics) {
        super.paintComponent(graphics);

        //Do math to figure out how big it should be ???

        //100 pixel buffer between layers?

        int[][][] coords = findNodeCoords();

        drawNodes(graphics, coords);
    }


}
