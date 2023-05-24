import javax.swing.*;
import java.awt.*;

public class GraphicsWindow {

    JPanel panel;
    JFrame frame;

    public GraphicsWindow(String windowName, int height, int width, MultilayerPerceptron mlp) {
        panel = new GraphicsMultilayerPerceptron(mlp);
        panel.setBackground(Color.DARK_GRAY.darker());
        frame = new JFrame(windowName);
        frame.setSize(height, width);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(panel, BorderLayout.CENTER);
        frame.add(panel);
        frame.setVisible(true);
    }

    public void newWindow() {

    }
}
