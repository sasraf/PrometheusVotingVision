import metalearnerNN.*;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

public class AlgoForTesting {

    NeuralNetwork network;

    public AlgoForTesting() {
        network = new NeuralNetwork();
    }

    public void load(int num) throws IOException, ClassNotFoundException {
        network = network.load("src/TestNetwork" + (num + 1) + ".txt");
    }

    public double[][] processImage(double[][] inputData) throws IOException, ClassNotFoundException {

        return network.predict(inputData);
    }


}
