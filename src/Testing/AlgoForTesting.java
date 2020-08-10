package Testing;

import metalearnerNN.NeuralNetwork;

import java.io.IOException;

public class AlgoForTesting {

    NeuralNetwork network;

    public AlgoForTesting() {
        network = new NeuralNetwork();
    }

    public void load(int num) throws IOException, ClassNotFoundException {
        network = network.load("src/Testing/TestData/TestNetwork" + (num + 1) + ".txt");
    }

    public double[][] processImage(double[][] inputData) {

        return network.predict(inputData);
    }


}
