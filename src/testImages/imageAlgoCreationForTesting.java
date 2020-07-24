package testImages;

import metalearnerNN.ActivationLayer;
import metalearnerNN.Activations.tanhActivationFunction;
import metalearnerNN.FullyConnectedLayer;
import metalearnerNN.Loss.MeanSquaredErrorFunction;
import metalearnerNN.NeuralNetwork;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Arrays;

// Trains imageRecognition algos for testin on MNIST
public class imageAlgoCreationForTesting {

    private static final int epochs = 10;
    private static final double learningRate = .1;
    private static final String learningDataPath = "src/testImages/mnist_train";

    public static void main(String[] args) throws IOException, ClassNotFoundException {

        // Takes in serialized array
        double[][] mnist;
        FileInputStream readFile = new FileInputStream(learningDataPath);
        ObjectInputStream in = new ObjectInputStream(readFile);
        mnist = (double[][]) in.readObject();
        in.close();
        readFile.close();

        // Saves serialized arrays to inputData and expectedOutput
        double[][] inputData = new double[mnist.length][mnist[0].length - 1];
        double[][] expectedOutput = new double[mnist.length][10];

        for (int i = 0; i < mnist.length; i++) {
            expectedOutput[i] = oneHotEncodeMnist(mnist[i][0]);
            for (int n = 1; n < mnist[0].length; n++) {
                inputData[i][n - 1] = mnist[i][n];
            }
        }

        // Setting up networks with 3 layers & tanh activation functions that are saved as TestNetworki.txt where i = an int
        long totalMS = 0;
        for (int i = 0; i < 4; i++) {
            NeuralNetwork network = new NeuralNetwork();
            network.addLayer(new FullyConnectedLayer(784, 20));
            network.addLayer(new ActivationLayer(new tanhActivationFunction()));
            network.addLayer(new FullyConnectedLayer(20, 10));
            network.addLayer(new ActivationLayer(new tanhActivationFunction()));
            network.setLoss(new MeanSquaredErrorFunction());

            System.out.println("Training Network #" + (i + 1));

            // Training and keeping track of time it takes to train
            long startTime = System.nanoTime();
            network.train(inputData, expectedOutput, epochs, learningRate);
            long endTime = System.nanoTime();
            long trainingTime = endTime - startTime;
            totalMS += trainingTime / 1000000;
            System.out.println("Network #" + (i + 1) + " trained in " + trainingTime / 1000000 + "ms");

            System.out.println("Prediction: " + Arrays.deepToString(network.predict(new double[][]{inputData[50]})) + "Actual: " + Arrays.toString(expectedOutput[50]) + "\n\n");

            network.save("src/testImages/TestNetwork" + (i + 1) + ".txt");
        }
    }

    // Puts outputs in onehotencoding format
    private static double[] oneHotEncodeMnist(double num) {
        double[] oneHotEncoded = new double[10];
        for (int i = 0; i < oneHotEncoded.length; i++) {
            oneHotEncoded[i] = 0;
        }
        oneHotEncoded[(int) num] = 1;

        return oneHotEncoded;
    }
}
