package metalearnerNN.TestNetworks;

import metalearnerNN.ActivationLayer;
import metalearnerNN.Activations.reluActivationFunction;
import metalearnerNN.Activations.softmaxActivationFunction;
import metalearnerNN.FullyConnectedLayer;
import metalearnerNN.Loss.MeanSquaredErrorFunction;
import metalearnerNN.NeuralNetwork;

import java.io.IOException;
import java.util.Arrays;

public class XORTest {
    public static void main(String[] args) throws IOException, ClassNotFoundException {

        // Sample training data
        double[][] inputData = new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] expectedOutput = new double[][] {{0, 1}, {1, 0}, {1, 0}, {0, 1}};

        // Setting up network with 3 layers & tanh activation functions with a softmax output layer
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(new FullyConnectedLayer(2, 3));
        network.addLayer(new ActivationLayer(new reluActivationFunction()));
        network.addLayer(new FullyConnectedLayer(3, 2));
//        network.addLayer(new ActivationLayer(new tanhActivationFunction()));
        network.addLayer(new ActivationLayer(new softmaxActivationFunction()));


        network.setLoss(new MeanSquaredErrorFunction());

        // Training and keeping track of time it takes to train
        long startTime = System.nanoTime();
        network.train(inputData, expectedOutput, 500, .1);
        long endTime = System.nanoTime();

        long trainingTime = endTime - startTime;

        // Test saving/loading
//        network.save("bob.txt");
//        network = network.load("bob.txt");


        // Test network
        double[][] output = network.predict(inputData);

        for (int i = 0; i < inputData.length; i++) {
            System.out.println("For set " + Arrays.toString(inputData[i]) + " my prediction is " + Arrays.toString(output[i]) + " while the correct value is " + Arrays.toString(expectedOutput[i]));
        }

        System.out.println("\nTraining time: " + trainingTime / 1000000 + "ms");
    }
}
