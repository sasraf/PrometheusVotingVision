package metalearnerNN.TestNetworks;

import metalearnerNN.ActivationLayer;
import metalearnerNN.Activations.softmaxActivationFunction;
import metalearnerNN.Activations.tanhActivationFunction;
import metalearnerNN.FullyConnectedLayer;
import metalearnerNN.Loss.MeanSquaredErrorFunction;
import metalearnerNN.NeuralNetwork;

import java.util.Arrays;

public class NNTest {
    public static void main(String[] args) {

        // Sample training data
        double[][] inputData = new double[][] {{0}, {1}};
        double[][] expectedOutput = new double[][] {{0, 1}, {1, 0}};

        // Setting up network with 3 layers & tanh activation functions
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(new FullyConnectedLayer(1, 2));
        network.addLayer(new ActivationLayer(new softmaxActivationFunction()));

        network.setLoss(new MeanSquaredErrorFunction());

        // Training and keeping track of time it takes to train
        long startTime = System.nanoTime();
        network.train(inputData, expectedOutput, 500, .1);
        long endTime = System.nanoTime();

        long trainingTime = endTime - startTime;

        System.out.println("\n\nTraining time: " + trainingTime + "nanoseconds\n\n");

        // Test network
        double[][] output = network.predict(inputData);

        for (int i = 0; i < inputData.length; i++) {
            System.out.println("For set " + Arrays.toString(inputData[i]) + " my prediction is " + Arrays.toString(output[i]) + " while the correct value is " + Arrays.toString(expectedOutput[i]));
        }

        System.out.println("\n\nTraining time: " + trainingTime + "nanoseconds\n\n");
    }
}
