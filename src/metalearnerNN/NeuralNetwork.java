package metalearnerNN;

import metalearnerNN.Loss.Loss;
import metalearnerNN.Loss.MeanSquaredErrorFunction;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class NeuralNetwork implements Serializable {

    private ArrayList<Layer> layers;
    private Loss loss;

    public NeuralNetwork() {
        layers = new ArrayList<Layer>();
        loss = null;
    }

    // Serialize
    public void save(String path) throws IOException {
        FileOutputStream writeFile = new FileOutputStream(path);
        ObjectOutputStream out = new ObjectOutputStream(writeFile);

        out.writeObject(this);

        out.close();
        writeFile.close();
    }

    // Deserialize
    public NeuralNetwork load(String path) throws IOException, ClassNotFoundException {
        FileInputStream readFile = new FileInputStream(path);
        ObjectInputStream in = new ObjectInputStream(readFile);

        NeuralNetwork nn = (NeuralNetwork)in.readObject();

        in.close();
        readFile.close();

        return nn;
    }

    // Adds layer
    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    // Sets loss function
    public void setLoss(Loss inputLoss) {
        loss = inputLoss;
    }

    // Returns a prediction for a given set of data
    public double[][] predict(double[][] inputData) {

        // Create an array to hold outputs
        double[][] results = new double[inputData.length][inputData[0].length];
        double[] output;

        // For each input
        for (int i = 0; i < inputData.length; i++) {

            output = inputData[i];

            // For each layer of the neural network
            for (Layer curLayer : layers) {

                // Feed the output of the previous layer into the next layer until it's been passed through all layers
                output = curLayer.feedForward(output);
            }

            // Add output to results
            results[i] = output;
        }

        return results;
    }

    public void train(double[][] inputs, double[][] expected, int epochs, double learningRate) {

        for (int epoch = 0; epoch <= epochs; epoch++) {
            double displayError = 0;
            for (int input = 0; input < inputs.length; input++) {
                double[] output = inputs[input];
                // Feed through neural network
                for (Layer layer : layers) {
                    output = layer.feedForward(output);
                }

                // Calculate loss
                displayError += loss.function(expected[input], output);

                // Backprop
                double[] error = loss.derivative(expected[input], output);

                for (int layer = layers.size() - 1; layer >= 0; layer--) {

                    // backProp() adjusts the weights/biases of current layer, returns inputError for backprop of previous layer
                    error = layers.get(layer).backProp(error, learningRate, expected[input]);
                }

            }

            // Calculates avg error, displays error/epoch
            displayError /= inputs.length;
            System.out.println("Epoch " + epoch + " with error " + displayError);
        }

        System.out.println();

    }
}
