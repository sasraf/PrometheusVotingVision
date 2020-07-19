package metalearnerNN;

import metalearnerNN.Loss.Loss;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class NeuralNetwork {

    private ArrayList<Layer> layers;
    private Loss loss;

    public NeuralNetwork() {
        layers = new ArrayList<Layer>();
        loss = null;
    }

    public void save(String dirPath) throws IOException {

        File saveFile = new File(dirPath);

        // If file already exists, ask user if they want to clear it
        if (saveFile.exists()) {
            System.out.println("File " + dirPath + " already exits. Would you like to overwrite? (Y/N)");
            Scanner scanner = new Scanner(System.in);
            String input = scanner.nextLine();
            input = input.toLowerCase();

            if (input.equals("n") || input.equals("no")) {
                throw new IllegalArgumentException("NN was not saved");
            }
            else if (!input.equals("y") || input.equals("yes")) {
                throw new IllegalArgumentException("User response was not understood. Requested (Y/N) but received: " + input);
            }
        }

        // Append = false: overwrites
        FileWriter fileWriter = new FileWriter(saveFile,false);

        //First write to file the specifications of the NN
        //Write NN as header then state the number of layers
        fileWriter.write("NN\nLAYERS:" + layers.size() + "\n");

        // State what type of layer each layer is
        for (int i = 0; i < layers.size(); i++) {
            fileWriter.write(layers.get(i).getClass().getSimpleName() + ",");
        }


        // Now save weights/biases of each layer
        for (int i = 0; i < layers.size(); i++) {
            layers.get(i).save(fileWriter);
        }

        // Now save loss function
        fileWriter.write("\nLOSS: " + loss.getClass().getSimpleName());

        fileWriter.close();
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
                    error = layers.get(layer).backProp(error, learningRate);
                }
            }

            // Calculates avg error, displays error/epoch
            displayError /= inputs.length;
            System.out.println("Epoch " + epoch + " with error " + displayError);
        }

        System.out.println();

    }
}
