package metalearnerNN;

import java.util.ArrayList;
import java.util.function.Function;

public class NeuralNetwork {

    private ArrayList<Layer> layers;
    private Loss loss;

    public NeuralNetwork() {
        layers = new ArrayList<Layer>();
        loss = null;
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

        // Create an arraylist to hold outputs
        ArrayList<double[]> results = new ArrayList<double[]>();
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
            results.add(output);
        }

        //TODO: maybe keep everything as an arraylist?

        // Converts arraylist back to an array (toArray creates an arrayList of Double, not double)
        double[][] outputLayer = new double[results.size()][results.get(0).length];
        for (int n = 0; n < outputLayer.length; n++) {
            outputLayer[n] = results.get(n);
        }

        return outputLayer;

    }

    public void train(double[][] inputs, double[][] expected, int epochs, double learningRate) {

        for (int epoch = 0; epoch < epochs; epoch++) {
            double displayError = 0;
            for (int input = 0; input < inputs.length; input++) {
                double[] output = inputs[input];

                // Feed through neural network
                for (Layer layer : layers) {
                    output = layer.feedForward(output);
                }

                // Calculate loss
                //TODO: create a loss interface??
                displayError += loss.function(expected[input], output);

                // Backprop
                double[] error = loss.derivative(expected[input], output);
                for (int layer = layers.size() - 1; layer >= 0; layer--) {

                    // backProp() adjusts the weights/biases of current layer, returns inputError for backprop of previous layer
                    error = layers.get(layer).backProp(error, learningRate);
                }

                // Calculates avg error, displays error/epoch
                displayError /= inputs.length;
                System.out.println("Epoch " + epoch + " with error " + displayError);

            }
        }

    }


}
