package metalearnerNN;

import metalearnerNN.Activations.ActivationFunction;

import java.io.FileWriter;
import java.io.IOException;

public class ActivationLayer implements Layer {

    private ActivationFunction activationFunction;
    private double[] input;
    private double[] output;

    public void save(FileWriter fileWriter) throws IOException {
        fileWriter.write("\nACTIVATION: " + activationFunction.getClass().getSimpleName());
    }

    // Stores activaiton function for use on initialization
    public ActivationLayer(ActivationFunction passedActivationFunction) {
        activationFunction = passedActivationFunction;
    }

    // Feedforward on activationlayer
    public double[] feedForward(double[] layerInput) {
        input = layerInput;
        output = activationFunction.activation(input);
        return output;
    }

    // Runs backprop on activationlayer
    public double[] backProp(double[] outputError, double learningRate) {
        return Matrix.matrixMultiply(activationFunction.activationDerivative(input), outputError);
    }
}
