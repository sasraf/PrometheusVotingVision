package metalearnerNN;

import metalearnerNN.Activations.ActivationFunction;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;

public class ActivationLayer implements Layer, Serializable {

    private ActivationFunction activationFunction;
    private double[] input;
    private double[] output;

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
    public double[] backProp(double[] outputError, double learningRate, double[] expected) {

        // Deals with softmax
        if (activationFunction.getClass().getSimpleName().equals("softmaxActivationFunction")) {
            return activationFunction.dEdX(input, expected);
        }

        return activationFunction.dEdX(input, outputError);
    }
}
