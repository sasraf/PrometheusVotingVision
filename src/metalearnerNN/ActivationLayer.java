package metalearnerNN;

public class ActivationLayer implements Layer {

    private ActivationFunction activationFunction;
    private double[] input;
    private double[] output;

    public ActivationLayer(ActivationFunction passedActivationFunction) {
        activationFunction = passedActivationFunction;
    }

    public double[] feedForward(double[] layerInput) {
        input = layerInput;
        output = activationFunction.activation(input);
        return output;
    }

    public double[] backProp(double[] outputError, double learningRate) {
        return Matrix.matrixMultiply(input, outputError);
    }
}
