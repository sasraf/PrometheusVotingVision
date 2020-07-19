package metalearnerNN.Activations;

public class softmaxActivationFunction implements ActivationFunction {

    private double esum;

    public double[] activation(double[] x) {
        //Get sum of each number taken to the power of e
        for (int i = 0; i < x.length; i++) {
            esum += Math.exp(x[i]);
        }

        // Run softmax function on each input in x
        for (int i = 0; i < x.length; i++) {
            x[i] = Math.exp(x[i]) / esum;
        }

        return x;
    }

    public double[] activationDerivative(double[] x) {

        double[] softmax = activation(x);

        double[] output = new double[x.length];
        for(int neuron = 0; neuron < output.length; neuron++) {
            output[neuron] = softmax[neuron] * (1d - softmax[neuron]);
        }

        return output;
        
    }

    private double kroneckerDelta(double i, double j) {
        if (i != j) {
            return 0;
        }
        else {
            return 1;
        }
    }
}
