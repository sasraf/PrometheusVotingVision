package metalearnerNN.Activations;

import java.io.Serializable;


public class softmaxActivationFunction implements ActivationFunction, Serializable {

    private double esum;

    public double[] activation(double[] x) {
        //Get sum of each number taken to the power of e
        esum = 0;
        for (int i = 0; i < x.length; i++) {

            double exp = Math.exp(x[i]);

            // Deals with if Math.exp(x[i]) is infinity
            exp = Math.min(exp, Double.MAX_VALUE);
            esum += exp;
        }

        // Run softmax function on each input in x
        double[] c = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double exp = Math.exp(x[i]);

            // Deals with if Math.exp(x[i]) is infinity
            exp = Math.min(exp, Double.MAX_VALUE);
            c[i] = exp / esum;
        }
        return c;
    }

    // Formula from https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
    public double[] dEdX(double[] input, double[] output) {
        double tSum = 0;
        for (int i = 0; i < output.length; i++) {
            tSum += output[i];
        }

        double[] o = activation(input);

        double[] c = new double[input.length];

        for (int i = 0; i < o.length; i++) {
            c[i] = o[i] * tSum - output[i];
        }

        return c;
    }
}
