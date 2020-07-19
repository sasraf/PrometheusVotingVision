package metalearnerNN.Loss;

public interface Loss {

    public double function(double[] actual, double[] expected);

    public double[] derivative(double[] actual, double[] expected);
}
