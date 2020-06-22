public class TestAlgo implements Algorithm {

    double[] testOutput;

    public TestAlgo(double[] input) {
        testOutput = input;
    }

    public void load() {
    }

    @Override
    public double[] processImage(String imagePath) {
        return testOutput;
    }
}
