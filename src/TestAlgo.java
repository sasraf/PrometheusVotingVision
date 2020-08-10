public class TestAlgo implements Algorithm {

    double[] testOutput;

    public TestAlgo(double[] input) {
        testOutput = input;
    }

    public void load(String filePath) {
    }

    @Override
    public double[] processImage(String imagePath) {
        return testOutput;
    }
}
