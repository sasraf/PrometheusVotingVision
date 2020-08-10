public interface Algorithm {

    // Loads any saved training data into the current instance of the algorithm
    public void load(String filePath);

    // Returns output values of the algorithm given the local path to an image
    public double[] processImage(String imagePath);
}
