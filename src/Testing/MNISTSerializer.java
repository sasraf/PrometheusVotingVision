package Testing;

import java.io.*;
import java.util.ArrayList;

// Reads in MNIST CSV, saves as a serialized Array for use in other programs
// Meant to work with https://pjreddie.com/projects/mnist-in-csv/
public class MNISTSerializer {

    private static final int numOfPixelsInImage = 785;

    // Path to file being read, path to file where the serialized array is saved
    private static final String path = "src/Testing/TestData/mnist_test.csv"; //mnist_train.csv";
    private static final String saveFile = "src/Testing/TestData/mnist_test"; //mnist_train";

    public static void main(String[] args) throws IOException, ClassNotFoundException {

        ArrayList<Integer> MNIST = new ArrayList<Integer>();
        BufferedReader reader = new BufferedReader(new FileReader(path));
        String line = reader.readLine();

        // Reads MNIST CSV
        while (line != null) {
            String[] nums = line.split(",");
            for (int i = 0; i < nums.length; i++) {
                MNIST.add(Integer.parseInt(nums[i]));
            }
            line = reader.readLine();
        }

        if (MNIST.size() % numOfPixelsInImage != 0) {
            throw new IllegalArgumentException("numOfIntsInImage is incorrect. Remainder of " + MNIST.size() % numOfPixelsInImage);
        }

        // Converts arraylist to array
        double[][] arrayToSave = new double[MNIST.size() / numOfPixelsInImage][numOfPixelsInImage];
        int imageCounter = -1;
        for (int i = 0; i < MNIST.size(); i++) {
            if (i % numOfPixelsInImage == 0) {
                imageCounter++;
            }
            arrayToSave[imageCounter][i % numOfPixelsInImage] = MNIST.get(i);
        }

        // Saves array to file
        FileOutputStream writeFile = new FileOutputStream(saveFile);
        ObjectOutputStream out = new ObjectOutputStream(writeFile);
        out.writeObject(arrayToSave);
        out.close();
        writeFile.close();
    }
}
