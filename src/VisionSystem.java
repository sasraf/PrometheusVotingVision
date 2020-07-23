import java.util.ArrayList;
import java.util.Scanner;
import java.io.File;

public class VisionSystem {

    public static void main(String[] args) {

        ModelStack modelStack = new ModelStack();
        MetaLearner metaLearner = new MetaLearner();

        Scanner scanner = new Scanner(System.in);

        String input = "";

        while (!input.equals("q")) {

            System.out.println("Please select what you would like to do or press 'q' to quit:");
            System.out.println("A.) Test Models\nB.) Train Models\nC.) Test ModelStack");

            input = scanner.nextLine();

            if (input.equals("a") || input.equals("A")) {

                String dirPath = "testImages";

                // Set a dir containing images to pass through the visionsystem
                System.out.println("Enter the name of your folder containing images or press enter to use the default \"testImages\" folder: ");
                input = scanner.nextLine();
                if (!input.equals("")) {
                   dirPath = input;
                }

                // Creates an arraylist of the paths of each image in the given directory
                ArrayList<String> imagePaths = listImagesInDir(dirPath);

                // For each image, outputs of all algorithms into processedImage, collect those inputs into a metaLearnerInput double array to be passed to the metaLearner
                double[] processedImage = modelStack.processImage(imagePaths.get(0));
                double[][] metaLearnerInputs = new double[imagePaths.size()][processedImage.length];
                metaLearnerInputs[0] = processedImage;
                for (int i = 1; i < imagePaths.size(); i++) {
                    processedImage = modelStack.processImage(imagePaths.get(i));
                    metaLearnerInputs[i] = processedImage;
                }

                // Pass inputs to metaLearner
                double[][] metaOutputs = metaLearner.feedForward(metaLearnerInputs);

                //TODO: print out outputs


            }
            else if (input.equals("b") || input.equals("B")) {
                //TODO: train models
            }
            // Print processImage output
            else if (input.equals("c") || input.equals("C")) {
                double[] stackOutput = modelStack.processImage("test");
                System.out.println();

                for (int i = 0; i < stackOutput.length; i++) {
                    System.out.print(stackOutput[i] + ", ");
                }
                System.out.println();
            }
            else {
                System.out.println("Pick an action from one of the below choices:");
            }
        }
    }

    // Given a folder with images, creates an arraylist of the images' filePaths
    private static ArrayList<String> listImagesInDir(String filePath) {

        ArrayList<String> imagePaths = new ArrayList<String>();

        // Check that given path is a folder
        File thisdir = new File(filePath);
        if (!thisdir.isDirectory()) {
            throw  new IllegalArgumentException("Image filePath is not a directory");
        }

        File[] files = new File(filePath).listFiles();

        // Checks that there are more than 0 images in folder
        if (files.length == 0) {
            throw new IllegalArgumentException("There are no files in the image filePath");
        }

        // Adds image paths to arraylist
        for (File file : files) {
            imagePaths.add(filePath + file.getName());
        }

        return imagePaths;
    }
}
