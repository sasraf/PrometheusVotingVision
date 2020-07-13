package metalearnerNN;

import java.lang.Math;
import java.util.Arrays;

public class Matrix {

    // Returns mean of
    public static double mean(double[] input) {
        double sum = 0;
        for (int i = 0; i < input.length; i++)
        {
            sum += input[i];
        }

        return sum / input.length;
    }

    // Takes each element of the array to the power of the inputted int
    public static double[] constantPower(double[] input, int power) {
        double[] newMatrix = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            newMatrix[i] = Math.pow(input[i], power);
        }

        return newMatrix;
    }

    // Subtracts one matrix from another
    public static double[] matrixSubtract(double[] minuend, double[] subtrahend) {
        if (minuend.length != subtrahend.length) {
            throw new IllegalArgumentException("matrixSubtract failed due to the matrices being different sizes");
        }

        double[] c = new double[minuend.length];

        for (int i = 0; i < minuend.length; i++) {
            c[i] = minuend[i] - subtrahend[i];
        }

        return  c;
    }

    // Matrix subtraction for 2d matrices
    public static  double[][] matrixSubtract(double[][] minuend, double[][] subtrahend) {
        if (minuend.length != subtrahend.length || minuend[0].length != subtrahend[0].length) {
            throw new IllegalArgumentException("matrixSubtract failed due to the matrices being different sizes");
        }

        double[][] output = new double[minuend.length][minuend[0].length];

        for (int i = 0; i < minuend.length; i++) {
            for (int n = 0; n < minuend[0].length; n++) {
                output[i][n] = minuend[i][n] - subtrahend[i][n];
            }
        }

        return output;
    }


    public static double[] matrixAdd(double[] a, double[] b) {

        if (a.length != b.length)
        {
            throw new IllegalArgumentException("matrixAdd failed due to the matrices being different sizes");
        }

        double[] c = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }

        return c;
    }

    // Multiplies two matrices
    // Code from https://introcs.cs.princeton.edu/java/22library/Matrix.java.html
    public static double[][] matrixMultiply(double[][] matrix1, double[][] matrix2) {
        // Stores lengths in appropriately labeled variables for ease of writing/reading
        int m1 = matrix1.length;
        int n1 = matrix1[0].length;
        int m2 = matrix2.length;
        int n2 = matrix2[0].length;

        double[][] output;
        // Scalar multiplication
        if (m1 == 1 && n1 == 1) {
            output= new double[m2][n2];

            for (int m = 0; m < m2; m++) {
                for (int n = 0; n < n2; n++) {
                    output[m][n] = matrix1[0][0] * matrix2[m][n];
                }
            }
        }
        // Deals with if matrix1 is a 1xm matrix and matrix2 is a nx1 matrix
        else if (matrix1.length == 1 && matrix2[0].length == 1) {
            output = new double[matrix2.length][matrix1[0].length];
            for (int i = 0; i < matrix2.length; i++) {
                for (int j = 0; j < matrix1[0].length; j++) {
                    output[i][j] = matrix2[i][0] * matrix1[0][j];
                }
            }
        }
        else {
            // ensures that matrices are multiplicable
            if (n1 != m2) {
                throw new IllegalArgumentException("Matrices Inputted into Matrix.matrixMultiply cannot be multiplied");
            }

            output = new double[m1][n2];

            for (int i = 0; i < m1; i++) {
                for (int j = 0; j < n2; j++) {
                    for (int k = 0; k < n1; k++) {
                        output[i][j] += matrix1[i][k] * matrix2[k][j];
                    }
                }
            }
        }

        return output;
    }

    // Matrix.dot operates the same way as numpy.dot() from numpy: https://www.javatpoint.com/numpy-dot#:~:text=%E2%86%92%20%E2%86%90%20prev-,numpy.,vectors%20(without%20complex%20conjugation).
    //TODO: fix naming conventions

    // For multiplying a 2d matrix with a 1d matrix
    // Code from https://introcs.cs.princeton.edu/java/22library/Matrix.java.html
//    public static double[] matrixMultiply(double[] x, double[][] a) {
//        int m = a.length;
//        int n = a[0].length;
//        if (x.length != n) throw new RuntimeException("Illegal matrix dimensions.");
//        double[] y = new double[m];
//        for (int i = 0; i < m; i++)
//            for (int j = 0; j < n; j++)
//                y[i] += a[i][j] * x[j];
//        return y;
//    }

    public static double[] matrixMultiply(double[] a, double[][] b) {
        // Deals with if b is an n x m matrix
        double[] c;
        if (b.length != 1) {
            c = new double[b[0].length];
            for (int i = 0; i < b[0].length; i++) {

                double sum = 0;
                for (int n = 0; n < a.length; n++) {

                    sum += a[n] * b[n][i];
                }
                c[i] = sum;
            }
        }
        // deals with multiplying b by a scalar
        else if (a.length == 1) {
            if (b.length != 1)
            {
                throw new IllegalArgumentException("Array shapes 1x" + a.length + " and " + b.length + "x" + b[0].length + " are not multiplicable" );
            }
            c = new double[b[0].length];
            for (int i = 0; i < b[0].length; i++) {
                c[i] = a[0] * b[0][i];
            }

        }
        // Deals with if b happens to be a 2d array of size n x 1
        else {
            if (a.length != b[0].length) {
                printArray(a);
                printArray(b);
                throw new IllegalArgumentException("Inputted Matrices are not multiplicable");
            }

            c = new double[] {0};
            for (int i = 0; i < a.length; i++) {
                c[0] += a[i] * b[0][i];
            }

        }
        return c;
    }
//    public static double[] matrixMultiply(double[] matrix1, double[][] matrix2)
//    {
//        int m = matrix2.length;
//
//        int n = matrix2[0].length;
//
//        // ensures that matrices are multiplicable
//        if (matrix1.length != m) {
//            System.out.println("Matrices: " + matrix1.toString() + " and " +matrix2.toString());
//            throw new IllegalArgumentException("Matrices Inputted into Matrix.matrixMultiply cannot be multiplied");
//        }
//
//        double[] output = new double[m];
//
//        for (int i = 0; i < m; i++) {
//            for (int j = 0; j < n; j++) {
//                output[i] += matrix2[i][j] * matrix1[j];
//            }
//        }
//
//        return output;
//    }

    public static double[] matrixMultiply(double[] matrix1, double[] matrix2) {
        if (matrix1.length != matrix2.length) {
            throw new IllegalArgumentException("Matrices Inputted into Matrix.matrixMultiply cannot be multiplied");
        }

        double[] output = new double[matrix1.length];

        for (int i = 0; i < matrix1.length; i++) {
            output[i] = matrix1[i] * matrix2[i];
        }

        return output;
    }


    // Multiplies a 1d matrix by a constant
    public static double[] constantMultiply(double[] matrix, double multiplier) {
        double[] c = new double[matrix.length];

        for (int i = 0; i < matrix.length; i++) {
            c[i] = matrix[i] * multiplier;
        }

        return c;
    }

    // Multiplies a 2d matrix by a constant
    public static double[][] constantMultiply(double[][] matrix, double multiplier) {
        double[][] c = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int n = 0; n < matrix[0].length; n++)
            {
                c[i][n] = matrix[i][n] * multiplier;
            }
        }

        return c;
    }

    // Divides a matrix by a constant
    public static double[] constantDivide(double[] matrix, double divisor) {
        double[] c = new double[matrix.length];

        for (int i = 0; i < matrix.length; i++) {
            c[i] = matrix[i] / divisor;
        }

        return c;
    }

    // Transposes matrix
    public static double[][] transpose(double[][] input) {
        int m = input.length;
        int n = input[0].length;

        double[][] output = new double[n][m];

        for (int i = 0; i < m; i++) {
            for (int k = 0; k < n; k++) {
                output[k][i] = input[i][k];
            }
        }
        return output;
    }

    public static double[][] transpose(double[] input) {
        int m = 1;
        int n = input.length;

        double[][] output = new double[n][m];

        for (int i = 0; i < n; n++) {
            output[m][i] = input[i];
        }

        return output;

    }

    // For testing purposes
    private static void printArray(double[] a) {

        System.out.println("[ ");
        for (int i = 0; i < a.length; i++)
        {
            System.out.print(a[i] + " ");
        }
        System.out.print(" ]");
    }
    private static void printArray(double[][] a) {

        System.out.println("[ ");
        for (int i = 0; i < a.length; i++)
        {
//            System.out.print(", " + a[i]);
            System.out.print("[ ");
            for (int n = 0; n < a[0].length; n++) {
                System.out.print(a[i][n] + " ");
            }
            System.out.print("]");
        }
        System.out.print(" ]");
    }

    //TODO: TESTING
    public static void main(String[] args) {
//        double[] a = new double[] {1, 1};
//        double[][] b = new double[][] {{-1.29, 1.25, -1.74}, {-1.30, 1.32, -1.76}};

//        double[]a = new double[] {-.5742435, .98794503, -.99557553};
//        double[][] b = new double[][] {{2.17395526, .98836921, -1.65973694}};

        double[][] a = new double[][] {{ 0.000000235161797, -0.00310982273, -0.0000271501153}};
        double[][] b = new double[][] {{1}, {1}};

        if (a.length == 1 && b[0].length == 1) {

        }

        double[][] c = matrixMultiply(a, b);
//        double[][] c;

//        c = new double[b.length][a[0].length];
//        for (int i = 0; i < b.length; i++) {
//            for (int j = 0; j < a[0].length; j++) {
//                c[i][j] = b[i][0] * a[0][j];
//            }
//        }

        System.out.println(Arrays.deepToString(c));


//        // Deals with if b is an n x m matrix
//        double[] c;
//        if (b.length != 1) {
//            c = new double[b[0].length];
//            for (int i = 0; i < b[0].length; i++) {
//
//                double sum = 0;
//                for (int n = 0; n < a.length; n++) {
//
//                    sum += a[n] * b[n][i];
//                }
//                c[i] = sum;
//            }
//        }
//        // Deals with if b happens to be a 2d array of size n x 1
//        else {
//            if (a.length != b[0].length) {
//                throw new IllegalArgumentException("Inputted Matrices are not multiplicable");
//            }
//
//            c = new double[] {0};
//            for (int i = 0; i < a.length; i++) {
//                c[0] += a[i] * b[0][i];
//            }
//
//        }
//
//        double[] c = matrixMultiply(a, b);
//
//        System.out.println();
//        System.out.print("[ ");
//
//        for (int i = 0; i < c.length; i++)
//        {
//            System.out.print(c[i] + ", ");
//        }
//        System.out.print("]");
    }
}
