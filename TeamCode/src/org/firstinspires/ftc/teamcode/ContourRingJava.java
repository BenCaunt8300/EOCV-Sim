package org.firstinspires.ftc.teamcode;

import javafx.scene.effect.GaussianBlur;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.List;

public class ContourRingJava extends OpenCvPipeline {
    Height height = Height.NOT_SCANNED;
    private Mat mat = new Mat();
    private Mat ret = new Mat();
    private Scalar lowerOrange = new Scalar(0.0, 80.0, 150.0, 255.0);
    private Scalar upperOrange = new Scalar(100.0, 170.0, 255.0,255.0);
    private double CAMERA_WIDTH = 320;
    private int HORIZON = (int)((100.0 / 320.0) * CAMERA_WIDTH);
    private double MIN_WIDTH = (50.0 / 320.0) * CAMERA_WIDTH;
    private final double BOUND_RATIO = 0.7;

    @Override
    public Mat processFrame(Mat input) {
        ret.release();
        try {
            Imgproc.cvtColor(input, mat, Imgproc.COLOR_RGB2YCrCb);

            Mat mask = new Mat(mat.rows(), mat.cols(), CvType.CV_8UC1);
            Core.inRange(mat, lowerOrange, upperOrange, mask);

            Core.bitwise_and(input,input,ret,mask);

            Imgproc.GaussianBlur(mask, mask, new Size(5,15),0);

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();

            Imgproc.findContours(mask,contours,hierarchy,Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_NONE);

            double maxWidth = 0;
            Rect maxRect = new Rect();
            for (MatOfPoint c : contours) {
                MatOfPoint2f copy = new MatOfPoint2f(c.toArray());
                Rect rect = Imgproc.boundingRect(copy);

                double w = rect.width;
                if (w > maxWidth && rect.y + rect.height > HORIZON) {
                    maxWidth = w;
                    maxRect = rect;
                }
                c.release();
                copy.release();
            }
            Imgproc.rectangle(ret, maxRect, new Scalar(0.0, 0.0, 255.0), 2);
            System.out.println("x: " + maxRect.x + "y: " + maxRect.y);
            Imgproc.line(ret, new Point(.0, HORIZON), new Point(CAMERA_WIDTH, HORIZON), new Scalar(255.0, .0, 255.0));
            if (maxWidth >= MIN_WIDTH) {
                double aspectRatio = (double)maxRect.height / (double)maxRect.width;
                /** checks if aspectRatio is greater than BOUND_RATIO
                 * to determine whether stack is ONE or FOUR
                 */
                if (aspectRatio > BOUND_RATIO) {
                    height = Height.FOUR; // height variable is now FOUR
                }
                else {
                    height = Height.ONE; // height variable is now ONE
                }
            }  else {
                height = Height.ZERO;
            }
            System.out.println(height);
            mat.release();
            mask.release();
            hierarchy.release();
        } catch (Exception e) {

        }

        return ret;
    }


    private enum Height {
        ZERO, ONE, FOUR, NOT_SCANNED
    }
}
