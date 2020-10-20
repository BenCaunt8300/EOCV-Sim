/*
 * Copyright (c) 2020 OpenFTC Team
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.firstinspires.ftc.teamcode;

import javafx.scene.effect.GaussianBlur;
import org.opencv.core.*;
import org.opencv.features2d.SimpleBlobDetector;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.List;

import static org.opencv.core.Core.inRange;
import static org.opencv.core.Core.minMaxLoc;

public class ringdetection extends OpenCvPipeline {
    boolean viewportPaused = false;

    /*
     * NOTE: if you wish to use additional Mat objects in your processing pipeline, it is
     * highly recommended to declare them here as instance variables and re-use them for
     * each invocation of processFrame(), rather than declaring them as new local variables
     * each time through processFrame(). This removes the danger of causing a memory leak
     * by forgetting to call mat.release(), and it also reduces memory pressure by not
     * constantly allocating and freeing large chunks of memory.
     */

    @Override
    public Mat processFrame(Mat input)
    {
        boolean forBlue = true;
        /*
         * IMPORTANT NOTE: the input Mat that is passed in as a parameter to this method
         * will only dereference to the same image for the duration of this particular
         * invocation of this method. That is, if for some reason you'd like to save a copy
         * of this particular frame for later use, you will need to either clone it or copy
         * it to another Mat.
         */
        Scalar lowerBounds;
        Scalar upperBounds;
        lowerBounds = new Scalar(0,50,150);
        upperBounds = new Scalar(40,100,255);

        inRange(input,lowerBounds,upperBounds,input);
        Imgproc.GaussianBlur(input, input, new Size(5,5),0);


        //Core.MinMaxLocResult locationOfGoal = minMaxLoc(input);
        //System.out.println("target is at: x" + locationOfGoal.maxLoc.x + ", y" + locationOfGoal.maxLoc.y);
        /*
         * Draw a simple box around the middle 1/2 of the entire frame
         */

        Imgproc.rectangle(
                input,
                new Point(40,
                        70),
                new Point(
                        80,
                        40),
                new Scalar(255, 255, 255), 4);


        /**
         * NOTE: to see how to get data from your pipeline to your OpMode as well as how
         * to change which stage of the pipeline is rendered to the viewport when it is
         * tapped, please see {@link PipelineStageSwitchingExample}
         */

        return input;
    }

    /**
     * Filter out an area of an image using a binary mask.
     * @param input The image on which the mask filters.
     * @param mask The binary image that is used to filter.
     * @param output The image in which to store the output.
     */
    public static void cvMask(Mat input, Mat mask, Mat output) {
        mask.convertTo(mask, CvType.CV_8UC1);
        Core.bitwise_xor(output, output, output);
        input.copyTo(output, mask);
    }
    /**
     * Filters in an area of an image using a binary mask.
     * @param input The image on which the mask filters.
     * @param mask The binary image that is used to filter.
     * @param output The image in which to store the output.
     */
    public static void cvInvertedMask(Mat input, Mat mask, Mat output) {
        Mat cMask = mask.clone();
        mask.convertTo(cMask, CvType.CV_8UC1);
        Core.bitwise_not(cMask, cMask);
        Core.bitwise_xor(output, output, output);
        input.copyTo(output, cMask);
        cMask.release();
    }
    public static void cvFindContours(Mat input, boolean externalOnly, List<MatOfPoint> contours) {

        Mat hierarchy = new Mat();
        contours.clear();

        int mode;
        if (externalOnly) {
            mode = Imgproc.RETR_EXTERNAL;
        } else {
            mode = Imgproc.RETR_LIST;
        }

        int method = Imgproc.CHAIN_APPROX_SIMPLE;
        Imgproc.findContours(input, contours, hierarchy, mode, method);

    }

}

