package ru.yrgu.services;

import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;

@RunWith(SpringRunner.class)
@SpringBootTest
public class DLImageRecognitionServiceTest {

    @Autowired
    private DLImageRecognitionService imageRecognitionService;
    @Autowired
    private CifarDataSetIterator dataSetIterator;

    @Test
    public void recognizeFile1() throws Exception {
        File testImage = new File(this.getClass().getResource("test_image.jpg").toURI());
        ArrayList<Float> result = imageRecognitionService.recognizeInputStream(new FileInputStream(testImage));

        System.out.println(dataSetIterator.getLabels());
        System.out.println(result);
    }

    @Test
    public void recognizeFile2() throws Exception {
        File testImage = new File(this.getClass().getResource("test_image_2.jpg").toURI());
        ArrayList<Float> result = imageRecognitionService.recognizeFile(testImage);

        System.out.println(dataSetIterator.getLabels());
        System.out.println(result);
    }

    @Test
    public void recognizeFile3() throws Exception {
        File testImage = new File(this.getClass().getResource("test_image_3.jpg").toURI());
        ArrayList<Float> result = imageRecognitionService.recognizeFile(testImage);

        System.out.println(dataSetIterator.getLabels());
        System.out.println(result);
    }
}