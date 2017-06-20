package ru.yrgu.services;

import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;

import static org.junit.Assert.*;

@RunWith(SpringRunner.class)
@SpringBootTest
public class DLImageRecognitionServiceTest {

    @Autowired
    private DLImageRecognitionService imageRecognitionService;
    @Autowired
    private CifarDataSetIterator dataSetIterator;

    @Test
    public void recognizeFile() throws Exception {
        File testImage = new File(this.getClass().getResource("test_image_3.jpg").toURI());
        ArrayList<Float> result = imageRecognitionService.recognize(new FileInputStream(testImage));

        System.out.println(dataSetIterator.getLabels());
        System.out.println(result);
    }
}