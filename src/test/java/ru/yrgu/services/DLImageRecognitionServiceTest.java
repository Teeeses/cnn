package ru.yrgu.services;

import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

@RunWith(SpringRunner.class)
@SpringBootTest
public class DLImageRecognitionServiceTest {

    @Autowired
    private DLImageRecognitionService imageRecognitionService;
    @Autowired
    private CifarDataSetIterator dataSetIterator;

    @Autowired
    private MultiLayerNetwork multiLayerNetwork;

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

    @Test
    public void train() throws Exception {
        /*List<String> LABELS = Arrays.asList("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck");
        int splitTrainNum = (int) (50 * 0.8);
        int listenerFreq = 50 / 5;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();

        System.out.println("Train model");
        while (dataSetIterator.hasNext()) {
            multiLayerNetwork.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
            DataSet cifarDataSet = dataSetIterator.next();
            SplitTestAndTrain trainAndTest = cifarDataSet.splitTestAndTrain(splitTrainNum, new Random(12345));
            DataSet trainInput = trainAndTest.getTrain();
            testInput.add(trainAndTest.getTest().getFeatureMatrix());
            testLabels.add(trainAndTest.getTest().getLabels());
            multiLayerNetwork.fit(trainInput);
        }

        System.out.println("Evaluate model");
        Evaluation eval = new Evaluation(LABELS.size());
        for (int i = 0; i < testInput.size(); i++) {
            INDArray output = multiLayerNetwork.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }

        System.out.println(eval.stats());*/

        while(dataSetIterator.hasNext()){
            DataSet next = dataSetIterator.next();
            multiLayerNetwork.fit(next);
        }

        dataSetIterator.reset();
        Evaluation eval = new Evaluation();
        while(dataSetIterator.hasNext()){
            DataSet next = dataSetIterator.next();
            INDArray predict2 = multiLayerNetwork.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), predict2);
        }

        System.out.println(eval.stats());
    }
}