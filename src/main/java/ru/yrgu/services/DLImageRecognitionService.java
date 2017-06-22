package ru.yrgu.services;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;

@Service
public class DLImageRecognitionService implements ImageRecognitionService {

    @Autowired
    private MultiLayerNetwork multiLayerNetwork;

    @Override
    public ArrayList<Float> recognizeInputStream(InputStream inputStream) throws IOException {

        Path tempFile = Files.createTempFile(null, ".png");
        try {
            Files.copy(inputStream, tempFile, StandardCopyOption.REPLACE_EXISTING);

            ParentPathLabelGenerator gen = new ParentPathLabelGenerator();
            ImageRecordReader reader = new ImageRecordReader(32, 32, 3, gen);
            reader.initialize(new FileSplit(new File(tempFile.toString())));
            DataSetIterator dataIter = new RecordReaderDataSetIterator(reader, 50);
            INDArray array = null;
            while (dataIter.hasNext()) {
                DataSet set = dataIter.next();
                array = multiLayerNetwork.output(set.getFeatures(), false);
            }

            ArrayList<Float> result = new ArrayList<>();
            for(int i=0; i<array.data().length(); i++){
                result.add(array.data().getFloat(i));
            }

            return result;
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    @Override
    public ArrayList<Float> recognizeFile(File file) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(32, 32, 3);
        INDArray image = loader.asMatrix(file);

        INDArray array = multiLayerNetwork.output(image);

        ArrayList<Float> result = new ArrayList<>();
        for(int i=0; i<array.data().length(); i++){
            result.add(array.data().getFloat(i));
        }

        return result;
    }


}
