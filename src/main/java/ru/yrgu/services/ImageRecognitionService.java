package ru.yrgu.services;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

public interface ImageRecognitionService {

    ArrayList<Float> recognizeInputStream(InputStream inputStream) throws IOException;
    ArrayList<Float> recognizeFile(File file) throws IOException;
}
