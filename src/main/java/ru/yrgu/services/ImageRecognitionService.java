package ru.yrgu.services;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

public interface ImageRecognitionService {

    public ArrayList<Float> recognize(InputStream inputStream) throws IOException;
}
