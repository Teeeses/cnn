package ru.yrgu.web;

import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import ru.yrgu.services.ImageRecognitionService;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/services/image/recognize")
public class RecognizeController {

    @Autowired
    private ImageRecognitionService imageRecognitionService;
    @Autowired
    private CifarDataSetIterator dataSetIterator;

    @PostMapping
    public List<ProbabilityDTO> recognize(MultipartFile file) throws IOException {
        ArrayList<Float> result = imageRecognitionService.recognize(file.getInputStream());
        List<String> labels = dataSetIterator.getLabels();
        return result.stream().map(value -> new ProbabilityDTO(labels.get(result.indexOf(value)), value)).collect(Collectors.toList());
    }
}
