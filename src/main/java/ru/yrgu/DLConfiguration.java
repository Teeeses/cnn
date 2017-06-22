package ru.yrgu;

import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

@Configuration
public class DLConfiguration {

    @Value("${network.backup.filepath}")
    private String networkBackup;

    @Bean
    public ConvolutionLayer layer0() {
        return new ConvolutionLayer.Builder(5, 5)
                .nIn(3)
                .nOut(16)
                .stride(1, 1)
                .padding(2, 2)
                .weightInit(WeightInit.XAVIER)
                .name("First convolution layer")
                .activation(Activation.RELU)
                .build();
    }

    @Bean
    public SubsamplingLayer layer1() {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .name("First subsampling layer")
                .build();
    }

    @Bean
    public ConvolutionLayer layer2() {
        return new ConvolutionLayer.Builder(5, 5)
                .nOut(20)
                .stride(1, 1)
                .padding(2, 2)
                .weightInit(WeightInit.XAVIER)
                .name("Second convolution layer")
                .activation(Activation.RELU)
                .build();
    }

    @Bean
    public SubsamplingLayer layer3() {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .name("Second subsampling layer")
                .build();
    }

    @Bean
    public ConvolutionLayer layer4() {
        return new ConvolutionLayer.Builder(5, 5)
                .nOut(20)
                .stride(1, 1)
                .padding(2, 2)
                .weightInit(WeightInit.XAVIER)
                .name("Third convolution layer")
                .activation(Activation.RELU)
                .build();
    }

    @Bean
    public SubsamplingLayer layer5() {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .name("Third subsampling layer")
                .build();
    }

    @Bean
    public OutputLayer layer6() {
        return new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .name("Output")
                .nOut(10)
                .build();
    }

    @Bean
    public MultiLayerConfiguration multiLayerConfiguration(
            ConvolutionLayer layer0,
            SubsamplingLayer layer1,
            ConvolutionLayer layer2,
            SubsamplingLayer layer3,
            ConvolutionLayer layer4,
            SubsamplingLayer layer5,
            OutputLayer layer6) {
        return new NeuralNetConfiguration.Builder()
                .seed(12345)
                .iterations(50)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.001)
                .regularization(true)
                .l2(0.0004)
                .updater(Updater.NESTEROVS)
                .momentum(0.9)
                .list()
                .layer(0, layer0)
                .layer(1, layer1)
                .layer(2, layer2)
                .layer(3, layer3)
                .layer(4, layer4)
                .layer(5, layer5)
                .layer(6, layer6)
                .pretrain(false)
                .backprop(true)
                .setInputType(InputType.convolutional(32, 32, 3))
                .build();
    }

    @Bean
    public CifarDataSetIterator dataSetIterator() {
        return new CifarDataSetIterator(2, 5000, true);
    }

    @Bean
    public MultiLayerNetwork multiLayerNetwork(MultiLayerConfiguration multiLayerConfiguration,
                                               CifarDataSetIterator dataSetIterator) throws IOException {
        MultiLayerNetwork network;
        if(Files.exists(Paths.get(networkBackup))){
            File locationToSave = new File(networkBackup);
            network = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
        }else{
            network = new MultiLayerNetwork(multiLayerConfiguration);
            network.init();
            network.fit(dataSetIterator);
            try {
                File locationToSave = new File(networkBackup);
                ModelSerializer.writeModel(network, locationToSave, true);
            } catch (IOException e) {
                System.out.println(e);
            }
        }
        return network;
    }
}
