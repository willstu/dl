package com.willstu.dl

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.LoggerFactory

object Test {

    private val log = LoggerFactory.getLogger(Test::class.java)

    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {

        //number of rows and columns in the input pictures
        val numRows = 28
        val numColumns = 28
        val outputNum = 10 // number of output classes
        val batchSize = 128 // batch size for each epoch
        val rngSeed = 123 // random number seed for reproducibility
        val numEpochs = 15 // number of epochs to perform

        //Get the DataSetIterators:
        val mnistTrain = MnistDataSetIterator(batchSize, true, rngSeed)
        val mnistTest = MnistDataSetIterator(batchSize, false, rngSeed)

        log.info("Build model....")
        val conf = NeuralNetConfiguration.Builder()
                .seed(rngSeed.toLong()) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .updater(Nesterovs(0.006, 0.9)) //specify the rate of change of the learning rate.
                .l2(1e-4)
                .list()
                .layer(0, DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(numRows * numColumns)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true) //use backpropagation to adjust weights
                .build()

        val model = MultiLayerNetwork(conf)
        model.init()
        //print the score with every 1 iteration
        model.setListeners(ScoreIterationListener(1))

        log.info("Train model....")
        for (i in 0..numEpochs - 1) {
            model.fit(mnistTrain)
        }

        log.info("Evaluate model....")
        val eval = Evaluation(outputNum) //create an evaluation object with 10 possible classes
        while (mnistTest.hasNext()) {
            val next = mnistTest.next()
            val output = model.output(next.features) //get the networks prediction
            eval.eval(next.getLabels(), output) //check the prediction against the true class
        }

        log.info(eval.stats())
        log.info("****************Example finished********************")

    }
}
