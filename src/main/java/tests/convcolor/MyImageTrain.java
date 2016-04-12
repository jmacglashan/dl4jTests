package tests.convcolor;

import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetPreProcessor;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class MyImageTrain {

	public static void main(String[] args) {
		trainNet("simpleImages", "simpleImagesTest", 300);
	}

	public static void trainNet(String pathTrain, String pathTest, int batchSize){


		int nChannels = 3;
		int outputNum = 3;
		int nEpochs = 50;
		int iterations = 1;
		int seed = 123;

		String junk = "this shouldn't matter";
		DataSetIterator iter = canovaConstruct(pathTrain, 28, 28, nChannels, batchSize);
		DataSetIterator iter2 = canovaConstruct(pathTest, 28, 28, nChannels, batchSize);

		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.regularization(true).l2(0.0005)
				.learningRate(0.01)
				.weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS).momentum(0.9)
				.list(4)
				.layer(0, new ConvolutionLayer.Builder(5, 5)
						.nIn(nChannels)
						.stride(1, 1)
						.nOut(20).dropOut(0.5)
						.activation("relu")
						.build())
				.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2,2)
						.stride(2,2)
						.build())
				.layer(2, new DenseLayer.Builder().activation("relu")
						.nOut(500).build())
				.layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nOut(outputNum)
						.activation("softmax")
						.build())
				.backprop(true).pretrain(false);
		new ConvolutionLayerSetup(builder,28,28,3);

		MultiLayerConfiguration conf = builder.build();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		model.setListeners(new ScoreIterationListener(1));
		for( int i=0; i<nEpochs; i++ ) {
			model.fit(iter);

			Evaluation eval = new Evaluation(outputNum);
			while(iter2.hasNext()){
				DataSet ds = iter2.next();
				INDArray output = model.output(ds.getFeatureMatrix());
				eval.eval(ds.getLabels(), output);

			}

			System.out.println(eval.stats());
			iter.reset();
			iter2.reset();

		}


	}

	public static DataSetIterator canovaConstruct(String labeledPath, int w, int h, int channels, int batchSize){


		//create array of strings called labels
		List<String> labels = new ArrayList<>();

		//traverse dataset to get each label
		File rootDir = new File(labeledPath);
		for(File f : rootDir.listFiles()) {
			labels.add(f.getName());
		}

		// Instantiating RecordReader. Specify height and width of images.
		ImageRecordReader recordReader = new ImageRecordReader(w, h, channels, true, labels);

		try {
			recordReader.initialize(new FileSplit(new File(labeledPath)));
		} catch(IOException e) {
			e.printStackTrace();
		}

		DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, w*h*channels, labels.size());
		iter.setPreProcessor(new DataSetPreProcessor() {
			@Override
			public void preProcess(org.nd4j.linalg.dataset.api.DataSet toPreProcess) {
				toPreProcess.getFeatureMatrix().divi(255);
			}
		});

		return iter;

	}

}
