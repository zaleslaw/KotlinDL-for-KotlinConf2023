package first.transferLearning

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.SavingFormat
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotUniform
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsForFrozenLayers
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.embedded.dogsCatsSmallDatasetPath
import org.jetbrains.kotlinx.dl.dataset.generator.FromFolders
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import java.awt.image.BufferedImage
import java.io.File

private const val EPOCHS = 3
private const val TRAINING_BATCH_SIZE = 8
private const val TEST_BATCH_SIZE = 16
private const val NUM_CLASSES = 2
private const val NUM_CHANNELS = 3
private const val IMAGE_SIZE = 300
private const val TRAIN_TEST_SPLIT_RATIO = 0.7
private const val PATH_TO_MODEL = "savedmodels/customResNet50"

fun main() {
    val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = TFModels.CV.ResNet50(noTop = true, inputShape = intArrayOf(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    val noTopModel = modelHub.loadModel(modelType)

    val preprocessing = pipeline<BufferedImage>()
        .resize {
            outputHeight = IMAGE_SIZE
            outputWidth = IMAGE_SIZE
            interpolation = InterpolationType.BILINEAR
        }
        .convert { colorMode = ColorMode.BGR }
        .toFloatArray { }
        .call(TFModels.CV.ResNet50().preprocessor)

    val dataset = OnFlyImageDataset.create(
        File(dogsCatsSmallDatasetPath()),
        FromFolders(mapping = mapOf("cat" to 0, "dog" to 1)),
        preprocessing
    ).shuffle()
    val (train, test) = dataset.split(TRAIN_TEST_SPLIT_RATIO)

    val hdfFile = modelHub.loadWeights(modelType)

    val topModel = Sequential.of(
        GlobalAvgPool2D(
            name = "top_avg_pool",
        ),
        Dense(
            name = "top_dense",
            kernelInitializer = GlorotUniform(),
            biasInitializer = GlorotUniform(),
            outputSize = 200,
            activation = Activations.Relu
        ),
        Dense(
            name = "pred",
            kernelInitializer = GlorotUniform(),
            biasInitializer = GlorotUniform(),
            outputSize = NUM_CLASSES,
            activation = Activations.Linear
        ),
        noInput = true
    )

    val model = Functional.of(pretrainedModel = noTopModel, topModel = topModel)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.loadWeightsForFrozenLayers(hdfFile)
        val accuracyBeforeTraining = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy before training $accuracyBeforeTraining")

        it.fit(
            dataset = train,
            batchSize = TRAINING_BATCH_SIZE,
            epochs = EPOCHS
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")

        it.save(
            File(PATH_TO_MODEL),
            SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
            writingMode = WritingMode.OVERRIDE
        )
    }
}