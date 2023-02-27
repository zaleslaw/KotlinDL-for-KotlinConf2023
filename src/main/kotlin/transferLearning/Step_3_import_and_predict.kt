package transferLearning

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.RMSProp
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.predictTop5Labels
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import org.jetbrains.kotlinx.dl.impl.summary.logSummary
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
    val model = Functional.loadModelConfiguration(File("$PATH_TO_MODEL/modelConfig.json"))

    val fileDataLoader = pipeline<BufferedImage>()
        .resize {
            outputHeight = IMAGE_SIZE
            outputWidth = IMAGE_SIZE
            interpolation = InterpolationType.BILINEAR
        }
        .convert { colorMode = ColorMode.BGR }
        .toFloatArray { }
        .call(TFModels.CV.ResNet50().preprocessor)
        .fileLoader()

    model.use {
        setUpModel(it)

        for (i in 0..49) {
            val inputData = fileDataLoader.load(File("cache/datasets/small-dogs-vs-cats/cat/cat.$i.jpg")).first
            val res = it.predict(inputData)
            println("Predicted object for cat.$i.jpg is $res")
        }
    }
}

private fun setUpModel(it: Functional) {
    it.compile(
        optimizer = RMSProp(),
        loss = Losses.MAE,
        metric = Metrics.ACCURACY
    )
    it.logSummary()

    it.loadWeights(File(PATH_TO_MODEL))
}