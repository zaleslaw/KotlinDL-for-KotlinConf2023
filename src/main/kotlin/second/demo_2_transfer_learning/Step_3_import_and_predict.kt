package second.demo_2_transfer_learning

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.RMSProp
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import org.jetbrains.kotlinx.dl.impl.summary.logSummary
import java.awt.image.BufferedImage
import java.io.File

private const val IMAGE_SIZE = 300
private const val PATH_TO_MODEL = "savedmodels/customResNet50"

fun main() {
    val model = Functional.loadModelConfiguration(File("$PATH_TO_MODEL/modelConfig.json"))

    val fileDataLoader = getFileDataLoader()

    model.use {
        setUpModel(it)

        for (i in 0..49) {
            val inputData = fileDataLoader.load(File("cache/datasets/small-dogs-vs-cats/cat/cat.$i.jpg")).first
            val res = it.predict(inputData)
            println("Predicted object for cat.$i.jpg is $res")
        }
    }
}

private fun getFileDataLoader() = pipeline<BufferedImage>()
    .resize {
        outputHeight = IMAGE_SIZE
        outputWidth = IMAGE_SIZE
        interpolation = InterpolationType.BILINEAR
    }
    .convert { colorMode = ColorMode.BGR }
    .toFloatArray { }
    .call(TFModels.CV.ResNet50().preprocessor)
    .fileLoader()

private fun setUpModel(it: Functional) {
    it.compile(
        optimizer = RMSProp(),
        loss = Losses.MAE,
        metric = Metrics.ACCURACY
    )
    it.logSummary()
    println ("Starting load model weights.....")
    it.loadWeights(File(PATH_TO_MODEL))
}