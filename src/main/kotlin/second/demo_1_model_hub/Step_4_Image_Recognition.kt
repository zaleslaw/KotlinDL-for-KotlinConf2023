package second.demo_1_model_hub

import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import second.getFileFromResource
import java.io.File

fun resnet18LightAPIPrediction() {
    val modelHub =
        ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))

    val model = ONNXModels.CV.ResNet50.pretrainedModel(modelHub)
    model.printSummary()

    model.use {
        val imageFile = getFileFromResource("datasets/kdog.jpg")

        val recognizedObject = it.predictObject(imageFile = imageFile)
        println(recognizedObject)

        val top5 = it.predictTopKObjects(imageFile = imageFile, topK = 5)
        println(top5.toString())
    }
}

/** */
fun main(): Unit = resnet18LightAPIPrediction()
