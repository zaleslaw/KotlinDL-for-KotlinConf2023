package second.demo_1_model_hub

import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.visualization.swing.createDetectedObjectsPanel
import org.jetbrains.kotlinx.dl.visualization.swing.showFrame
import second.getFileFromResource
import java.io.File
import javax.imageio.ImageIO


fun main() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.ObjectDetection.EfficientDetD2.pretrainedModel(modelHub)
    model.printSummary()

    model.use { detectionModel ->
        val file = getFileFromResource("datasets/detection/image4.jpg")
        val image = ImageConverter.toBufferedImage(file)
        val detectedObjects = detectionModel.detectObjects(image)

        detectedObjects.forEach {
            println("Found ${it.label} with probability ${it.probability}")
        }

        showFrame("Detection result for ${file.name}", createDetectedObjectsPanel(image, detectedObjects))

        ImageIO.write(image, "jpg", File("serverFiles/image4.jpg"))
    }
}