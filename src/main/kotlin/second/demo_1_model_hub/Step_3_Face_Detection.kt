package second.demo_1_model_hub

import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.resize
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.facealignment.Fan2D106FaceAlignmentModel
import org.jetbrains.kotlinx.dl.visualization.swing.createDetectedLandmarksPanel
import org.jetbrains.kotlinx.dl.visualization.swing.showFrame
import second.getFileFromResource
import java.awt.GridLayout
import java.awt.image.BufferedImage
import java.io.File
import javax.swing.JPanel

/**
 * This examples demonstrates the light-weight inference API with [Fan2D106FaceAlignmentModel] on Fan2d106 model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts landmarks on a few images located in resources.
 * - The detected landmarks are drawn on the images used for prediction.
 */
fun main() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.FaceAlignment.Fan2d106.pretrainedModel(modelHub)
    model.printSummary()

    model.use {
        val result = mutableMapOf<BufferedImage, List<Landmark>>()
        for (i in 3..8) {
            val file = getFileFromResource("datasets/faces/image$i.jpeg")
            val image = ImageConverter.toBufferedImage(file)
            val landmarks = it.detectLandmarks(image)
            result[image] = landmarks
        }

        val panel = JPanel(GridLayout(2, 4))
        val resize = pipeline<BufferedImage>().resize { outputWidth = 200; outputHeight = 200 }
        for ((image, landmarks) in result) {
            panel.add(createDetectedLandmarksPanel(resize.apply(image), landmarks))
        }
        showFrame("Face Landmarks", panel)
    }
}