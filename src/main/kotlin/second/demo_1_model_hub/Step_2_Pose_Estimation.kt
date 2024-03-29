package second.demo_1_model_hub

import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.resize
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.visualization.swing.createDetectedPosePanel
import org.jetbrains.kotlinx.dl.visualization.swing.showFrame
import second.getFileFromResource
import java.awt.FlowLayout
import java.awt.image.BufferedImage
import java.io.File
import javax.swing.JPanel

fun poseDetectionMoveNetLightAPI() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.PoseDetection.MoveNetSinglePoseLighting.pretrainedModel(modelHub)
    model.printSummary()

    model.use { poseDetectionModel ->
        val result = mutableMapOf<BufferedImage, DetectedPose>()
        for (i in 1..3) {
            val file = getFileFromResource("datasets/poses/$i.jpg")
            val image = ImageConverter.toBufferedImage(file)
            val detectedPose = poseDetectionModel.detectPose(image)

            detectedPose.landmarks.forEach {
                println("Found ${it.label} with probability ${it.probability}")
            }

            detectedPose.edges.forEach {
                println("The ${it.label} starts at ${it.start.label} and ends with ${it.end.label}")
            }

            result[image] = detectedPose
        }

        val panel = JPanel(FlowLayout(FlowLayout.CENTER, 0, 0))
        val height = 300
        for ((image, detectedPose) in result) {
            val displayedImage = pipeline<BufferedImage>()
                .resize {
                    outputWidth = (height * image.width) / image.height;
                    outputHeight = height
                }
                .apply(image)
            panel.add(createDetectedPosePanel(displayedImage, detectedPose))
        }
        showFrame("Detection results", panel)
    }
}

/** */
fun main(): Unit = poseDetectionMoveNetLightAPI()