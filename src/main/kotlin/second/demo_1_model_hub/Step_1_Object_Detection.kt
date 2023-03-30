package second.demo_1_model_hub

import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.visualization.swing.createDetectedObjectsPanel
import org.jetbrains.kotlinx.dl.visualization.swing.showFrame
import second.getFileFromResource
import java.awt.*
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.abs


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

fun drawRectanglesForDetectedObjects(
    image: File,
    detectedObjects: List<DetectedObject>
): BufferedImage {


    /**
     *
     *
     *     val IMAGEWIDTH = 800
    val IMAGEHEIGHT = 100
    val width = min(ceil(IMAGEWIDTH.toDouble() / size).roundToInt(), 5)
    val bufferedImage = BufferedImage(IMAGEWIDTH, IMAGEHEIGHT, BufferedImage.TYPE_INT_ARGB)
    val graphics2D = bufferedImage.createGraphics()
    graphics2D.setColor(Color(0,0,0,0));
    graphics2D.fillRect(0,0,IMAGEWIDTH,IMAGEHEIGHT);
    val max = this.maxOf { it.toDouble().absoluteValue }
    graphics2D.setColor(Color.decode("#0b5394"));
    forEachIndexed { index, value ->
    val height = (50 * (1.0 * value.toDouble() / max)).roundToInt()
    if (height > 0)
    graphics2D.fillRect(index * (width + 1), 50 - height, width, height)
    else
    graphics2D.fillRect(index * (width + 1), 50, width, height.absoluteValue)
    }
    graphics2D.dispose()
    return bufferedImage
     */
    val bufferedImage = ImageIO.read(image)
    val r = RenderingHints.KEY_ANTIALIASING
    val newGraphics = bufferedImage.createGraphics()
    newGraphics.drawImage(bufferedImage, 0, 0, null)

    detectedObjects.forEach {

        val top = it.yMin * bufferedImage.height
        val left = it.xMin * bufferedImage.width
        val bottom = it.yMax * bufferedImage.height
        val right = it.xMax * bufferedImage.width
        if (abs(top - bottom) > 400 || abs(right - left) > 400) return@forEach

        newGraphics as Graphics2D
        val stroke1: Stroke = BasicStroke(4f)
        when (it.label) {
            "person" -> newGraphics.color = Color.RED
            "bicycle" -> newGraphics.color = Color.BLUE
            "car" -> newGraphics.color = Color.GREEN
            "traffic light" -> newGraphics.color = Color.ORANGE
            "train" -> newGraphics.color = Color.PINK

            else -> newGraphics.color = Color.MAGENTA
        }
        newGraphics.stroke = stroke1
        newGraphics.drawRect(left.toInt(), bottom.toInt(), (right - left).toInt(), (top - bottom).toInt())
    }

    return bufferedImage
}