package second

import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Graphics2D
import java.awt.Stroke
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.abs

public fun drawRectanglesForDetectedObjects(
    image: File,
    detectedObjects: List<DetectedObject>
): BufferedImage {
    val bufferedImage = ImageIO.read(image)

    val newGraphics = bufferedImage.createGraphics()
    newGraphics.drawImage(bufferedImage, 0, 0, null)

    detectedObjects.forEach {
        val bottom = it.yMin * bufferedImage.height
        val left = it.xMin * bufferedImage.width
        val top = it.yMax * bufferedImage.height
        val right = it.xMax * bufferedImage.width
        if (abs(top - bottom) > 400 || abs(right - left) > 400) return@forEach

        newGraphics as Graphics2D
        val stroke1: Stroke = BasicStroke(6f)
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
    ImageIO.write(bufferedImage, "jpg", File("../../../serverFiles/image4.jpg"))
    newGraphics.dispose()
    return bufferedImage
}