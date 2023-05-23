package second

import java.io.File
import java.net.URISyntaxException
import java.net.URL

/** Converts resource string path to the file. */
@Throws(URISyntaxException::class)
fun getFileFromResource(fileName: String): File {
    val classLoader: ClassLoader = object {}.javaClass.classLoader
    val resource: URL? = classLoader.getResource(fileName)
    return if (resource == null) {
        throw IllegalArgumentException("file not found! $fileName")
    } else {
        File(resource.toURI())
    }
}

const val PROJECT_ROOT = "C:\\Users\\Aleksey.Zinovev\\IdeaProjects\\KotlinDL-for-KotlinConf2023/"

fun resToLabel(res: Int): String = if (res == 0) "cat" else "dog"