plugins {
    kotlin("jvm") version "1.8.0"
    id("org.jetbrains.kotlinx.dataframe") version "0.9.1"
}

group = "com.zaleslaw"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation ("org.jetbrains.kotlinx:kotlin-deeplearning-tensorflow:0.5.1")
    implementation ("org.jetbrains.kotlinx:kotlin-deeplearning-dataset:0.5.1")
    implementation ("org.jetbrains.kotlinx:kotlin-deeplearning-onnx:0.5.1")
    implementation ("org.jetbrains.kotlinx:kotlin-deeplearning-visualization:0.5.1")
    implementation ("org.apache.logging.log4j:log4j-api:2.17.2")
    implementation ("org.apache.logging.log4j:log4j-core:2.17.2")
    implementation ("org.apache.logging.log4j:log4j-slf4j-impl:2.17.2")
    implementation ("org.jetbrains.kotlinx:dataframe:0.9.1")
    implementation ("org.jetbrains.kotlinx:ggdsl-lets-plot:0.4.0-dev-12")
    testImplementation(kotlin("test"))
}

kotlin.sourceSets.getByName("main").kotlin.srcDir("build/generated/ksp/main/kotlin/")

dataframes {
    schema {
        data = "src/main/resources/titanic.csv"
        name = "com.zaleslaw.titanic.Passenger"
        csvOptions {
            delimiter = ';'
        }
    }
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    jvmToolchain(11)
}