plugins {
    kotlin("jvm") version "1.8.0"
}

group = "com.zaleslaw"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation ("org.jetbrains.kotlinx:kotlin-deeplearning-tensorflow:0.5.1")
    implementation ("org.jetbrains.kotlinx:kotlin-deeplearning-dataset:0.5.1")
    implementation ("org.apache.logging.log4j:log4j-api:2.17.2")
    implementation ("org.apache.logging.log4j:log4j-core:2.17.2")
    implementation ("org.apache.logging.log4j:log4j-slf4j-impl:2.17.2")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    jvmToolchain(11)
}