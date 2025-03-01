name := "sugesstify"

version := "0.1"

scalaVersion := "2.13.16"

ThisBuild / scalaVersion := "2.13.16"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / organization := "com.nexly"
ThisBuild / organizationName := "nexly"

lazy val root = (project in file("."))
  .settings(
    name := "sugesstify",
    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "0.7.29" % Test,
      "org.apache.logging.log4j" % "log4j-api" % "2.14.1",
      "org.apache.logging.log4j" % "log4j-core" % "2.14.1",
      "org.apache.logging.log4j" % "log4j-slf4j-impl" % "2.14.1",
      "com.github.scopt" %% "scopt" % "4.0.1",
      "org.scala-lang" % "scala-library" % "2.13.16"
    ),
  )
