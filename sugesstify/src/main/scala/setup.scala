import java.io.{BufferedReader, InputStreamReader}
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Executors, TimeUnit}
import scala.concurrent.{ExecutionContext, Future}
import scala.util.{Failure, Success}
import scala.sys.process._
import org.apache.logging.log4j.{LogManager, Logger}
import scopt.OptionParser

object GitRepoCloner {

  val logger: Logger = LogManager.getLogger(GitRepoCloner.getClass)

  def checkGitInstalled(): Unit = {
    try {
      val process = "git --version".run(ProcessLogger(line => println(line)))
      process.exitValue()
    } catch {
      case e: Exception =>
        logger.error("Git is not installed or not found in the system PATH.")
        System.exit(1)
    }
  }

  def cloneRepo(repoUrl: String, targetDir: String, retries: Int = 3, delay: Int = 5): Unit = {
    for (_ <- 1 to retries) {
      try {
        val repoName = repoUrl.split("/").last.replace(".git", "")
        val repoPath = Paths.get(targetDir, repoName)

        if (Files.exists(repoPath)) {
          logger.info(s"Repository $repoName already exists, skipping clone.")
          return
        }

        logger.info(s"Cloning $repoUrl...")
        s"git clone $repoUrl".! match {
          case 0 =>
            logger.info(s"Successfully cloned $repoUrl")
            return
          case _ =>
            logger.error(s"Error cloning $repoUrl")
            Thread.sleep(delay * 1000)
        }
      } catch {
        case e: Exception =>
          logger.error(s"Error cloning $repoUrl: $e")
          Thread.sleep(delay * 1000)
      }
    }
    logger.error(s"Failed to clone $repoUrl after $retries attempts.")
  }

  def cloneRepositories(directory: String, repos: Seq[String], maxWorkers: Int = 4): Unit = {
    val path = Paths.get(directory)
    if (!Files.exists(path)) {
      Files.createDirectories(path)
    }

    implicit val ec: ExecutionContext = ExecutionContext.fromExecutor(Executors.newFixedThreadPool(maxWorkers))

    val futures = repos.map { repo =>
      Future {
        cloneRepo(repo, directory)
      }
    }

    futures.foreach { future =>
      future.onComplete {
        case Success(_) => logger.info("Repository cloned successfully.")
        case Failure(e) => logger.error(s"An error occurred: ${e.getMessage}")
      }
    }

    futures.foreach(_.onComplete(_ => ()))
    Thread.sleep(2000)
  }

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[Config]("GitRepoCloner") {
      opt[String]('d', "directory")
        .action((x, c) => c.copy(directory = x))
        .text("Directory to create and clone repositories into")
        .withFallback(() => "3rdparty")

      opt[Int]('w', "workers")
        .action((x, c) => c.copy(workers = x))
        .text("Number of concurrent cloning processes")
        .withFallback(() => 4)
    }

    parser.parse(args, Config()) match {
      case Some(config) =>
        checkGitInstalled()

        val repositories = Seq(
          "https://github.com/OpenMathLib/OpenBLAS.git",
          "https://github.com/pybind/pybind11.git",
          "https://github.com/Unidata/netcdf-cxx4.git",
          "https://github.com/NVIDIA/TensorRT.git",
          "https://github.com/microsoft/Microsoft-MPI.git",
          "https://github.com/python/cpython.git",
          "https://github.com/NVIDIA/nccl.git",
          "https://github.com/pytorch/pytorch.git",
          "https://github.com/NVIDIA/cudnn-frontend.git",
          "https://github.com/open-source-parsers/jsoncpp.git",
          "https://github.com/google/highway.git",
          "https://github.com/google/googletest.git",
          "https://github.com/NVIDIA/cutlass.git",
          "https://github.com/GerHobbelt/pthread-win32.git",
          "https://github.com/boostorg/boost.git",
          "https://github.com/google/glog.git",
          "https://github.com/NVIDIA/cccl.git"
        )

        cloneRepositories(config.directory, repositories, config.workers)
        logger.info("All repositories have been processed.")

      case None =>
        logger.error("Failed to parse arguments.")
    }
  }

  case class Config(directory: String = "3rdparty", workers: Int = 4)
}
