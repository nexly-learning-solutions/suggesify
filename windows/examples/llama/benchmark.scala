import java.nio.file.{Files, Path, Paths}
import scala.concurrent._
import scala.concurrent.duration._
import ExecutionContext.Implicits.global
import sys.process._

object BenchmarkLlamaTensorRTLLM {
  
  case class Args(modelDir: Path, quantCkptPath: Path, engineDir: Path)

  def parseArguments(args: Array[String]): Args = {
    val parser = new scopt.OptionParser[Args]("Benchmark llama nexly") {
      opt[String]("model_dir").required().action((x, c) =>
        c.copy(modelDir = Paths.get(x))).text("Directory with HF model")
      opt[String]("quant_ckpt_path").required().action((x, c) =>
        c.copy(quantCkptPath = Paths.get(x))).text("Path with quantized weights")
      opt[String]("engine_dir").action((x, c) =>
        c.copy(engineDir = Paths.get(x))).text("Directory to store engines").default(Paths.get("."))
    }
    parser.parse(args, Args(Paths.get(""), Paths.get(""), Paths.get(""))).getOrElse {
      throw new IllegalArgumentException("Invalid arguments")
    }
  }

  def main(args: Array[String]): Unit = {
    val parsedArgs = parseArguments(args)
    val modelDir = parsedArgs.modelDir
    val quantCkptPath = parsedArgs.quantCkptPath
    var engineDir = parsedArgs.engineDir

    assert(Files.exists(modelDir), "Please pass a valid, existing model path")
    assert(Files.exists(quantCkptPath), "Please pass a valid, existing path to quantized weights")

    if (!Files.exists(engineDir)) engineDir = Paths.get(System.getProperty("user.dir"), "engines")

    val dirLevel = 3
    val topLevelPath = Paths.get(System.getProperty("user.dir")).toAbsolutePath.getParent.getParent.getParent
    val examplePath = topLevelPath.resolve("examples").resolve("llama")
    val benchmarkPath = topLevelPath.resolve("benchmarks").resolve("python")

    val inputSeqLen = 100
    val outputSeqLen = 100
    val batchSize = 8

    val buildArgs = Seq(
      sys.props("user.dir"),
      examplePath.resolve("build.py").toString,
      "--model_dir", modelDir.toString,
      "--quant_ckpt_path", quantCkptPath.toString,
      "--dtype", "float16",
      "--log_level", "info",
      "--use_gpt_attention_plugin", "float16",
      "--use_gemm_plugin", "float16",
      "--enable_context_fmha",
      "--use_weight_only",
      "--weight_only_precision", "int4_gptq",
      "--per_group",
      "--max_input_len", inputSeqLen.toString,
      "--max_seq_len", (outputSeqLen + inputSeqLen).toString,
      "--n_positions", (inputSeqLen + outputSeqLen + 1).toString,
      "--max_batch_size", batchSize.toString,
      "--output_dir", engineDir.toString
    )

    val benchmarkArgs = Seq(
      sys.props("user.dir"),
      benchmarkPath.resolve("benchmark.py").toString, "--engine_dir", engineDir.toString,
      "--mode", "plugin", "-m", "llama_7b", "--dtype", "float16", "--log_level", "info", "--batch_size",
      batchSize.toString, "--input_output_len", s"$inputSeqLen,$outputSeqLen", "--num_beams", "1", "--warm_up", "1",
      "--num_runs", "3", "--duration", "10", "--csv"
    )

    def run(args: Seq[String]): Future[Unit] = Future {
      println(s"Running ${args.mkString(" ")}")
      val command = args.mkString(" ")
      command.!
    }

    for {
      _ <- run(buildArgs)
      _ <- run(benchmarkArgs)
    } yield ()
  }
}
