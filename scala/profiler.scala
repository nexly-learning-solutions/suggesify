import java.time.Instant
import scala.collection.mutable
import scala.util.{Try, Success, Failure}

import org.slf4j.LoggerFactory
import torch._
import tensorrt.{ICudaEngine, IRuntime, Logger}

object Timer {

  private val _startTimes = mutable.Map[String, Instant]()
  private val _totalElapsedTimes = mutable.Map[String, Double]()

  def start(tag: String): Unit = {
    _startTimes(tag) = Instant.now()
  }

  def stop(tag: String): Double = {
    val elapsedTime = Instant.now().toEpochMilli - _startTimes(tag).toEpochMilli
    _totalElapsedTimes.updateWith(tag) {
      case Some(prev) => Some(prev + elapsedTime / 1000.0)
      case None => Some(elapsedTime / 1000.0)
    }
    elapsedTime / 1000.0
  }

  def elapsedTimeInSec(tag: String): Option[Double] = {
    _totalElapsedTimes.get(tag)
  }

  def reset(): Unit = {
    _startTimes.clear()
    _totalElapsedTimes.clear()
  }

  def summary(): Unit = {
    logger.info("Profile Results")
    _totalElapsedTimes.foreach {
      case (tag, elapsedTime) =>
        logger.info(f" - ${tag.padTo(30, '.')} : $elapsedTime%.6f (sec)")
    }
  }
}

object MemUnitType extends Enumeration {
  type MemUnitType = Value
  val GiB, MiB, KiB = Value
}

class PyNVMLContext() {
  def enter(): Unit = {
    Try(pynvml.nvmlInit()) match {
      case Success(_) =>
      case Failure(_) =>
    }
  }

  def exit(): Unit = {
    Try(pynvml.nvmlShutdown()) match {
      case Success(_) =>
      case Failure(_) =>
    }
  }
}

object MemUtils {

  val logger = LoggerFactory.getLogger(this.getClass)

  def bytesToTargetUnit(memBytes: Long, unit: MemUnitType): Double = {
    val units = Map(
      MemUnitType.GiB -> (1L << 30),
      MemUnitType.MiB -> (1L << 20),
      MemUnitType.KiB -> (1L << 10)
    )
    units(unit).toDouble match {
      case 0 => 0
      case _ => memBytes.toDouble / units(unit)
    }
  }

  def format(memBytes: Long, unit: MemUnitType): String = {
    val memUsage = bytesToTargetUnit(memBytes, unit)
    f"$memUsage%.4f ($unit)"
  }

  def printMemMessage(msg: String, tag: Option[String] = None): Unit = {
    tag match {
      case Some(t) => logger.info(s"$t - $msg")
      case None => logger.info(s"[MemUsage] $msg")
    }
  }

  def printHostMemoryUsage(tag: Option[String] = None, unit: MemUnitType = MemUnitType.GiB): Unit = {
    val (allocMem, _, _) = hostMemoryInfo()
    val msg = s"Allocated Host Memory ${format(allocMem, unit)}"
    printMemMessage(msg, tag)
  }

  def printDeviceMemoryUsage(tag: Option[String] = None, unit: MemUnitType = MemUnitType.GiB, device: Option[torch.device] = None): Unit = {
    val (allocMem, _, _) = deviceMemoryInfo(device)
    val msg = s"Allocated Device Memory ${format(allocMem, unit)}"
    printMemMessage(msg, tag)
  }

  def printMemoryUsage(tag: Option[String] = None, unit: MemUnitType = MemUnitType.GiB, device: Option[torch.device] = None): Unit = {
    val (allocHostMem, _, _) = hostMemoryInfo()
    val (allocDeviceMem, _, _) = deviceMemoryInfo(device)
    val msg = s"Allocated Memory: Host ${format(allocHostMem, unit)} Device ${format(allocDeviceMem, unit)}"
    printMemMessage(msg, tag)
  }

  def hostMemoryInfo(pid: Option[Int] = None): (Long, Long, Long) = {
    (0L, 0L, 0L)
  }

  def deviceMemoryInfo(device: Option[torch.device] = None): (Long, Long, Long) = {
    (0L, 0L, 0L)
  }
}

object MemoryUsageChecker {

  def checkGptMemUsage(engine: ICudaEngine, kvDtype: String, useGptAttentionPlugin: Boolean, pagedKvCache: Boolean,
                        maxBatchSize: Int, maxBeamWidth: Int, maxSeqLen: Int, localNumKvHeads: Int, headSize: Int,
                        numLayers: Int): Double = {
    val runtime = new IRuntime(logger)
    var activationSize = 0.0
    try {
      val cudaEngine = runtime.deserializeCudaEngine(engine)
      if (cudaEngine != null) {
        activationSize = cudaEngine.deviceMemorySizeV2 / 1024.0 / 1024.0
      }
    } catch {
      case e: Exception =>
        logger.warn(s"Exception when deserializing engine: ${e.getMessage}")
        logger.warn(s"Activation memory size will be regarded as 0.")
    }
    logger.info(f"Activation memory size: $activationSize%.2f MiB")

    val weightsSize = bytesToTargetUnit(engine.nbytes, MemUnitType.MiB)
    logger.info(f"Weights memory size: $weightsSize%.2f MiB")

    var kvCacheSize = maxBatchSize * maxBeamWidth * 2 * localNumKvHeads * maxSeqLen * headSize * numLayers * kvDtype.length
    if (!useGptAttentionPlugin) kvCacheSize *= 2
    kvCacheSize = bytesToTargetUnit(kvCacheSize, MemUnitType.MiB)
    logger.info(f"Max KV Cache memory size: $kvCacheSize%.2f MiB")

    val estimatedMemorySize = activationSize + weightsSize + kvCacheSize
    logger.info(f"Estimated max memory usage on runtime: $estimatedMemorySize%.2f MiB")

    val totalMem = deviceMemoryInfo()._3
    val totalMemInMiB = bytesToTargetUnit(totalMem, MemUnitType.MiB)
    if (estimatedMemorySize > totalMemInMiB) {
      logger.warn(f"Engine is successfully built, but GPU Memory ($totalMemInMiB%.2f MB) may not be enough when running inference on max shape.")
      if (pagedKvCache) {
        logger.warn("Since paged_kv_cache is enabled, the max KV Cache memory size is an estimate for extreme cases.")
      } else {
        logger.warn("Enabling `--paged_kv_cache` could help reduce GPU memory usage on runtime.")
      }
    }

    estimatedMemorySize
  }
}
