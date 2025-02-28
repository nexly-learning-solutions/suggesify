import java.nio.file.{Files, Paths}
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets
import scala.collection.mutable
import scala.util.Using
import scala.util.control.NonFatal
import org.slf4j.LoggerFactory

object TensorRT {
  class ICudaEngine {
    def serialize(): Array[Byte] = Array()
  }
  
  class Runtime(logger: Logger) {
    def deserializeCudaEngine(data: Array[Byte]): Option[ICudaEngine] = Some(new ICudaEngine())
  }
}

object Logger {
  private val logger = LoggerFactory.getLogger("TensorRT-LLM")

  def info(msg: String): Unit = logger.info(msg)
  def warning(msg: String): Unit = logger.warn(msg)
  def error(msg: String): Unit = logger.error(msg)
}

var net: Option[Network] = None
var _inited: Boolean = false

trait Network {
  var dtype: String
  def trtNetwork: Any
}

def _init(logLevel: Option[Any] = None): Unit = {
  if (_inited) return
  _inited = true

  logLevel.foreach(Logger.info)

  if (sys.env.getOrElse("TRT_LLM_NO_LIB_INIT", "0") == "1") {
    Logger.info("Skipping TensorRT-LLM init.")
    return
  }

  Logger.info("Starting TensorRT-LLM init.")

  try {
    registerFake()
  } catch {
    case NonFatal(e) =>
      throw new Exception(s"FATAL: Decoding operators failed to load. Please rebuild TensorRT-LLM. ${e.getMessage}")
  }

  MpiComm.localInit()
  Logger.info("TensorRT-LLM initialized.")
}

def registerFake(): Unit = {
  Logger.info("Registering fake tensor operations.")
}

def defaultNet: Network = net.getOrElse(throw new IllegalStateException("Use builder to create network first"))

def defaultTrtNet: Any = defaultNet.trtNetwork

def setNetwork(network: Network): Unit = {
  net = Some(network)
}

def switchNetDtype(curDtype: String): String = {
  val prevDtype = defaultNet.dtype
  defaultNet.dtype = curDtype
  prevDtype
}

def precision(dtype: String)(block: => Unit): Unit = {
  val prevDtype = switchNetDtype(dtype)
  try block finally switchNetDtype(prevDtype)
}

def serializeEngine(engine: TensorRT.ICudaEngine, path: String): Unit = {
  Logger.info(s"Serializing engine to $path...")
  val start = System.currentTimeMillis()
  val data = engine.serialize()
  Files.write(Paths.get(path), data)
  val duration = (System.currentTimeMillis() - start) / 1000
  Logger.info(s"Engine serialized. Total time: ${duration}s")
}

def deserializeEngine(path: String): TensorRT.ICudaEngine = {
  Logger.info(s"Loading engine from $path...")
  val start = System.currentTimeMillis()
  val runtime = new TensorRT.Runtime(Logger)
  val data = Files.readAllBytes(Paths.get(path))
  val engine = runtime.deserializeCudaEngine(data).getOrElse(
    throw new IllegalStateException("Failed to deserialize CUDA engine")
  )
  val duration = (System.currentTimeMillis() - start) / 1000
  Logger.info(s"Engine loaded. Total time: ${duration}s")
  engine
}

val fieldDtypeToNpDtypeMap: Map[Int, String] = Map(
  1 -> "Float16",
  2 -> "Float32",
  3 -> "Float64",
  4 -> "Int8",
  5 -> "Int16",
  6 -> "Int32"
)

def fieldDtypeToNpDtype(dtype: Int): String = {
  fieldDtypeToNpDtypeMap.getOrElse(dtype, throw new IllegalArgumentException(s"Unsupported dtype: $dtype"))
}

def convertCapsuleToVoidP(capsule: Any): Long = {
  0L
}

def getNumpyArrayFromVoidP(voidPointer: Long, elemSize: Int, fieldDtype: Int): Array[Byte] = {
  Logger.info(s"Retrieving numpy array from pointer: $voidPointer, element size: $elemSize")
  val npDtype = fieldDtypeToNpDtype(fieldDtype)
  val bufferSize = elemSize * 4
  new Array[Byte](bufferSize)
}

def getScalarFromField(field: { def data: Any; def `type`: Int }): Any = {
  val voidP = convertCapsuleToVoidP(field.data)
  val npArray = getNumpyArrayFromVoidP(voidP, 1, field.`type`)
  npArray(0)
}

class BuildingFlag extends AutoCloseable {
  sys.props += ("IS_BUILDING" -> "1")
  override def close(): Unit = sys.props -= "IS_BUILDING"
}

def isBuilding[T](f: => T): T = {
  Using(new BuildingFlag) { _ =>
    f
  }.get
}

def checkMaxNumTokens(
    maxNumTokens: Option[Int],
    optNumTokens: Option[Int],
    maxBatchSize: Int,
    maxInputLen: Int,
    maxSeqLen: Int,
    maxBeamWidth: Int,
    removeInputPadding: Boolean,
    enableContextFmha: Boolean,
    tokensPerBlock: Int,
    multipleProfiles: Boolean
): (Int, Option[Int]) = {

  var finalMaxNumTokens = maxNumTokens.getOrElse(maxSeqLen * maxBatchSize)
  var finalOptNumTokens = optNumTokens

  if (!removeInputPadding) {
    if (maxNumTokens.isDefined || optNumTokens.isDefined) {
      finalMaxNumTokens = maxBatchSize * maxSeqLen
      Logger.warning("remove_input_padding is not enabled, ignoring max_num_tokens/opt_num_tokens.")
    }
    return (finalMaxNumTokens, finalOptNumTokens)
  }

  if (finalMaxNumTokens > 16384) {
    Logger.warning(s"max_num_tokens ($finalMaxNumTokens) too large, may cause runtime errors.")
  }

  if (finalMaxNumTokens < maxInputLen && !enableContextFmha) {
    Logger.warning(s"max_num_tokens ($finalMaxNumTokens) should be at least max_input_len ($maxInputLen).")
    finalMaxNumTokens = maxInputLen
  }

  if (finalMaxNumTokens < tokensPerBlock && enableContextFmha) {
    Logger.warning(s"max_num_tokens ($finalMaxNumTokens) should be at least tokens_per_block ($tokensPerBlock).")
    finalMaxNumTokens = tokensPerBlock
  }

  if (finalOptNumTokens.exists(_ > finalMaxNumTokens)) {
    Logger.warning(s"opt_num_tokens ($finalOptNumTokens) should not exceed max_num_tokens ($finalMaxNumTokens).")
    finalOptNumTokens = Some(finalMaxNumTokens)
  }

  (finalMaxNumTokens, finalOptNumTokens)
}
