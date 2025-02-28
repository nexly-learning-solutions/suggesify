import scala.collection.mutable
import scala.math
import breeze.linalg._
import org.bytedeco.tensorRT.global.nvinfer.{DataType => TrtDataType}
import java.lang.ref.WeakReference

class Parameter(
    var value: Option[DenseMatrix[Double]] = None,
    var shape: Option[Seq[Int]] = None,
    var dtype: Option[TrtDataType] = None,
    var isBuffer: Boolean = false,
    var preferManaged: Boolean = false
) {
  private val defaultDType: TrtDataType = TrtDataType.kFLOAT
  private var _tensor: Option[Tensor] = None
  private var _network: Option[WeakReference[Network]] = None
  private var _name: Option[String] = None
  private var needTranspose: Boolean = false

  dtype = dtype.orElse(Some(defaultDType))

  if (value.isEmpty) {
    require(shape.isDefined, "Shape must be provided if value is None")
    this.shape = shape
  } else {
    this.shape = Some(value.get.size)
  }

  def getShape: Seq[Int] = shape.getOrElse(Seq())

  def getDtype: TrtDataType = dtype.get

  def getName: Option[String] = _name

  private def createManagedTensor(network: Network, needTranspose: Boolean = false): Tensor = {
    val num = network.inputs.size
    val name = s"managed_constant_$num"
    _name = Some(name)
    var tensorShape = getShape

    if (needTranspose) {
      require(tensorShape.length == 2)
      tensorShape = tensorShape.reverse
    }

    if (value.isEmpty) {
      value = Some(DenseMatrix.zeros[Double](tensorShape: _*))
      network.registerUnfilledWeights(name, value.get)
    }
    new Tensor(name, dtype.get, tensorShape)
  }

  def getManagedTensor(network: Network, needTranspose: Boolean = false): Tensor = {
    if (_network.isEmpty || _network.get.get() != network) {
      _network = Some(new WeakReference(network))
      _tensor = network.getParameterTensor(this)
      this.needTranspose = needTranspose
      if (_tensor.isEmpty) {
        _tensor = Some(createManagedTensor(network, needTranspose))
        network.setParameterTensor(this, _tensor.get)
      }
    }
    _tensor.get
  }

  private def createConstantTensor(): Tensor = {
    if (value.isDefined) {
      return Tensor.constant(value.get)
    }

    val tensorShape = getShape
    val dtypeScala = dtype.get match {
      case TrtDataType.kFLOAT => Double
      case TrtDataType.kINT8  => Byte
      case _                  => Double
    }

    val emptyMatrix = DenseMatrix.zeros[Double](tensorShape: _*)
    val tensor = Tensor.constant(emptyMatrix)
    defaultNet().registerUnfilledWeights(tensor.producerName, emptyMatrix, value)
    tensor
  }

  def getConstantTensor(network: Network): Tensor = {
    if (_network.isEmpty || _network.get.get() != network) {
      _network = Some(new WeakReference(network))
      _tensor = network.getParameterTensor(this)
      if (_tensor.isEmpty) {
        _tensor = Some(createConstantTensor())
        _name = Some(_tensor.get.producerName)
        network.setParameterTensor(this, _tensor.get)
      }
    }
    _tensor.get
  }

  def getTensor(network: Network): Tensor = {
    if (isManaged(network)) getManagedTensor(network) else getConstantTensor(network)
  }

  def isManaged(network: Network): Boolean = {
    preferManaged && network.pluginConfig.manageWeights
  }

  def getValue: Tensor = getTensor(defaultNet())

  def setValue(v: DenseMatrix[Double]): Unit = {
    require(v.size == getShape, s"Shape mismatch: expected ${getShape}, got ${v.size}")
    value = Some(v)
  }

  def isInitialized: Boolean = value.isDefined

  def getRawValue: DenseMatrix[Double] = {
    if (value.isEmpty) {
      value = Some(DenseMatrix.zeros[Double](getShape: _*))
      Parameter.xavierInit(value.get)
    }
    value.get
  }

  def setName(name: String, network: Network): Boolean = {
    _name = Some(name)
    if (isManaged(network)) {
      getWeights(network).foreach(_.name = name)
      true
    } else {
      network.trtNetwork.setWeightsName(getWeights(network).get, name)
    }
  }

  private def getWeights(network: Network): Option[Any] = {
    val tensor = network.getParameterTensor(this)
    if (isManaged(network)) {
      tensor
    } else if (tensor.isDefined) {
      Some(tensor.get.producer.weights)
    } else {
      None
    }
  }
}

object Parameter {
  def xavierInit(weights: DenseMatrix[Double]): Unit = {
    val shape = weights.size
    val vRange = if (shape.length == 2) {
      math.sqrt(6) / math.sqrt(shape(0) + shape(1))
    } else {
      0.1
    }

    val randMatrix = DenseMatrix.rand[Double](shape(0), shape(1)) * 2 - 1
    weights := randMatrix * vRange
  }
}
