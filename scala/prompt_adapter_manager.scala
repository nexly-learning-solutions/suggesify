import torch._
import scala.collection.mutable

private var _uidCounter: Int = 0
private val _uidToWeights: mutable.Map[String, Tensor] = mutable.Map[String, Tensor]()

def loadFromCkpt(modelDirs: List[String], modelConfig: ModelConfig, uids: Option[List[String]] = None): Unit = {
  var uidsToUse = uids.getOrElse(modelDirs.map(_ => generateUid()))

  assert(uidsToUse.length == modelDirs.length)

  val (newUids, newModelDirs) = modelDirs.zip(uidsToUse).foldLeft((List[String](), List[String]())) {
    case ((newUidsAcc, newModelDirsAcc), (modelDir, uid)) =>
      if (_uidToWeights.contains(uid)) {
        (newUidsAcc, newModelDirsAcc)
      } else {
        (newUidsAcc :+ uid, newModelDirsAcc :+ modelDir)
      }
  }

  if (newUids.nonEmpty && newModelDirs.nonEmpty) {
    newUids.zip(newModelDirs).foreach {
      case (uid, modelDir) =>
        val stateDict = loadStateDict(getModelPath(modelDir, "adapter_model"))
        _uidToWeights(uid) = stateDict("prompt_embeddings").to(strDtypeToTorch(modelConfig.dtype))
    }
  }
}

def uidToWeights: Map[String, Tensor] = _uidToWeights.toMap

private def generateUid(): String = {
  var uid = _uidCounter.toString
  while (_uidToWeights.contains(uid)) {
    _uidCounter += 1
    uid = _uidCounter.toString
  }
  _uidCounter += 1
  uid
}
