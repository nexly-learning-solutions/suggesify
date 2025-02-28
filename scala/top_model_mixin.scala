import scala.collection.mutable
import scala.util.Try

class LoraConfig
class Mapping
class PluginConfig {
  def fromDict(kwargs: mutable.Map[String, Any]): PluginConfig = {
    new PluginConfig()
  }
}

trait TopModelMixin {
  
  def fromHuggingFace(
    hfModelDir: String,
    dtype: Option[String] = Some("float16"),
    mapping: Option[Mapping] = None,
    kwargs: mutable.Map[String, Any] = mutable.Map()
  ): Unit = {
    throw new NotImplementedError("Subclass shall override this")
  }

  def useLora(loraConfig: LoraConfig): Unit = {
    throw new NotImplementedError("Subclass shall override this")
  }

  def usePromptTuning(
    maxPromptEmbeddingTableSize: String,
    promptTablePath: String
  ): Unit = {
    throw new NotImplementedError("Subclass shall override this")
  }

  def defaultPluginConfig(kwargs: mutable.Map[String, Any] = mutable.Map()): PluginConfig = {
    new PluginConfig().fromDict(kwargs)
  }
}
