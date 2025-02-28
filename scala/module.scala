import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.control.Breaks._
import scala.language.implicitConversions

object ModuleTranslation {

  import org.slf4j.LoggerFactory
  private val logger = LoggerFactory.getLogger(this.getClass)

  object _common {
    def default_net(): Any = {}
  }

  object parameter {
    class Parameter {}
  }

  private def _addindent(s_ : String, numSpaces : Int) : String = {
    val s = s_.split('\n').toList
    if (s.length == 1) {
      return s_
    }
    val first = s.head
    val rest = s.tail.map(line => (numSpaces * ' ') + line)
    val joinedRest = rest.mkString("\n")
    first + "\n" + joinedRest
  }

  class Module {
    private val _modules = mutable.Map[String, Module]()
    private val _parameters = mutable.Map[String, Any]()
    private val _network_outputs = mutable.Map[String, Any]()

    def forward(args : Any*): Any = {
      throw new NotImplementedError
    }

    def apply(args: Any*)(implicit current_net: Any = _common.default_net()): Any = {
      val currentNet = current_net
      val moduleCallStack = currentNet.asInstanceOf[{ def _module_call_stack: Any }]._module_call_stack

      if (!moduleCallStack.asInstanceOf[{ def module_names_set(): Boolean }].module_names_set()) {
          logger.debug("Initializing top level module")
          moduleCallStack.asInstanceOf[{ def set_module_names(module: Module): Unit }].set_module_names(this)
      }

      val uniqueName = moduleCallStack.asInstanceOf[{ def get_mod_name(module: Module): String }].get_mod_name(this)

      val stack = moduleCallStack.asInstanceOf[{ def call_stack_mgr(): { def append(name: String): Unit; def close(): Unit } }].call_stack_mgr()
      try {
        stack.append(uniqueName)
        val startLayerIdx = currentNet.asInstanceOf[{ def trt_network: { def num_layers: Int } }].trt_network.num_layers
        val output = forward(args: _*)
        val endLayerIdx = currentNet.asInstanceOf[{ def trt_network: { def num_layers: Int } }].trt_network.num_layers

        moduleCallStack.asInstanceOf[{ def set_layer_range(module: Module, range: Range): Unit }].set_layer_range(
          this, startLayerIdx to endLayerIdx)
        output
      } finally {
        stack.close()
      }
    }

    def getAttr(name: String): Any = {
      if (_parameters.contains(name)) {
        return _parameters(name)
      }

      if (_modules.contains(name)) {
        return _modules(name)
      }

      throw new NoSuchElementException(s"'${this.getClass.getName}' object has no attribute '$name'")
    }

    def setAttr(name: String, value: Any): Unit = {
      try {
        this.getClass.getMethod(name)
        if (value.isInstanceOf[parameter.Parameter]) {
          _modules.remove(name)
          _parameters(name) = value
        } else if (value.isInstanceOf[Module]) {
          _parameters.remove(name)
          _modules(name) = value
        } else {
          this.getClass.getMethods.filter(_.getName() == name).head.invoke(this, value.asInstanceOf[AnyRef])
        }
      } catch {
        case _: java.lang.NoSuchMethodException =>
          if (value.isInstanceOf[parameter.Parameter]) {
            _parameters(name) = value
          } else if (value.isInstanceOf[Module]) {
            _modules(name) = value
          } else {
            this.getClass.getMethods.filter(_.getName() == name).head.invoke(this, value.asInstanceOf[AnyRef])
          }
      }
    }

    def named_modules(memo: Option[mutable.Set[Module]] = None, prefix: String = "", remove_duplicate: Boolean = true): Iterable[(String, Module)] = {
      val _memo = memo.getOrElse(mutable.Set[Module]())
      if (!_memo.contains(this)) {
        if (remove_duplicate) {
          _memo.add(this)
        }
        Iterator((prefix, this)) ++ _modules.flatMap { case (name, module) =>
          if (module == null) Iterator.empty
          else {
            val submodule_prefix = prefix + (if (prefix.isEmpty) "" else ".") + name
            module.named_modules(Some(_memo), submodule_prefix, remove_duplicate)
          }
        }
      } else {
        Iterator.empty
      }
    }

    def named_modules_with_parent(memo: Option[mutable.Set[Module]] = None, prefix: String = "", parent: Module = null, remove_duplicate: Boolean = true): Iterable[(String, Module, Module)] = {
      val _memo = memo.getOrElse(mutable.Set[Module]())
      if (!_memo.contains(this)) {
        if (remove_duplicate) {
          _memo.add(this)
        }
        Iterator((prefix, this, parent)) ++ _modules.flatMap { case (name, module) =>
          if (module == null) Iterator.empty
          else {
            val submodule_prefix = prefix + (if (prefix.isEmpty) "" else ".") + name
            module.named_modules_with_parent(Some(_memo), submodule_prefix, this, remove_duplicate)
          }
        }
      } else {
        Iterator.empty
      }
    }

    def named_children(): Iterable[(String, Module)] = {
      val memo = mutable.Set[Module]()
      _modules.iterator.filter(_._2 != null).filter(t => !memo.contains(t._2)).map { case (name, module) =>
        memo.add(module)
        (name, module)
      }.toIterable
    }

    private def _named_members(get_members_fn: Module => Iterable[(String, Any)], prefix: String = "", recurse: Boolean = true): Iterable[(String, Any)] = {
      val memo = mutable.Set[Any]()
      val modules = if (recurse) named_modules(prefix = prefix).toList else List((prefix, this))
      modules.flatMap { case (module_prefix, module) =>
        val members = get_members_fn(module)
        members.filter { case (k, v) => v != null && !memo.contains(v) }.map { case (k, v) =>
          memo.add(v)
          val name = module_prefix + (if (module_prefix.isEmpty) "" else ".") + k
          (name, v)
        }
      }
    }

    def parameters(recurse: Boolean = true): Iterable[Any] = {
      named_parameters(recurse = recurse).map(_._2)
    }

    def named_parameters(prefix: String = "", recurse: Boolean = true): Iterable[(String, Any)] = {
      _named_members(module => module._parameters.toList, prefix = prefix, recurse = recurse)
    }

    def children(): Iterable[Module] = {
      named_children().map(_._2)
    }

    def apply(fn: Module => Unit): Module = {
      children().foreach(_.apply(fn))
      fn(this)
      this
    }

    def _get_name(): String = {
      this.getClass.getName.split('.').last
    }

    def register_parameter(name: String, param: Any): Unit = {
      _parameters(name) = param
    }

    def register_network_output(name: String, value: Any): Unit = {
      _network_outputs(name) = value
    }

    def named_network_outputs(): Iterable[(String, Any)] = {
      named_modules().flatMap { case (name, module) =>
        module._network_outputs.map { case (n, output) =>
          (name + (if (name.isEmpty) "" else ".") + n, output)
        }
      }
    }

    def update_parameters(torch_module: Any): Unit = {
      val m = named_parameters().toMap
      val tm = torch_module.asInstanceOf[{ def named_parameters(): Iterable[(String, Any)] }].named_parameters().toMap

      assert(m.keys.toList.sorted == tm.keys.toList.sorted, "The parameter names of the tensorrt-llm module must be the same with the torch module")

      named_parameters().foreach { case (k, v) =>
        v.asInstanceOf[{ def value: Any }].value = tm(k).asInstanceOf[{ def detach(): Any }].detach().asInstanceOf[{ def cpu(): Any }].cpu().asInstanceOf[{ def numpy(): Any }].numpy()
      }
    }

    override def toString: String = {
      val child_lines = _modules.map { case (key, module) =>
        val mod_str = module.toString
        val indented_mod_str = _addindent(mod_str, 2)
        s"($key): $indented_mod_str"
      }.toList

      var main_str = _get_name() + "("
      if (child_lines.nonEmpty) {
        main_str += "\n  " + child_lines.mkString("\n  ") + "\n"
      }
      main_str += ")"
      main_str
    }

    implicit def getModule(name: String): Module = _modules(name)
    implicit def getParameter(name: String): Any = _parameters(name)
  }

  class ModuleList(modules: Seq[Module]) extends Module {
    (0 until modules.length).foreach(i => _modules(i.toString) = modules(i))

    private def _get_abs_string_index(idx: Int): String = {
      val _idx = idx
      if (!(-this.length <= _idx && _idx < this.length)) {
        throw new IndexOutOfBoundsException(s"index $idx is out of range")
      }
      if (_idx < 0) {
        (_idx + this.length).toString
      } else {
        _idx.toString
      }
    }

    def getItem(idx: Int): Any = {
      if (idx < 0 || idx >= this.length) {
        throw new IndexOutOfBoundsException(s"Index $idx is out of range")
      }
      _modules(_get_abs_string_index(idx))
    }

    def setItem(idx: Int, module: Module): Unit = {
      val _idx = _get_abs_string_index(idx)
      setAttr(_idx, module)
    }

    def length: Int = {
      _modules.size
    }

    override def toString: String = {
      val list_of_reprs = (0 until length).map(i => this.getItem(i).toString).toList
      s"[$list_of_reprs.mkString(", ")]"
    }
  }
}
