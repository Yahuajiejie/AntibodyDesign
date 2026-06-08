"""
AffinityTransformer —  antigen-aware affinity modeling package.


"""

from .config import V3Config, cfg

__all__ = [
    "V3Config",
    "RegistryBuildResult",
    "build_registry",
    "build_registry_from_paths",
    "cfg",
    "write_registry_result",
]
# all 代表当前module可以被外部调用哪些函数

def __getattr__(name: str):
    """
    按需加载 registry 相关功能。
    平时用户可能只是想读取配置，例如：
        from AffinityTransformer import cfg
    这时不需要加载 pandas、openpyxl 这些比较重的库。

    但是 registry 构建功能会用到表格处理，因此可能依赖 pandas/openpyxl。
    所以这里不在包一开始 import 的时候就加载它们，而是等用户真的访问
    build_registry、write_registry_result 等函数时，再去导入 registry.workflow。

    输入：
      name: 访问的属性名。

    输出：
      对应对象。
    """
    if name in {
        "RegistryBuildResult",
        "build_registry",
        "build_registry_from_paths",
        "write_registry_result",
    }:
        from .registry import workflow

        return getattr(workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
