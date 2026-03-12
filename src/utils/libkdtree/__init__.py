try:
    from .pykdtree.kdtree import KDTree
except ImportError:
    from pykdtree.kdtree import KDTree


__all__ = [
    KDTree
]
