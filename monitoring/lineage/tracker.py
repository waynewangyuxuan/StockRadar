from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Set, Optional
from enum import Enum

class OperationType(Enum):
    READ = "READ"
    WRITE = "WRITE"
    TRANSFORM = "TRANSFORM"
    VALIDATE = "VALIDATE"
    CLEAN = "CLEAN"

@dataclass
class DataNode:
    """数据节点"""
    id: str
    name: str
    type: str  # 例如: "table", "file", "api"
    metadata: Dict[str, str]

@dataclass
class Operation:
    """数据操作"""
    type: OperationType
    timestamp: datetime
    operator: str  # 执行操作的组件名称
    details: Dict[str, str]

@dataclass
class LineageEdge:
    """血缘关系边"""
    source: DataNode
    target: DataNode
    operation: Operation

class LineageTracker:
    """数据血缘追踪器"""
    
    def __init__(self):
        self._nodes: Dict[str, DataNode] = {}
        self._edges: List[LineageEdge] = []

    def add_node(self, node: DataNode) -> None:
        """添加数据节点"""
        self._nodes[node.id] = node

    def add_edge(self, source_id: str, target_id: str, operation: Operation) -> None:
        """添加血缘关系"""
        source = self._nodes.get(source_id)
        target = self._nodes.get(target_id)
        
        if not source or not target:
            raise ValueError(f"Source or target node not found: {source_id} -> {target_id}")
            
        edge = LineageEdge(source, target, operation)
        self._edges.append(edge)

    def get_upstream_nodes(self, node_id: str) -> Set[DataNode]:
        """获取上游节点"""
        result = set()
        for edge in self._edges:
            if edge.target.id == node_id:
                result.add(edge.source)
        return result

    def get_downstream_nodes(self, node_id: str) -> Set[DataNode]:
        """获取下游节点"""
        result = set()
        for edge in self._edges:
            if edge.source.id == node_id:
                result.add(edge.target)
        return result

    def get_node_operations(self, node_id: str) -> List[Operation]:
        """获取节点相关的所有操作"""
        operations = []
        for edge in self._edges:
            if edge.source.id == node_id or edge.target.id == node_id:
                operations.append(edge.operation)
        return operations

    def export_graph(self) -> Dict:
        """导出血缘图数据"""
        return {
            "nodes": list(self._nodes.values()),
            "edges": self._edges
        } 