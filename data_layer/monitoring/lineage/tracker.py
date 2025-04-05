from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from enum import Enum
import pandas as pd

class OperationType(Enum):
    READ = "READ"
    WRITE = "WRITE"
    TRANSFORM = "TRANSFORM"
    VALIDATE = "VALIDATE"
    CLEAN = "CLEAN"

@dataclass
class DataNode:
    """Data node"""
    id: str
    name: str
    type: str  # e.g., "table", "file", "api"
    metadata: Dict[str, str]

@dataclass
class Operation:
    """Data operation"""
    type: OperationType
    timestamp: datetime
    operator: str  # Name of the component performing the operation
    details: Dict[str, str]

@dataclass
class LineageEdge:
    """Lineage relationship edge"""
    source: DataNode
    target: DataNode
    operation: Operation

class LineageTracker:
    """Data lineage tracker"""

    def __init__(self):
        """Initialize the data lineage tracker"""
        self.nodes = {}
        self.edges = []

    def create_source_node(self, source_id: str, source_name: str, metadata: Dict[str, Any] = None) -> DataNode:
        """Create source node"""
        node = DataNode(
            id=source_id,
            name=source_name,
            type="api",
            metadata=metadata
        )
        self.nodes[source_id] = node
        return node

    def record_operation(
        self,
        source_node: DataNode,
        operation: str,
        input_data: Dict[str, Any],
        output_data: pd.DataFrame
    ) -> None:
        """Record data operation"""
        # Create operation node
        operation_node = DataNode(
            id=f"{source_node.id}_{operation}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=operation,
            type="operation",
            metadata=input_data
        )
        self.nodes[operation_node.id] = operation_node

        # Create output node
        output_node = DataNode(
            id=f"{operation_node.id}_output",
            name="output",
            type="data",
            metadata={
                "shape": output_data.shape,
                "columns": list(output_data.columns)
            }
        )
        self.nodes[output_node.id] = output_node

        # Add edges
        self.edges.append((source_node.id, operation_node.id))
        self.edges.append((operation_node.id, output_node.id))

    def add_node(self, node: DataNode) -> None:
        """Add data node"""
        self.nodes[node.id] = node

    def add_edge(self, source_id: str, target_id: str, operation: Operation) -> None:
        """Add lineage relationship"""
        source = self.nodes.get(source_id)
        target = self.nodes.get(target_id)
        
        if not source or not target:
            raise ValueError(f"Source or target node not found: {source_id} -> {target_id}")
            
        edge = LineageEdge(source, target, operation)
        self.edges.append((source_id, target_id))

    def get_upstream_nodes(self, node_id: str) -> Set[DataNode]:
        """Get upstream nodes"""
        result = set()
        for edge in self.edges:
            if edge[1] == node_id:
                result.add(self.nodes[edge[0]])
        return result

    def get_downstream_nodes(self, node_id: str) -> Set[DataNode]:
        """Get downstream nodes"""
        result = set()
        for edge in self.edges:
            if edge[0] == node_id:
                result.add(self.nodes[edge[1]])
        return result

    def get_node_operations(self, node_id: str) -> List[Operation]:
        """Get all operations related to the node"""
        operations = []
        for edge in self.edges:
            if edge[0] == node_id or edge[1] == node_id:
                operations.append(Operation(
                    type=OperationType.READ,
                    timestamp=datetime.now(),
                    operator="",
                    details={}
                ))
        return operations

    def export_graph(self) -> Dict:
        """Export lineage graph data"""
        return {
            "nodes": list(self.nodes.values()),
            "edges": self.edges
        } 