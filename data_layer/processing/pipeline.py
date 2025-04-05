from typing import Any, Dict, List, Optional
import pandas as pd
from datetime import datetime

from .base import DataProcessor

class ProcessingPipeline:
    """Pipeline for chaining multiple data processors"""
    
    def __init__(
        self,
        name: str = "data_processing_pipeline",
        description: Optional[str] = None
    ):
        """Initialize processing pipeline
        
        Args:
            name: Name of the pipeline
            description: Optional description
        """
        self.name = name
        self.description = description or f"Data processing pipeline: {name}"
        self.processors: List[DataProcessor] = []
        self.metrics = {
            'total_rows_processed': 0,
            'total_errors': 0,
            'total_processing_time': 0.0,
            'processor_metrics': {}
        }
    
    def add_processor(self, processor: DataProcessor) -> None:
        """Add a processor to the pipeline
        
        Args:
            processor: Processor to add
        """
        self.processors.append(processor)
    
    def remove_processor(self, processor_name: str) -> None:
        """Remove a processor from the pipeline
        
        Args:
            processor_name: Name of the processor to remove
        """
        self.processors = [p for p in self.processors if p.name != processor_name]
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data through all processors in the pipeline
        
        Args:
            data: Input DataFrame to process
            
        Returns:
            Processed DataFrame
        """
        start_time = datetime.now()
        current_data = data.copy()
        
        try:
            # Process through each processor
            for processor in self.processors:
                if not processor.enabled:
                    continue
                
                # Validate data before processing
                processor.validate(current_data)
                
                # Process data
                current_data = processor.process(current_data)
                
                # Collect processor metrics
                self.metrics['processor_metrics'][processor.name] = processor.get_metrics()
            
            # Update pipeline metrics
            self.metrics['total_rows_processed'] = len(current_data)
            self.metrics['total_processing_time'] = (datetime.now() - start_time).total_seconds()
            
            return current_data
            
        except Exception as e:
            self.metrics['total_errors'] += 1
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics
        
        Returns:
            Dict containing pipeline metrics
        """
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset pipeline metrics"""
        self.metrics = {
            'total_rows_processed': 0,
            'total_errors': 0,
            'total_processing_time': 0.0,
            'processor_metrics': {}
        }
        
        # Reset metrics for all processors
        for processor in self.processors:
            processor.reset_metrics()
    
    def __str__(self) -> str:
        """String representation of the pipeline"""
        processor_names = [p.name for p in self.processors]
        return f"{self.name} (processors: {', '.join(processor_names)})" 