import argparse
from pathlib import Path
import sys
import logging
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.runner import StrategyRunner

def setup_logger():
    """设置日志器"""
    logger = logging.getLogger('WeeklySignal')
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建文件处理器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f'weekly_signal_{timestamp}.log')
    file_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行每周信号生成任务')
    parser.add_argument('--config', type=str, default='configs/template.yaml',
                      help='配置文件路径')
    parser.add_argument('--date', type=str, default=None,
                      help='信号生成日期 (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger()
    logger.info("开始执行每周信号生成任务")
    
    try:
        # 创建输出目录
        output_dir = project_root / 'output'
        output_dir.mkdir(exist_ok=True)
        
        # 运行策略
        runner = StrategyRunner(args.config)
        runner.run()
        
        logger.info("每周信号生成任务执行完成")
        
    except Exception as e:
        logger.error(f"任务执行出错: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 