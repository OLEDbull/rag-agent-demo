"""
为整个工程提供统一的绝对路径
"""

import os

def get_project_root():
    """
    获取项目根目录
    """
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在目录
    current_dir = os.path.dirname(current_file_path)
    # 获取项目根目录
    project_root = os.path.dirname(current_dir)
    return project_root

def get_abs_path(relative_path):
    """
    获取绝对路径
    :param relative_path: 相对路径
    :return: 绝对路径
    """
    project_root = get_project_root()
    abs_path = os.path.join(project_root, relative_path)
    return abs_path