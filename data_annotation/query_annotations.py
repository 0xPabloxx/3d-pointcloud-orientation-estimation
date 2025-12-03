#!/usr/bin/env python3
"""
快速查询辅助函数

使用方法:
    from query_annotations import AnnotationQuery

    query = AnnotationQuery('symmetry_annotations_indexed.json')

    # 查询某个category
    files = query.get_by_category('airplane')

    # 查询某个K值
    files = query.get_by_K(1)

    # 查询需要矫正的
    files = query.get_need_correction()
"""

import json

class AnnotationQuery:
    def __init__(self, indexed_json_file):
        with open(indexed_json_file, 'r') as f:
            self.data = json.load(f)
        self.annotations = self.data['annotations']
        self.indices = self.data['indices']
        self.stats = self.data['stats']

    def get_by_category(self, category):
        """获取某个category的所有文件"""
        return self.indices['by_category'].get(category, [])

    def get_by_K(self, K):
        """获取某个K值的所有文件"""
        return self.indices['by_K'].get(str(K), [])

    def get_need_correction(self):
        """获取所有需要矫正的文件"""
        return self.indices['need_correction']

    def get_aligned(self):
        """获取所有已对齐的文件"""
        return self.indices['by_aligned']['true']

    def get_annotation(self, file_path):
        """获取某个文件的标注"""
        return self.annotations.get(file_path)

    def get_stats(self):
        """获取统计信息"""
        return self.stats
