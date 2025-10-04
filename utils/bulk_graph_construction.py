#!/usr/bin/env python3
"""
Bulk数据集静态图构建脚本 - 使用预分割patch - 完整新逻辑版本
基于bulk特征数据和预分割patch构建图结构，支持后续的迁移学习

特征文件格式：
- train/*.parquet: 训练集特征文件
- test/*.parquet: 测试集特征文件
- 每个文件包含128维DINO特征和细胞位置信息

Patch文件格式：
- patches_dir/{patient_id}/*.png: 每个患者的预分割patch文件
- 文件名格式: {patient_id}_patch_tile_{tile_id}_level0_{x1}-{y1}-{x2}-{y2}.png
"""

import os
import pandas as pd
import numpy as np
import torch
import json
import pickle
import re
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class BulkStaticGraphBuilder:
    """Bulk数据集静态图构建器 - 使用预分割patch"""
    
    def __init__(self, 
                 train_features_dir,
                 test_features_dir,
                 bulk_csv_path,
                 patches_dir,                        # 预分割的patch目录
                 wsi_input_dir,                      # 原始WSI文件目录
                 intra_patch_distance_threshold=250,  # patch内细胞连接距离阈值（像素）
                 inter_patch_k_neighbors=6,          # patch间k近邻数量
                 use_deep_features=True,             # 使用深度特征
                 feature_dim=128,                    # 特征维度
                 max_cells_per_patch=None):          # 每个patch的最大细胞数
            
        self.train_features_dir = train_features_dir
        self.test_features_dir = test_features_dir
        self.bulk_csv_path = bulk_csv_path
        self.patches_dir = patches_dir
        self.wsi_input_dir = wsi_input_dir
        self.intra_patch_distance_threshold = intra_patch_distance_threshold
        self.inter_patch_k_neighbors = inter_patch_k_neighbors
        self.use_deep_features = use_deep_features
        self.feature_dim = feature_dim
        self.max_cells_per_patch = max_cells_per_patch
        
        self.processed_data = {}
        self.bulk_data = None
        self.valid_patient_ids = []
        
    def load_bulk_data(self):
        """加载bulk RNA-seq数据"""
        print("=== 加载bulk RNA-seq数据 ===")
        
        bulk_df = pd.read_csv(self.bulk_csv_path)
        bulk_df["gene_name"] = bulk_df['Unnamed: 0'].str[:15]
        bulk_df = bulk_df.drop(columns=['Unnamed: 0'])
        bulk_df = bulk_df.set_index('gene_name')
        
        original_ids = list(bulk_df.columns)
        patient_ids = [pid[:19] for pid in original_ids]
        patient_id_series = pd.Series(patient_ids)
        
        # 去重处理
        duplicate_ids = patient_id_series[patient_id_series.duplicated()].unique()
        print(f"发现重复患者ID: {len(duplicate_ids)}")
        
        valid_patient_ids = patient_id_series[~patient_id_series.isin(duplicate_ids)].unique()
        valid_original_ids = [oid for oid in original_ids if oid[:19] in valid_patient_ids]
        bulk_df = bulk_df[valid_original_ids]
        
        self.bulk_data = bulk_df
        self.valid_patient_ids = valid_patient_ids
        
        print(f"Bulk数据形状: {bulk_df.shape}")
        print(f"有效患者ID数: {len(valid_patient_ids)}")
        
    def extract_slide_id(self, file_path):
        """从文件路径或文件名提取切片ID（包括UUID）"""
        basename = os.path.basename(file_path)
        # 从文件名中提取完整的切片标识符
        # 例如：TCGA-AA-3872-01A-01-TS1.4f7d5598-e36a-4e30-9b7b-ab55cc6fc3a0_tile36_features.parquet
        # 提取：TCGA-AA-3872-01A-01-TS1.4f7d5598-e36a-4e30-9b7b-ab55cc6fc3a0
        if '_tile36_features.parquet' in basename:
            return basename.replace('_tile36_features.parquet', '')
        elif '_patch_tile_' in basename:
            # patch文件格式：TCGA-AA-3872-01A-01-BS1.e29045b5-113d-4dba-b03b-ba2e0d82a388_patch_tile_542_level0_5540-10952-5796-11208.png
            # 提取：TCGA-AA-3872-01A-01-BS1.e29045b5-113d-4dba-b03b-ba2e0d82a388
            parts = basename.split('_patch_tile_')
            if len(parts) >= 2:
                return parts[0]
        return basename
    
    def extract_patient_id_from_slide(self, slide_id):
        """从切片ID提取患者ID"""
        # 从 TCGA-AA-3872-01A-01-TS1.4f7d5598-e36a-4e30-9b7b-ab55cc6fc3a0 
        # 提取 TCGA-AA-3872-01A-01
        parts = slide_id.split('-')
        if len(parts) >= 4:
            return '-'.join(parts[:4]) + '-01'
        return slide_id[:19]
        
    def find_patch_files_by_slide(self, slide_id):
        """根据切片ID查找对应的patch文件"""
        # 直接在patches目录下搜索包含相同slide_id的patch文件
        matching_patch_files = []
        
        # 搜索所有子目录
        for root, dirs, files in os.walk(self.patches_dir):
            for file in files:
                if file.endswith(".png") and "_patch_tile_" in file:
                    file_slide_id = self.extract_slide_id(file)
                    if slide_id == file_slide_id:
                        matching_patch_files.append(os.path.join(root, file))
        
        print(f"  - 切片 {slide_id} 找到 {len(matching_patch_files)} 个匹配的patch文件")
        return matching_patch_files
    
    def parse_patch_coordinates(self, patch_filename):
        """从patch文件名解析坐标信息"""
        # 文件名格式: {patient_id}_patch_tile_{tile_id}_level0_{x1}-{y1}-{x2}-{y2}.png
        basename = os.path.basename(patch_filename)
        
        # 使用正则表达式提取坐标
        pattern = r'_patch_tile_(\d+)_level0_(\d+)-(\d+)-(\d+)-(\d+)\.png'
        match = re.search(pattern, basename)
        
        if match:
            tile_id = int(match.group(1))
            x1, y1, x2, y2 = map(int, match.groups()[1:])
            return {
                'tile_id': tile_id,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2,
                'width': x2 - x1,
                'height': y2 - y1
            }
        else:
            print(f"警告: 无法解析patch文件名: {basename}")
            return None
    
    def convert_to_absolute_coordinates(self, df):
        """将细胞相对坐标转换为WSI绝对坐标"""
        def parse_tile_coordinates(image_name):
            """从image_name解析tile的绝对坐标"""
            # 示例：TCGA-AA-3844-01A-01-BS1.xxx_patch_tile_1435_level0_12800-8727-13056-8983
            pattern = r'_patch_tile_\d+_level0_(\d+)-(\d+)-(\d+)-(\d+)$'
            match = re.search(pattern, image_name)
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                return x1, y1, x2, y2
            return None, None, None, None
        
        # 解析每个细胞所属tile的绝对坐标
        tile_coords = df['image_name'].apply(parse_tile_coordinates)
        
        # 将结果转换为单独的列
        df[['tile_x1', 'tile_y1', 'tile_x2', 'tile_y2']] = pd.DataFrame(tile_coords.tolist(), index=df.index)
        
        # 计算绝对坐标：tile起始坐标 + 细胞相对坐标
        df['abs_x'] = df['tile_x1'] + df['x']
        df['abs_y'] = df['tile_y1'] + df['y']
        
        # 用绝对坐标替换相对坐标
        df['x'] = df['abs_x']
        df['y'] = df['abs_y']
        
        # 清理临时列
        df = df.drop(columns=['tile_x1', 'tile_y1', 'tile_x2', 'tile_y2', 'abs_x', 'abs_y', 'image_name'])
        
        return df
    
    def assign_cells_to_patches(self, cells_df, patch_files):
        """将细胞分配到相应的patch"""
        print(f"  - 将 {len(cells_df)} 个细胞分配到 {len(patch_files)} 个patch")
        
        patches = []
        cells_df = cells_df.copy()
        cells_df['patch_id'] = -1
        
        for patch_file in patch_files:
            patch_coords = self.parse_patch_coordinates(patch_file)
            if patch_coords is None:
                continue
            
            # 找到在当前patch范围内的细胞
            patch_mask = ((cells_df['x'] >= patch_coords['x1']) & 
                         (cells_df['x'] < patch_coords['x2']) & 
                         (cells_df['y'] >= patch_coords['y1']) & 
                         (cells_df['y'] < patch_coords['y2']))
            
            patch_cells = cells_df[patch_mask].copy()
            
            if len(patch_cells) > 0:
                patch_id = patch_coords['tile_id']
                cells_df.loc[patch_mask, 'patch_id'] = patch_id
                
                # 调整细胞坐标到patch内的相对位置
                patch_cells_relative = patch_cells.copy()
                patch_cells_relative['x'] = patch_cells['x'] - patch_coords['x1']
                patch_cells_relative['y'] = patch_cells['y'] - patch_coords['y1']
                
                patches.append({
                    'patch_id': patch_id,
                    'cells': patch_cells_relative,
                    'center': [patch_coords['center_x'], patch_coords['center_y']],
                    'bounds': [patch_coords['x1'], patch_coords['x2'], 
                              patch_coords['y1'], patch_coords['y2']],
                    'size': [patch_coords['width'], patch_coords['height']]
                })
        
        assigned_count = len(cells_df[cells_df['patch_id'] >= 0])
        print(f"  - 成功分配 {assigned_count}/{len(cells_df)} 个细胞到 {len(patches)} 个有效patch")
        
        return patches
    
    def build_intra_patch_graphs(self, patches):
        """构建patch内的图结构（基于细胞）"""
        intra_patch_graphs = {}
        
        for patch_info in tqdm(patches, desc="构建patch内图"):
            patch_id = patch_info['patch_id']
            patch_cells = patch_info['cells']
            
            if len(patch_cells) < 2:
                # 单个细胞的patch
                if len(patch_cells) == 1:
                    cell_row = patch_cells.iloc[0]
                    cell_features = self.extract_cell_feature_vector(cell_row)
                    x = torch.tensor([cell_features], dtype=torch.float32)
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    pos = torch.tensor([[cell_row['x'], cell_row['y']]], dtype=torch.float32)
                    
                    graph = Data(x=x, edge_index=edge_index, pos=pos)
                    intra_patch_graphs[patch_id] = graph
                continue
            
            # 提取位置和特征（使用patch内相对坐标）
            positions = patch_cells[['x', 'y']].values
            cell_features = np.array([
                self.extract_cell_feature_vector(row) 
                for _, row in patch_cells.iterrows()
            ])
            
            # 计算距离矩阵
            distances = squareform(pdist(positions))
            
            # 基于距离阈值构建邻接矩阵
            adj_matrix = (distances <= self.intra_patch_distance_threshold) & (distances > 0)
            
            # 转换为边列表
            edge_indices = np.where(adj_matrix)
            edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
            
            # 如果没有边，使用k近邻连接
            if edge_index.shape[1] == 0:
                k = min(3, len(patch_cells) - 1)
                if k > 0:
                    nbrs = NearestNeighbors(n_neighbors=k+1).fit(positions)
                    _, indices = nbrs.kneighbors(positions)
                    
                    edges = []
                    for i, neighbors in enumerate(indices):
                        for neighbor in neighbors[1:]:  # 跳过自己
                            edges.extend([[i, neighbor], [neighbor, i]])
                    
                    if edges:
                        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
                    else:
                        edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # 创建图数据
            x = torch.tensor(cell_features, dtype=torch.float32)
            pos = torch.tensor(positions, dtype=torch.float32)
            
            graph = Data(x=x, edge_index=edge_index, pos=pos)
            intra_patch_graphs[patch_id] = graph
        
        return intra_patch_graphs
    
    def build_inter_patch_graph(self, patches):
        """构建patch间的图结构"""
        if len(patches) < 2:
            # 只有一个patch的情况
            if len(patches) == 1:
                patch_center = patches[0]['center']
                patch_features = torch.tensor([patch_center], dtype=torch.float32)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                pos = torch.tensor([patch_center], dtype=torch.float32)
                return Data(x=patch_features, edge_index=edge_index, pos=pos)
            else:
                # 没有patch的情况
                return Data(x=torch.empty((0, 2)), edge_index=torch.empty((2, 0)), pos=torch.empty((0, 2)))
        
        # 获取patch中心点（使用原始WSI坐标）
        patch_positions = np.array([patch['center'] for patch in patches])
        
        # 使用k近邻构建patch间连接
        k = min(self.inter_patch_k_neighbors, len(patches) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(patch_positions)
        _, indices = nbrs.kneighbors(patch_positions)
        
        # 构建边列表
        edges = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:  # 跳过自己
                edges.extend([[i, neighbor], [neighbor, i]])
        
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)
        
        # patch特征：使用位置特征
        patch_features = torch.tensor(patch_positions, dtype=torch.float32)
        pos = torch.tensor(patch_positions, dtype=torch.float32)
        
        inter_patch_graph = Data(x=patch_features, edge_index=edge_index, pos=pos)
        
        return inter_patch_graph
    
    def extract_cell_feature_vector(self, cell_row):
        """提取细胞特征向量"""
        if self.use_deep_features:
            # 使用深度特征
            features = [cell_row[f'feature_{i}'] for i in range(self.feature_dim)]
            return np.array(features, dtype=np.float32)
        else:
            # 使用几何特征（如果需要的话）
            features = [
                cell_row['x'],
                cell_row['y'],
                cell_row.get('area', 100.0),
                cell_row.get('perimeter', 35.4),
            ]
            # 扩展到目标维度
            features = np.array(features, dtype=np.float32)
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)), mode='constant')
            return features[:self.feature_dim]
    
    def load_slide_features_from_dino_files(self, split='train'):
        """直接从DINO parquet文件加载所有切片的细胞特征数据 - 切片级别匹配"""
        print(f"=== 从DINO文件加载{split}集切片特征数据 ===")
        
        # 根据split选择目录
        if split == 'train':
            features_dir = self.train_features_dir
        else:
            features_dir = self.test_features_dir
        
        # 找到所有parquet文件
        feature_files = []
        for root, _, files in os.walk(features_dir):
            for file in files:
                if file.endswith(".parquet"):
                    full_path = os.path.join(root, file)
                    feature_files.append(full_path)
        
        print(f"找到 {len(feature_files)} 个{split}集特征文件")
        
        # 加载每个切片的数据
        slides_data = {}
        
        for feature_file in tqdm(feature_files, desc=f"加载{split}集DINO特征"):
            # 提取切片ID（包括UUID）
            slide_id = self.extract_slide_id(feature_file)
            
            # 提取患者ID，检查是否在有效患者列表中
            patient_id = self.extract_patient_id_from_slide(slide_id)
            if patient_id not in self.valid_patient_ids:
                continue
                
            # 加载parquet文件
            try:
                df = pd.read_parquet(feature_file)
                
                # 检查必要的列
                required_columns = [f'feature_{i}' for i in range(128)] + ['x', 'y', 'image_name', 'cluster_label']
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    print(f"警告: 文件 {feature_file} 缺少列: {missing_cols}")
                    continue
                
                # 转换坐标为绝对坐标
                df_processed = self.convert_to_absolute_coordinates(df.copy())
                
                slides_data[slide_id] = {
                    'slide_id': slide_id,
                    'patient_id': patient_id,
                    'cells_df': df_processed,
                    'num_cells': len(df_processed)
                }
                
                print(f"  加载切片 {slide_id} (患者 {patient_id}): {len(df_processed)} 个细胞")
                
            except Exception as e:
                print(f"错误: 无法加载 {feature_file}: {e}")
                continue
        
        print(f"成功加载 {len(slides_data)} 个{split}集切片的特征数据")
        return slides_data
    
    def process_all_slides_new_logic(self):
        """使用新逻辑处理所有切片数据：按切片级别匹配patch"""
        print("=== 使用切片级别匹配逻辑处理所有数据 ===")
        
        # 分别加载训练集和测试集的DINO特征数据（按切片）
        train_slides_data = self.load_slide_features_from_dino_files('train')
        test_slides_data = self.load_slide_features_from_dino_files('test')
        
        # 处理训练集
        print("\n处理训练集...")
        self.processed_data['train'] = {}
        for slide_id, slide_data in tqdm(train_slides_data.items(), desc="处理训练集切片"):
            result = self.process_single_slide_new_logic(slide_data['cells_df'], slide_id, slide_data['patient_id'])
            if result:
                self.processed_data['train'][slide_id] = result
        
        # 处理测试集
        print("\n处理测试集...")
        self.processed_data['test'] = {}
        for slide_id, slide_data in tqdm(test_slides_data.items(), desc="处理测试集切片"):
            result = self.process_single_slide_new_logic(slide_data['cells_df'], slide_id, slide_data['patient_id'])
            if result:
                self.processed_data['test'][slide_id] = result
        
        print(f"\n处理完成:")
        print(f"  - 训练集切片: {len(self.processed_data['train'])}")
        print(f"  - 测试集切片: {len(self.processed_data['test'])}")
        
        # 统计建图情况
        total_slides = len(self.processed_data['train']) + len(self.processed_data['test'])
        slides_with_graphs = 0
        slides_without_graphs = 0
        
        for split_data in [self.processed_data['train'], self.processed_data['test']]:
            for slide_data in split_data.values():
                if slide_data.get('has_graphs', False):
                    slides_with_graphs += 1
                else:
                    slides_without_graphs += 1
        
        print(f"\n建图统计:")
        print(f"  - 总切片数: {total_slides}")
        print(f"  - 成功建图切片: {slides_with_graphs}")
        print(f"  - 仅保留原始特征切片: {slides_without_graphs}")
        
    def process_single_slide_new_logic(self, cells_df, slide_id, patient_id):
        """处理单个切片数据 - 新逻辑：保证所有细胞数据都被保留"""
        print(f"处理切片: {slide_id} (患者: {patient_id})")
        
        if cells_df is None or len(cells_df) == 0:
            print(f"  - 警告: 切片 {slide_id} 没有细胞数据")
            return None
            
        print(f"  - 细胞数量: {len(cells_df)}")
        
        # 提取所有细胞的特征、坐标和聚类标签
        all_cell_features = self.extract_all_cell_features_with_clusters(cells_df)
        all_cell_positions = cells_df[['x', 'y']].values.astype(np.float32)
        cluster_labels = cells_df['cluster_label'].values
        
        # 尝试构建图结构（使用切片ID查找对应的patch）
        patch_files = self.find_patch_files_by_slide(slide_id)
        print(f"  - 匹配的Patch文件数量: {len(patch_files)}")
        
        has_graphs = False
        intra_patch_graphs = {}
        inter_patch_graph = Data(x=torch.empty((0, 2)), edge_index=torch.empty((2, 0)), pos=torch.empty((0, 2)))
        patches = []
        
        if len(patch_files) > 0:
            # 尝试将细胞分配到patch并构建图
            patches = self.assign_cells_to_patches(cells_df, patch_files)
            
            if len(patches) > 0:
                # 构建图结构
                intra_patch_graphs = self.build_intra_patch_graphs(patches)
                inter_patch_graph = self.build_inter_patch_graph(patches)
                has_graphs = True
                
                print(f"  - 成功构建图: Patch内图 {len(intra_patch_graphs)} 个")
                print(f"  - Patch间图: {inter_patch_graph.edge_index.shape[1]} 条边")
            else:
                print(f"  - 未能成功分配细胞到patch，将保留原始特征")
        else:
            print(f"  - 未找到匹配的patch文件，将保留原始特征")
        
        return {
            'slide_id': slide_id,
            'patient_id': patient_id,
            'cells_df': cells_df,
            'patches': patches,
            'intra_patch_graphs': intra_patch_graphs,
            'inter_patch_graph': inter_patch_graph,
            'bulk_expr': self.get_bulk_expression(patient_id),  # 仍然使用患者ID获取bulk表达
            'has_graphs': has_graphs,
            'all_cell_features': all_cell_features,          # 所有细胞的DINO特征
            'all_cell_positions': torch.tensor(all_cell_positions),  # 所有细胞的空间坐标
            'cluster_labels': torch.tensor(cluster_labels),         # 所有细胞的聚类标签
            'cell_to_graph_mapping': self.build_cell_to_graph_mapping(cells_df, patches) if has_graphs else None
        }
    
    def extract_all_cell_features_with_clusters(self, cells_df):
        """提取所有细胞的DINO特征"""
        if cells_df is None or len(cells_df) == 0:
            return torch.empty((0, self.feature_dim), dtype=torch.float32)
        
        # 提取DINO特征 (feature_0 到 feature_127)
        feature_columns = [f'feature_{i}' for i in range(128)]
        features_matrix = cells_df[feature_columns].values.astype(np.float32)
        
        return torch.tensor(features_matrix, dtype=torch.float32)
    
    def build_cell_to_graph_mapping(self, cells_df, patches):
        """构建细胞到图的映射关系"""
        if not patches:
            return None
            
        cell_to_graph = {}
        
        for patch_info in patches:
            patch_id = patch_info['patch_id']
            patch_cells = patch_info['cells']
            
            # 为这个patch中的每个细胞建立映射
            for cell_idx in patch_cells.index:
                cell_to_graph[cell_idx] = {
                    'patch_id': patch_id,
                    'has_graph': True
                }
        
        return cell_to_graph
    
    def get_bulk_expression(self, patient_id):
        """获取患者的bulk表达数据"""
        if self.bulk_data is None:
            return None
            
        # 找到匹配的列
        bulk_col = [col for col in self.bulk_data.columns if col[:19] == patient_id]
        if len(bulk_col) == 1:
            return self.bulk_data[bulk_col[0]].values.astype(np.float32)
        elif len(bulk_col) > 1:
            # 多个bulk列：取平均值
            print(f"信息: 患者 {patient_id} 有 {len(bulk_col)} 列bulk数据，使用平均值")
            print(f"  - 可用列: {bulk_col}")
            bulk_values = self.bulk_data[bulk_col].values.astype(np.float32)
            return np.mean(bulk_values, axis=1)
        else:
            print(f"警告: 未找到患者 {patient_id} 的bulk数据")
            return None
    
    def save_graphs_slide_logic(self, output_dir):
        """保存构建的图数据 - 切片逻辑：保存完整的细胞特征数据和切片映射"""
        print("=== 保存图结构和完整细胞数据（切片级别）===")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 分别保存训练集和测试集的图数据
        for split in ['train', 'test']:
            if split not in self.processed_data:
                continue
                
            split_data = self.processed_data[split]
            
            # 准备保存的数据
            intra_graphs = {}
            inter_graphs = {}
            bulk_expressions = {}
            all_cell_features = {}       # 所有细胞的DINO特征
            all_cell_positions = {}      # 所有细胞的空间坐标  
            cluster_labels = {}          # 所有细胞的聚类标签
            graph_status = {}            # 每个切片是否有图数据的状态
            cell_to_graph_mappings = {}  # 细胞到图的映射关系
            slide_to_patient_mapping = {} # 切片到患者的映射关系
            metadata = {}
            
            for slide_id, slide_data in split_data.items():
                intra_graphs[slide_id] = slide_data['intra_patch_graphs']
                inter_graphs[slide_id] = slide_data['inter_patch_graph']
                bulk_expressions[slide_id] = slide_data['bulk_expr']
                all_cell_features[slide_id] = slide_data['all_cell_features']
                all_cell_positions[slide_id] = slide_data['all_cell_positions']
                cluster_labels[slide_id] = slide_data['cluster_labels']
                graph_status[slide_id] = slide_data.get('has_graphs', False)
                cell_to_graph_mappings[slide_id] = slide_data.get('cell_to_graph_mapping', None)
                slide_to_patient_mapping[slide_id] = slide_data['patient_id']  # 保存切片到患者的映射
                
                metadata[slide_id] = {
                    'slide_id': slide_id,
                    'patient_id': slide_data['patient_id'],
                    'num_cells': len(slide_data['cells_df']),
                    'num_patches': len(slide_data['patches']),
                    'intra_graph_count': len(slide_data['intra_patch_graphs']),
                    'inter_graph_edges': slide_data['inter_patch_graph'].edge_index.shape[1],
                    'has_bulk_expr': slide_data['bulk_expr'] is not None,
                    'has_graphs': slide_data.get('has_graphs', False),
                    'total_cell_features': slide_data['all_cell_features'].shape[0],
                    'cell_feature_dim': slide_data['all_cell_features'].shape[1]
                }
            
            # 保存文件
            intra_path = os.path.join(output_dir, f"bulk_{split}_intra_patch_graphs.pkl")
            inter_path = os.path.join(output_dir, f"bulk_{split}_inter_patch_graphs.pkl")
            bulk_path = os.path.join(output_dir, f"bulk_{split}_expressions.pkl")
            features_path = os.path.join(output_dir, f"bulk_{split}_all_cell_features.pkl")
            positions_path = os.path.join(output_dir, f"bulk_{split}_all_cell_positions.pkl")
            clusters_path = os.path.join(output_dir, f"bulk_{split}_cluster_labels.pkl")
            status_path = os.path.join(output_dir, f"bulk_{split}_graph_status.pkl")
            mappings_path = os.path.join(output_dir, f"bulk_{split}_cell_to_graph_mappings.pkl")
            slide_mappings_path = os.path.join(output_dir, f"bulk_{split}_slide_to_patient_mapping.pkl")  # 新增
            metadata_path = os.path.join(output_dir, f"bulk_{split}_metadata.json")
            
            with open(intra_path, 'wb') as f:
                pickle.dump(intra_graphs, f)
            
            with open(inter_path, 'wb') as f:
                pickle.dump(inter_graphs, f)
            
            with open(bulk_path, 'wb') as f:
                pickle.dump(bulk_expressions, f)
                
            with open(features_path, 'wb') as f:
                pickle.dump(all_cell_features, f)
                
            with open(positions_path, 'wb') as f:
                pickle.dump(all_cell_positions, f)
                
            with open(clusters_path, 'wb') as f:
                pickle.dump(cluster_labels, f)
                
            with open(status_path, 'wb') as f:
                pickle.dump(graph_status, f)
                
            with open(mappings_path, 'wb') as f:
                pickle.dump(cell_to_graph_mappings, f)
                
            with open(slide_mappings_path, 'wb') as f:  # 新增：保存切片到患者的映射
                pickle.dump(slide_to_patient_mapping, f)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # 统计信息
            total_slides = len(split_data)
            slides_with_graphs = sum([status for status in graph_status.values()])
            slides_without_graphs = total_slides - slides_with_graphs
            unique_patients = len(set(slide_to_patient_mapping.values()))
            
            print(f"{split}集保存完成:")
            print(f"  - 总切片数: {total_slides}")
            print(f"  - 覆盖患者数: {unique_patients}")
            print(f"  - 有图数据切片: {slides_with_graphs}")
            print(f"  - 无图数据切片: {slides_without_graphs} (保留完整DINO特征)")
            print(f"  - Patch内图: {intra_path}")
            print(f"  - Patch间图: {inter_path}")
            print(f"  - Bulk表达: {bulk_path}")
            print(f"  - 细胞特征: {features_path}")
            print(f"  - 细胞坐标: {positions_path}")
            print(f"  - 聚类标签: {clusters_path}")
            print(f"  - 图状态: {status_path}")
            print(f"  - 细胞映射: {mappings_path}")
            print(f"  - 切片映射: {slide_mappings_path}")  # 新增
            print(f"  - 元数据: {metadata_path}")
        
        # 保存全局配置
        config = {
            'feature_dim': self.feature_dim,
            'intra_patch_distance_threshold': self.intra_patch_distance_threshold,
            'inter_patch_k_neighbors': self.inter_patch_k_neighbors,
            'use_deep_features': self.use_deep_features,
            'max_cells_per_patch': self.max_cells_per_patch,
            'num_genes': len(self.bulk_data.index) if self.bulk_data is not None else 0,
            'gene_names': self.bulk_data.index.tolist() if self.bulk_data is not None else [],
            'patches_dir': self.patches_dir,
            'wsi_input_dir': self.wsi_input_dir,
            'supports_no_graph_patients': True,
            'uses_dino_files_directly': True,
            'preserves_cluster_labels': True,
            'uses_slide_level_matching': True,  # 新增：标记使用切片级别匹配
            'allows_multiple_slides_per_patient': True  # 新增：允许一个患者多个切片
        }
        
        config_path = os.path.join(output_dir, "bulk_graph_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n配置文件: {config_path}")
        return metadata
        """保存构建的图数据 - 新逻辑：保存完整的细胞特征数据"""
        print("=== 保存图结构和完整细胞数据 ===")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 分别保存训练集和测试集的图数据
        for split in ['train', 'test']:
            if split not in self.processed_data:
                continue
                
            split_data = self.processed_data[split]
            
            # 准备保存的数据
            intra_graphs = {}
            inter_graphs = {}
            bulk_expressions = {}
            all_cell_features = {}       # 所有细胞的DINO特征
            all_cell_positions = {}      # 所有细胞的空间坐标  
            cluster_labels = {}          # 所有细胞的聚类标签
            graph_status = {}            # 每个患者是否有图数据的状态
            cell_to_graph_mappings = {}  # 细胞到图的映射关系
            metadata = {}
            
            for patient_id, patient_data in split_data.items():
                intra_graphs[patient_id] = patient_data['intra_patch_graphs']
                inter_graphs[patient_id] = patient_data['inter_patch_graph']
                bulk_expressions[patient_id] = patient_data['bulk_expr']
                all_cell_features[patient_id] = patient_data['all_cell_features']
                all_cell_positions[patient_id] = patient_data['all_cell_positions']
                cluster_labels[patient_id] = patient_data['cluster_labels']
                graph_status[patient_id] = patient_data.get('has_graphs', False)
                cell_to_graph_mappings[patient_id] = patient_data.get('cell_to_graph_mapping', None)
                
                metadata[patient_id] = {
                    'num_cells': len(patient_data['cells_df']),
                    'num_patches': len(patient_data['patches']),
                    'intra_graph_count': len(patient_data['intra_patch_graphs']),
                    'inter_graph_edges': patient_data['inter_patch_graph'].edge_index.shape[1],
                    'has_bulk_expr': patient_data['bulk_expr'] is not None,
                    'has_graphs': patient_data.get('has_graphs', False),
                    'total_cell_features': patient_data['all_cell_features'].shape[0],
                    'cell_feature_dim': patient_data['all_cell_features'].shape[1]
                }
            
            # 保存文件
            intra_path = os.path.join(output_dir, f"bulk_{split}_intra_patch_graphs.pkl")
            inter_path = os.path.join(output_dir, f"bulk_{split}_inter_patch_graphs.pkl")
            bulk_path = os.path.join(output_dir, f"bulk_{split}_expressions.pkl")
            features_path = os.path.join(output_dir, f"bulk_{split}_all_cell_features.pkl")
            positions_path = os.path.join(output_dir, f"bulk_{split}_all_cell_positions.pkl")
            clusters_path = os.path.join(output_dir, f"bulk_{split}_cluster_labels.pkl")
            status_path = os.path.join(output_dir, f"bulk_{split}_graph_status.pkl")
            mappings_path = os.path.join(output_dir, f"bulk_{split}_cell_to_graph_mappings.pkl")
            metadata_path = os.path.join(output_dir, f"bulk_{split}_metadata.json")
            
            with open(intra_path, 'wb') as f:
                pickle.dump(intra_graphs, f)
            
            with open(inter_path, 'wb') as f:
                pickle.dump(inter_graphs, f)
            
            with open(bulk_path, 'wb') as f:
                pickle.dump(bulk_expressions, f)
                
            with open(features_path, 'wb') as f:
                pickle.dump(all_cell_features, f)
                
            with open(positions_path, 'wb') as f:
                pickle.dump(all_cell_positions, f)
                
            with open(clusters_path, 'wb') as f:
                pickle.dump(cluster_labels, f)
                
            with open(status_path, 'wb') as f:
                pickle.dump(graph_status, f)
                
            with open(mappings_path, 'wb') as f:
                pickle.dump(cell_to_graph_mappings, f)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # 统计信息
            total_patients = len(split_data)
            patients_with_graphs = sum([status for status in graph_status.values()])
            patients_without_graphs = total_patients - patients_with_graphs
            
            print(f"{split}集保存完成:")
            print(f"  - 总患者数: {total_patients}")
            print(f"  - 有图数据患者: {patients_with_graphs}")
            print(f"  - 无图数据患者: {patients_without_graphs} (保留完整DINO特征)")
            print(f"  - Patch内图: {intra_path}")
            print(f"  - Patch间图: {inter_path}")
            print(f"  - Bulk表达: {bulk_path}")
            print(f"  - 细胞特征: {features_path}")
            print(f"  - 细胞坐标: {positions_path}")
            print(f"  - 聚类标签: {clusters_path}")
            print(f"  - 图状态: {status_path}")
            print(f"  - 细胞映射: {mappings_path}")
            print(f"  - 元数据: {metadata_path}")
        
        # 保存全局配置
        config = {
            'feature_dim': self.feature_dim,
            'intra_patch_distance_threshold': self.intra_patch_distance_threshold,
            'inter_patch_k_neighbors': self.inter_patch_k_neighbors,
            'use_deep_features': self.use_deep_features,
            'max_cells_per_patch': self.max_cells_per_patch,
            'num_genes': len(self.bulk_data.index) if self.bulk_data is not None else 0,
            'gene_names': self.bulk_data.index.tolist() if self.bulk_data is not None else [],
            'patches_dir': self.patches_dir,
            'wsi_input_dir': self.wsi_input_dir,
            'supports_no_graph_patients': True,
            'uses_dino_files_directly': True,  # 新增：标记直接使用DINO文件
            'preserves_cluster_labels': True   # 新增：标记保留聚类标签
        }
        
        config_path = os.path.join(output_dir, "bulk_graph_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n配置文件: {config_path}")
        return metadata


def main():
    """主函数"""
    
    # 配置参数
    train_features_dir = "/data/yujk/hovernet2feature/hovernet2feature/extracted_features_dino_tile36_train"
    test_features_dir = "/data/yujk/hovernet2feature/hovernet2feature/extracted_features_dino_tile36_test"
    bulk_csv_path = "/data/yujk/hovernet2feature/basic_model/tpm-TCGA-COAD_intersection_million.csv"
    patches_dir = "/data/yujk/hovernet2feature/output_patches"
    wsi_input_dir = "/data/yujk/hovernet2feature/WSI/COAD"
    output_dir = "/data/yujk/hovernet2feature/bulk_static_graphs_new_all_graph"
    
    # 图构建参数（方案3：提升GPU利用率的新参数）
    intra_patch_distance_threshold = 256   # patch内细胞连接距离阈值（像素）- 从250增加到256
    inter_patch_k_neighbors = 8           # patch间k近邻数量 - 从6增加到8
    use_deep_features = True              # 使用深度特征
    feature_dim = 128                     # 特征维度
    max_cells_per_patch = None           # 每个patch的最大细胞数 - 不限制
    
    print("=== Bulk数据集静态图构建（使用预分割patch）- 新逻辑版本 ===")
    print(f"训练特征目录: {train_features_dir}")
    print(f"测试特征目录: {test_features_dir}")
    print(f"Bulk数据文件: {bulk_csv_path}")
    print(f"Patch目录: {patches_dir}")
    print(f"WSI输入目录: {wsi_input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"配置参数:")
    print(f"  - Patch内距离阈值: {intra_patch_distance_threshold}px")
    print(f"  - Patch间k近邻: {inter_patch_k_neighbors}")
    print(f"  - 使用深度特征: {use_deep_features}")
    print(f"  - 特征维度: {feature_dim}")
    print(f"  - 每patch最大细胞数: {max_cells_per_patch}")
    
    # 检查输入目录
    for path, name in [(train_features_dir, "训练特征目录"), (test_features_dir, "测试特征目录"), 
                       (bulk_csv_path, "Bulk数据文件"), (patches_dir, "Patch目录"), (wsi_input_dir, "WSI输入目录")]:
        if not os.path.exists(path):
            print(f"错误: {name}不存在: {path}")
            return
    
    # 创建图构建器
    try:
        builder = BulkStaticGraphBuilder(
            train_features_dir=train_features_dir,
            test_features_dir=test_features_dir,
            bulk_csv_path=bulk_csv_path,
            patches_dir=patches_dir,
            wsi_input_dir=wsi_input_dir,
            intra_patch_distance_threshold=intra_patch_distance_threshold,
            inter_patch_k_neighbors=inter_patch_k_neighbors,
            use_deep_features=use_deep_features,
            feature_dim=feature_dim,
            max_cells_per_patch=max_cells_per_patch
        )
        
        # 加载bulk数据
        builder.load_bulk_data()
        
        # 处理所有切片 - 使用切片级别匹配逻辑
        builder.process_all_slides_new_logic()
        
        # 构建并保存图 - 使用切片级别保存逻辑
        metadata = builder.save_graphs_slide_logic(output_dir)
        
        print("\n=== 图构建完成（切片级别匹配，0%数据丢失版本）===")
        for split in ['train', 'test']:
            total_slides = len(builder.processed_data.get(split, {}))
            print(f"{split}集:")
            print(f"  - 切片数: {total_slides}")
            if total_slides > 0:
                # 从processed_data中计算统计信息
                split_slides = builder.processed_data.get(split, {})
                if split_slides:
                    avg_cells = np.mean([len(s['cells_df']) for s in split_slides.values()])
                    avg_patches = np.mean([len(s['patches']) for s in split_slides.values()])
                    has_graphs_count = sum([1 for s in split_slides.values() if s.get('has_graphs', False)])
                    no_graphs_count = total_slides - has_graphs_count
                    unique_patients = len(set([s['patient_id'] for s in split_slides.values()]))
                    print(f"  - 覆盖患者数: {unique_patients}")
                    print(f"  - 平均细胞数/切片: {avg_cells:.0f}")
                    print(f"  - 平均patch数/切片: {avg_patches:.1f}")
                    print(f"  - 有图切片: {has_graphs_count}")
                    print(f"  - 无图切片: {no_graphs_count} (保留完整DINO特征)")
        
        print("\n✅ 完成：实现切片级别精确匹配，支持混合处理（有图增强 + 无图原始特征）")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()