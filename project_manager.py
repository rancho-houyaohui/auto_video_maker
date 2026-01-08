import json
import os
import time
import uuid
import config  # [核心修改] 引入配置以获取用户文档路径

# [核心修改] 数据库文件必须存在用户文档目录下，不能存放在 app 内部
PROJECT_DB_FILE = os.path.join(config.USER_DOCS, "projects.json")

class ProjectManager:
    def __init__(self):
        self.db_file = PROJECT_DB_FILE
        self._ensure_db()

    def _ensure_db(self):
        # 确保父目录存在 (虽然 config.py 里做了，但这层防御很必要)
        parent_dir = os.path.dirname(self.db_file)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        if not os.path.exists(self.db_file):
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def _load_data(self):
        # 增加容错，防止 JSON 损坏导致程序打不开
        try:
            with open(self.db_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_data(self, data):
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_all(self):
        data = self._load_data()
        # 按创建时间倒序
        return sorted(data, key=lambda x: x.get('created_at', 0), reverse=True)

    def get_one(self, project_id):
        data = self._load_data()
        for item in data:
            if item['id'] == project_id:
                return item
        return None

    def create(self, title, script, cover_path=""):
        data = self._load_data()
        new_project = {
            "id": str(uuid.uuid4())[:8],
            "title": title,
            "script": script,
            "cover_path": cover_path, # 封面图路径
            "video_path": "",         # 成片路径
            "status": "draft",        # draft(草稿), generated(已生成), published(已发布)
            "publish_time": "",       # 计划发布时间
            "created_at": time.time(),
            "scenes_data": []         # 暂存分镜数据
        }
        data.append(new_project)
        self._save_data(data)
        return new_project

    def update(self, project_id, update_dict):
        data = self._load_data()
        for item in data:
            if item['id'] == project_id:
                # 状态简单流转逻辑
                if item['status'] == 'published' and 'status' not in update_dict:
                    # 允许更新，但在业务层可能会限制，这里暂放开
                    pass 
                
                item.update(update_dict)
                self._save_data(data)
                return True
        return False

    def delete(self, project_id):
        data = self._load_data()
        new_data = [d for d in data if d['id'] != project_id]
        self._save_data(new_data)
        return True

# 初始化实例
project_mgr = ProjectManager()