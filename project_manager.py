import json
import os
import time
import uuid

PROJECT_DB_FILE = "projects.json"

class ProjectManager:
    def __init__(self):
        self.db_file = PROJECT_DB_FILE
        self._ensure_db()

    def _ensure_db(self):
        if not os.path.exists(self.db_file):
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def _load_data(self):
        with open(self.db_file, 'r', encoding='utf-8') as f:
            return json.load(f)

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

    def create(self, title, script, publish_time=""):
        data = self._load_data()
        new_project = {
            "id": str(uuid.uuid4())[:8],
            "title": title,
            "script": script,
            "cover_path": "", # 封面图路径
            "video_path": "",         # 成片路径
            "status": "draft",        # draft(草稿), generated(已生成), published(已发布)
            "publish_time": publish_time,       # 计划发布时间
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
                # 如果已发布，仅允许更新特定字段(如状态回退)，这里做简单限制
                if item['status'] == 'published' and 'status' not in update_dict:
                    return False # 禁止修改已发布内容
                
                item.update(update_dict)
                self._save_data(data)
                return True
        return False

    def delete(self, project_id):
        data = self._load_data()
        new_data = [d for d in data if d['id'] != project_id]
        self._save_data(new_data)
        return True

project_mgr = ProjectManager()